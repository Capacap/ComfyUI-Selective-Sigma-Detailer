from __future__ import annotations

import os
import time

import numpy as np
import torch
from PIL import Image
import folder_paths

from .ssd_core import (
    build_sampler,
    make_schedule,
    mask_to_preview_image,
    normalize_mask,
)


_MASK_CLIP_PERCENTILE = 0.2
_SCHEDULE_START = 0.2
_MASK_EMA = 0.9


def _mask_delta(denoised, x, sigma, state, p):
    prev = state.get("prev_denoised")
    state["prev_denoised"] = denoised.detach()
    if prev is None:
        return None
    raw = (denoised - prev).abs().mean(dim=1, keepdim=True)
    m = normalize_mask(raw, _MASK_CLIP_PERCENTILE)
    prev_mask = state.get("mask")
    if prev_mask is not None:
        if prev_mask.shape != m.shape:
            prev_mask = torch.nn.functional.interpolate(
                prev_mask, size=m.shape[-2:], mode="bilinear", align_corners=False,
            )
        m = _MASK_EMA * prev_mask + (1 - _MASK_EMA) * m
    state["mask"] = m
    shift = 2 * (p["coverage"] - 0.5)
    if shift != 0:
        m = (m + shift).clamp(0, 1)
    return m


class SelectiveSigmaDetailerDeltaV2Node:
    DESCRIPTION = (
        "Masks using |denoised_t - denoised_{t-1}| with percentile-clipped "
        "normalization. Coverage shifts the mask threshold: 0 empty, 0.5 "
        "normal, 1.0 full. Per-step sigma adjustment is normalized by active "
        "step count so intensity is roughly step-count invariant."
    )
    CATEGORY = "sampling/custom_sampling/samplers"
    RETURN_TYPES = ("SAMPLER", "SSD_MASK_REF")
    RETURN_NAMES = ("sampler", "mask_ref")
    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "intensity": ("FLOAT", {"default": 2.0, "min": -100.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Strength of the sigma adjustment on masked regions."}),
                "coverage": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0 = empty mask (no effect), 0.5 = normal delta mask, 1.0 = full mask (applied everywhere)."}),
            }
        }

    def go(self, sampler, intensity, coverage):
        def schedule_fn(steps):
            return make_schedule(steps, _SCHEDULE_START, intensity)

        mask_params = {"coverage": coverage}
        mask_ref = {}
        ksampler = build_sampler(
            wrapped_sampler=sampler,
            make_schedule_fn=schedule_fn,
            mask_fn=_mask_delta,
            mask_params=mask_params,
            mask_ref=mask_ref,
            normalize_by_active_steps=True,
        )
        return (ksampler, mask_ref)


_REFERENCE_ACTIVE_STEPS = 16


class SelectiveSigmaDetailerPatchModelNode:
    DESCRIPTION = (
        "Model-patch variant. Attaches the mask-and-blend logic to the model "
        "itself via set_model_unet_function_wrapper, so any sampler node "
        "(KSampler, KSamplerAdvanced, SamplerCustom) picks it up. Progress "
        "is measured in sigma-space since the full schedule is not visible "
        "at the model layer; intensity is calibrated against a 16-active-step "
        "reference, so magnitude may need retuning on very short or long runs."
    )
    CATEGORY = "sampling/custom_sampling/samplers"
    RETURN_TYPES = ("MODEL", "SSD_MASK_REF")
    RETURN_NAMES = ("model", "mask_ref")
    FUNCTION = "patch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "intensity": ("FLOAT", {"default": 2.0, "min": -100.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Strength of the sigma adjustment on masked regions."}),
                "coverage": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0 = empty mask (no effect), 0.5 = normal delta mask, 1.0 = full mask (applied everywhere)."}),
            }
        }

    def patch(self, model, intensity, coverage):
        mask_ref = {}
        state = {}
        per_step_adjustment = intensity * 0.1 / _REFERENCE_ACTIVE_STEPS

        def wrapper(apply_model, options):
            x = options["input"]
            sigma = options["timestep"]
            c = options["c"]

            sigma_float = float(sigma.max().detach().cpu())
            sigma_max = state.get("sigma_max", 0.0)
            if sigma_float > sigma_max * 1.001 or sigma_max == 0.0:
                state.clear()
                state["sigma_max"] = sigma_float
                sigma_max = sigma_float

            denoised_normal = apply_model(x, sigma, **c)

            progress = 1.0 - sigma_float / sigma_max if sigma_max > 0 else 0.0
            if progress < _SCHEDULE_START:
                state["prev_denoised"] = denoised_normal.detach()
                return denoised_normal

            prev = state.get("prev_denoised")
            state["prev_denoised"] = denoised_normal.detach()
            if prev is None or prev.shape != denoised_normal.shape:
                return denoised_normal

            raw = (denoised_normal - prev).abs().mean(dim=1, keepdim=True)
            m = normalize_mask(raw, _MASK_CLIP_PERCENTILE)
            prev_mask = state.get("mask")
            if prev_mask is not None and prev_mask.shape == m.shape:
                m = _MASK_EMA * prev_mask + (1 - _MASK_EMA) * m
            state["mask"] = m

            shift = 2 * (coverage - 0.5)
            if shift <= -1.0:
                return denoised_normal
            if shift != 0:
                m = (m + shift).clamp(0, 1)

            mask_ref["mask"] = m.detach()

            adjusted_sigma = sigma * max(1e-6, 1.0 - per_step_adjustment)
            denoised_detailed = apply_model(x, adjusted_sigma, **c)
            m_match = m.to(denoised_normal)
            return denoised_normal * (1 - m_match) + denoised_detailed * m_match

        patched = model.clone()
        patched.set_model_unet_function_wrapper(wrapper)
        return (patched, mask_ref)


class SelectiveSigmaDetailerMaskPreviewNode:
    DESCRIPTION = (
        "Displays the mask captured during sampling. Place between a Selective "
        "Sigma Detailer sampler's mask_ref output and downstream latent use; "
        "the latent passthrough forces this node to run after sampling."
    )
    CATEGORY = "sampling/custom_sampling/samplers"
    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("preview", "latent")
    FUNCTION = "preview"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_ref": ("SSD_MASK_REF",),
                "latent": ("LATENT",),
                "upscale": ("INT", {"default": 8, "min": 1, "max": 16, "step": 1,
                    "tooltip": "Nearest-neighbor upscale factor for the preview image."}),
            }
        }

    def preview(self, mask_ref, latent, upscale):
        mask = mask_ref.get("mask") if isinstance(mask_ref, dict) else None
        if mask is None:
            img = torch.zeros(1, 64, 64, 3)
        else:
            img = mask_to_preview_image(mask, upscale=upscale)

        out_dir = folder_paths.get_temp_directory()
        os.makedirs(out_dir, exist_ok=True)
        filename = f"ssd_mask_{int(time.time() * 1000)}.png"
        path = os.path.join(out_dir, filename)
        arr = (img[0].numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).save(path, compress_level=1)

        return {
            "ui": {"images": [{"filename": filename, "subfolder": "", "type": "temp"}]},
            "result": (img, latent),
        }
