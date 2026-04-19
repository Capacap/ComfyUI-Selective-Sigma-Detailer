from __future__ import annotations

import os
import time

import numpy as np
import torch
from PIL import Image
import folder_paths

from .ssd_core import (
    MASK_COMMON_INPUTS,
    SCHEDULE_INPUTS,
    build_sampler,
    local_variance,
    make_schedule,
    mask_to_preview_image,
    postprocess_mask,
    sobel_magnitude,
)


def _schedule_closure(kw):
    def inner(steps):
        return make_schedule(
            steps, kw["start"], kw["end"], kw["bias"], kw["detail_amount"],
            kw["exponent"], kw["start_offset"], kw["end_offset"],
            kw["fade"], kw["smooth"],
        )
    return inner


def _pop_schedule_kwargs(kwargs):
    return {k: kwargs.pop(k) for k in list(SCHEDULE_INPUTS.keys())}


def _pop_mask_common(kwargs):
    return {
        "blur": kwargs.pop("mask_blur"),
        "threshold": kwargs.pop("mask_threshold"),
        "gamma": kwargs.pop("mask_gamma"),
    }


def _mask_variance_snapshot(denoised, x, sigma, state, p):
    if "mask" in state:
        return state["mask"]
    raw = local_variance(denoised, p["variance_kernel"])
    m = postprocess_mask(raw, p["blur"], p["threshold"], p["gamma"])
    state["mask"] = m
    return m


def _mask_variance_dynamic(denoised, x, sigma, state, p):
    raw = local_variance(denoised, p["variance_kernel"])
    m = postprocess_mask(raw, p["blur"], p["threshold"], p["gamma"])
    prev = state.get("mask")
    ema = p["ema"]
    if prev is not None and ema > 0:
        if prev.shape != m.shape:
            prev = torch.nn.functional.interpolate(
                prev, size=m.shape[-2:], mode="bilinear", align_corners=False,
            )
        m = ema * prev + (1 - ema) * m
    state["mask"] = m
    return m


def _mask_delta(denoised, x, sigma, state, p):
    prev = state.get("prev_denoised")
    state["prev_denoised"] = denoised.detach()
    if prev is None:
        return None
    raw = (denoised - prev).abs().mean(dim=1, keepdim=True)
    m = postprocess_mask(raw, p["blur"], p["threshold"], p["gamma"])
    prev_mask = state.get("mask")
    ema = p["ema"]
    if prev_mask is not None and ema > 0:
        if prev_mask.shape != m.shape:
            prev_mask = torch.nn.functional.interpolate(
                prev_mask, size=m.shape[-2:], mode="bilinear", align_corners=False,
            )
        m = ema * prev_mask + (1 - ema) * m
    state["mask"] = m
    return m


def _mask_edges(denoised, x, sigma, state, p):
    raw = sobel_magnitude(denoised)
    m = postprocess_mask(raw, p["blur"], p["threshold"], p["gamma"])
    state["mask"] = m
    return m


class _SSDBase:
    CATEGORY = "sampling/custom_sampling/samplers"
    RETURN_TYPES = ("SAMPLER", "SSD_MASK_REF")
    RETURN_NAMES = ("sampler", "mask_ref")
    FUNCTION = "go"

    MASK_FN = None  # set by subclass

    @classmethod
    def _base_inputs(cls):
        return {
            "sampler": ("SAMPLER",),
            **SCHEDULE_INPUTS,
            **MASK_COMMON_INPUTS,
        }

    def go(self, sampler, **kwargs):
        sched_kwargs = _pop_schedule_kwargs(kwargs)
        mask_common = _pop_mask_common(kwargs)
        mask_params = {**mask_common, **kwargs}
        mask_ref = {}
        ksampler = build_sampler(
            wrapped_sampler=sampler,
            make_schedule_fn=_schedule_closure(sched_kwargs),
            cfg_scale_override=sched_kwargs["cfg_scale_override"],
            mask_fn=self.MASK_FN,
            mask_params=mask_params,
            mask_ref=mask_ref,
        )
        return (ksampler, mask_ref)


class SelectiveSigmaDetailerNode(_SSDBase):
    DESCRIPTION = (
        "Boosts detail in dense regions using a frozen variance mask snapshot "
        "taken at the first active step. Costs 2x model calls during active steps."
    )
    MASK_FN = staticmethod(_mask_variance_snapshot)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **cls._base_inputs(),
                "variance_kernel": ("INT", {"default": 5, "min": 3, "max": 15, "step": 2,
                    "tooltip": "Window size for local variance. Larger = coarser density estimate."}),
            }
        }


class SelectiveSigmaDetailerDynamicVarianceNode(_SSDBase):
    DESCRIPTION = (
        "Like the variance variant, but recomputes the mask each active step "
        "with an EMA blend so the mask follows composition as it develops."
    )
    MASK_FN = staticmethod(_mask_variance_dynamic)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **cls._base_inputs(),
                "variance_kernel": ("INT", {"default": 5, "min": 3, "max": 15, "step": 2,
                    "tooltip": "Window size for local variance. Larger = coarser density estimate."}),
                "ema": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend with previous mask. 0 = fully dynamic, 1 = fully frozen at first step."}),
            }
        }


class SelectiveSigmaDetailerDeltaNode(_SSDBase):
    DESCRIPTION = (
        "Masks using |denoised_t - denoised_{t-1}|: targets regions the model "
        "is still actively refining between steps. First active step is a "
        "no-op (no previous prediction); detailing begins on the second."
    )
    MASK_FN = staticmethod(_mask_delta)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **cls._base_inputs(),
                "ema": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend with previous mask. 0 = per-step, 1 = lock the first computed mask."}),
            }
        }


class SelectiveSigmaDetailerEdgesNode(_SSDBase):
    DESCRIPTION = (
        "Masks using Sobel gradient magnitude on the denoised prediction: "
        "targets contours and silhouettes rather than textured fields."
    )
    MASK_FN = staticmethod(_mask_edges)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": cls._base_inputs()}


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
