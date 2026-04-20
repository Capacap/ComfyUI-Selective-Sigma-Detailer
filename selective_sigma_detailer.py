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
    make_schedule,
    mask_to_preview_image,
    postprocess_mask,
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


def _mask_delta(denoised, x, sigma, state, p):
    prev = state.get("prev_denoised")
    state["prev_denoised"] = denoised.detach()
    if prev is None:
        return None
    raw = (denoised - prev).abs().mean(dim=1, keepdim=True)
    m = postprocess_mask(
        raw, p["blur"], p["threshold"], p["gamma"],
        clip_percentile=p.get("clip_percentile", 0.0),
    )
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


class _SSDBase:
    CATEGORY = "sampling/custom_sampling/samplers"
    RETURN_TYPES = ("SAMPLER", "SSD_MASK_REF")
    RETURN_NAMES = ("sampler", "mask_ref")
    FUNCTION = "go"

    MASK_FN = None
    NORMALIZE_BY_ACTIVE_STEPS = False
    MASK_CLIP_PERCENTILE = 0.0

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
        clip_percentile = kwargs.pop("mask_clip_percentile", self.MASK_CLIP_PERCENTILE)
        mask_params = {**mask_common, **kwargs, "clip_percentile": clip_percentile}
        mask_ref = {}
        ksampler = build_sampler(
            wrapped_sampler=sampler,
            make_schedule_fn=_schedule_closure(sched_kwargs),
            cfg_scale_override=sched_kwargs["cfg_scale_override"],
            mask_fn=self.MASK_FN,
            mask_params=mask_params,
            mask_ref=mask_ref,
            normalize_by_active_steps=self.NORMALIZE_BY_ACTIVE_STEPS,
        )
        return (ksampler, mask_ref)


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


class SelectiveSigmaDetailerDeltaV2Node(_SSDBase):
    DESCRIPTION = (
        "Experimental delta variant. Divides the per-step sigma adjustment by "
        "the active-step count so the integrated detail push is roughly "
        "invariant to total step count. Expect detail_amount to need a larger "
        "value (order ~N_active) than the v1 node for comparable output."
    )
    MASK_FN = staticmethod(_mask_delta)
    NORMALIZE_BY_ACTIVE_STEPS = True
    MASK_CLIP_PERCENTILE = 0.1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **cls._base_inputs(),
                "ema": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend with previous mask. 0 = per-step, 1 = lock the first computed mask."}),
                "mask_clip_percentile": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.49, "step": 0.005,
                    "tooltip": "Clip the top/bottom fraction of delta values before min/max stretch. 0 = pure min/max, higher = stronger outlier rejection."}),
            }
        }


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
