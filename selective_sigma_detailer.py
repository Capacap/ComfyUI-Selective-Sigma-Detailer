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


def _mask_delta(denoised, x, sigma, state, p):
    prev = state.get("prev_denoised")
    state["prev_denoised"] = denoised.detach()
    if prev is None:
        return None
    raw = (denoised - prev).abs().mean(dim=1, keepdim=True)
    m = normalize_mask(raw, p["clip_percentile"])
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


class SelectiveSigmaDetailerDeltaV2Node:
    DESCRIPTION = (
        "Masks using |denoised_t - denoised_{t-1}| with percentile-clipped "
        "normalization. Per-step sigma adjustment is divided by the count of "
        "active schedule steps so detail_amount is roughly step-count "
        "invariant. First active step is a no-op (no previous prediction)."
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
                "detail_amount": ("FLOAT", {"default": 5.0, "min": -100.0, "max": 100.0, "step": 0.1}),
                "start": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ema": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend with previous mask. 0 = per-step, higher = stronger temporal smoothing."}),
                "mask_clip_percentile": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.49, "step": 0.005,
                    "tooltip": "Clip the top/bottom fraction of delta values before min/max stretch. 0 = pure min/max, higher = stronger outlier rejection."}),
            }
        }

    def go(self, sampler, detail_amount, start, end, ema, mask_clip_percentile):
        def schedule_fn(steps):
            return make_schedule(steps, start, end, detail_amount)

        mask_params = {"ema": ema, "clip_percentile": mask_clip_percentile}
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
