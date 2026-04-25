from __future__ import annotations

import glob
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
_SCHEDULE_START = 0.1
_SCHEDULE_END = 0.9
_MASK_EMA = 0.9


_MASK_OVERRIDES = ("off", "ones", "half", "zeros")


def _mask_delta(denoised, x, sigma, state, p):
    """Build a detail mask from the change between consecutive denoised predictions.

    Regions where the model's prediction is moving step-to-step are regions
    still gaining structure, which is where the extra detail pass is useful.
    Flat/settled regions produce a small delta and get left alone.

    coverage semantics: an additive threshold shift applied AFTER the EMA and
    AFTER writing the smoothed mask back to state, so the shift doesn't
    compound through the feedback loop. 0.5 -> raw normalized delta, 1.0 ->
    saturates to 1 everywhere. (coverage <= 0 is short-circuited at the node
    boundary and never reaches here.)
    """
    # Debug override: force the mask to a constant so the blend path can be
    # compared against the coverage=1 fast path (ones), the pure baseline
    # (zeros), or a flat mid-value (half). Bypasses coverage shift and EMA.
    override = p.get("mask_override", "off")
    if override != "off":
        shape = (denoised.shape[0], 1, denoised.shape[2], denoised.shape[3])
        fill = {"ones": 1.0, "half": 0.5, "zeros": 0.0}[override]
        return torch.full(shape, fill, device=denoised.device, dtype=denoised.dtype)

    prev = state.get("prev_denoised")
    state["prev_denoised"] = denoised.detach()
    # First step has no prior frame to diff against.
    if prev is None:
        return None
    # Channel-mean of the absolute delta collapses the 4-channel latent into a
    # single-channel activity heatmap.
    raw = (denoised - prev).abs().mean(dim=1, keepdim=True)
    m = normalize_mask(raw, p["clip_percentile"])
    prev_mask = state.get("mask")
    ema = p["ema"]
    if prev_mask is not None and ema > 0:
        # Upscaling samplers can change latent resolution mid-run; resize the
        # carried-over mask rather than dropping it.
        if prev_mask.shape != m.shape:
            prev_mask = torch.nn.functional.interpolate(
                prev_mask, size=m.shape[-2:], mode="bilinear", align_corners=False,
            )
        m = ema * prev_mask + (1 - ema) * m
    # Store the pre-shift mask so the coverage offset doesn't compound through
    # the EMA feedback on subsequent steps.
    state["mask"] = m
    shift = 2 * (p["coverage"] - 0.5)
    if shift != 0:
        m = (m + shift).clamp(0, 1)
    return m


class SelectiveSigmaDetailerNode:
    DESCRIPTION = (
        "Masks using |denoised_t - denoised_{t-1}| with percentile-clipped "
        "normalization. Coverage shifts the mask threshold: 0 empty, 0.5 "
        "normal, 1.0 full (skips the normal pass). Strength is the peak "
        "per-step fraction of sigma removed during the detail pass."
    )
    CATEGORY = "sampling/custom_sampling/samplers"
    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "strength": ("FLOAT", {"default": 0.1, "min": -1.0, "max": 1.0, "step": 0.005,
                    "tooltip": "Peak per-step fraction of sigma removed on masked regions. 0.1 = -10% sigma at peak."}),
                "coverage": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0 = empty mask (no effect), 0.5 = normal delta mask, 1.0 = full mask (applied everywhere, skips the normal pass)."}),
            }
        }

    def go(self, sampler, strength, coverage):
        # No path through the wrapper can produce a different result than the
        # input sampler — skip wrapping entirely so there's no per-call
        # overhead and no misleading [SSD] log line.
        if strength == 0.0 or coverage <= 0.0:
            return (sampler,)

        def schedule_fn(steps):
            return make_schedule(steps, _SCHEDULE_START, _SCHEDULE_END, strength)

        mask_params = {
            "coverage": coverage,
            "ema": _MASK_EMA,
            "clip_percentile": _MASK_CLIP_PERCENTILE,
        }
        ksampler = build_sampler(
            wrapped_sampler=sampler,
            make_schedule_fn=schedule_fn,
            mask_fn=_mask_delta,
            mask_params=mask_params,
            mask_ref={},
        )
        return (ksampler,)


class SelectiveSigmaDetailerDebugNode:
    DESCRIPTION = (
        "Debug variant of the SAMPLER node. Exposes the hardcoded constants "
        "(schedule start, ema, mask clip percentile) and the mask_ref output "
        "so the mask can be inspected via the debug preview node. Defaults "
        "match the main node's internal values; change them only to "
        "experiment or diagnose unexpected behavior."
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
                "strength": ("FLOAT", {"default": 0.1, "min": -1.0, "max": 1.0, "step": 0.005}),
                "coverage": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "start": ("FLOAT", {"default": _SCHEDULE_START, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Fraction of schedule to skip before applying detail. At least 1 step is always skipped."}),
                "end": ("FLOAT", {"default": _SCHEDULE_END, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Fraction of schedule at which the tail taper hits zero. At least 1 step at the end is always clean."}),
                "ema": ("FLOAT", {"default": _MASK_EMA, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Temporal mask smoothing. 0 = per-step, higher = stronger carryover across steps."}),
                "mask_clip_percentile": ("FLOAT", {"default": _MASK_CLIP_PERCENTILE, "min": 0.0, "max": 0.49, "step": 0.005,
                    "tooltip": "Clip the top/bottom fraction of delta values before min/max stretch."}),
                "mask_override": (list(_MASK_OVERRIDES), {"default": "off",
                    "tooltip": "Force the mask to a constant for blend-path diagnostics. 'ones' at coverage<1 should match the coverage=1 fast path; 'zeros' should match baseline; 'half' isolates pure x0 blending."}),
            }
        }

    def go(self, sampler, strength, coverage, start, end, ema, mask_clip_percentile, mask_override):
        # Same short-circuit as the main node. Populate mask_ref so the
        # preview node downstream renders the conceptually-correct mask:
        # full white at coverage>=1 (matches the non-zero-strength full
        # coverage fast path), zeros otherwise. Real latent dims aren't
        # known here since sampling never runs; the preview upscales the
        # placeholder to a viewable size.
        if strength == 0.0 or coverage <= 0.0:
            if coverage >= 1.0 and strength == 0.0:
                return (sampler, {"mask": torch.ones(1, 1, 64, 64)})
            return (sampler, {})

        def schedule_fn(steps):
            return make_schedule(steps, start, end, strength)

        mask_params = {
            "coverage": coverage,
            "ema": ema,
            "clip_percentile": mask_clip_percentile,
            "mask_override": mask_override,
        }
        mask_ref = {}
        ksampler = build_sampler(
            wrapped_sampler=sampler,
            make_schedule_fn=schedule_fn,
            mask_fn=_mask_delta,
            mask_params=mask_params,
            mask_ref=mask_ref,
        )
        return (ksampler, mask_ref)


class SelectiveSigmaDetailerMaskPreviewNode:
    DESCRIPTION = (
        "Debug preview for the mask captured during sampling. Pairs with the "
        "Debug sampler's mask_ref output. The latent passthrough forces this "
        "node to run after sampling. Exposes the rendered mask as an IMAGE "
        "output for downstream wiring (e.g. composing into a comparison grid)."
    )
    CATEGORY = "sampling/custom_sampling/samplers"
    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "image")
    FUNCTION = "preview"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_ref": ("SSD_MASK_REF",),
                "latent": ("LATENT",),
            }
        }

    def preview(self, mask_ref, latent):
        mask = mask_ref.get("mask") if isinstance(mask_ref, dict) else None
        if mask is None:
            img = torch.zeros(1, 64, 64, 3)
        else:
            img = mask_to_preview_image(mask, upscale=8)

        out_dir = folder_paths.get_temp_directory()
        os.makedirs(out_dir, exist_ok=True)
        for old in glob.glob(os.path.join(out_dir, "ssd_mask_*.png")):
            try:
                os.remove(old)
            except OSError:
                pass
        filename = f"ssd_mask_{int(time.time() * 1000)}.png"
        path = os.path.join(out_dir, filename)
        arr = (img[0].numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).save(path, compress_level=1)

        return {
            "ui": {"images": [{"filename": filename, "subfolder": "", "type": "temp"}]},
            "result": (latent, img),
        }
