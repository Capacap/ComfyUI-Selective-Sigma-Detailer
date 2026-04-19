# Selective Sigma Detailer: boosts detail only in regions that are already
# dense (high local latent variance) at the moment the schedule activates.
# Runs early steps unmodified so composition settles, then on the first active
# step derives a density mask from the denoised prediction, freezes it, and
# for subsequent active steps blends a normal denoise with a sigma-shifted
# denoise weighted by that mask. Smooth regions stay smooth; busy regions
# get sharper.

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from comfy.samplers import KSAMPLER
from PIL import Image
import folder_paths

def _make_schedule(
    steps, start, end, bias, amount, exponent,
    start_offset, end_offset, fade, smooth,
):
    start = min(start, end)
    mid = start + bias * (end - start)
    multipliers = np.zeros(steps)

    start_idx, mid_idx, end_idx = [
        int(round(x * (steps - 1))) for x in [start, mid, end]
    ]

    start_values = np.linspace(0, 1, mid_idx - start_idx + 1)
    if smooth:
        start_values = 0.5 * (1 - np.cos(start_values * np.pi))
    start_values = start_values ** exponent
    if start_values.any():
        start_values *= amount - start_offset
        start_values += start_offset

    end_values = np.linspace(1, 0, end_idx - mid_idx + 1)
    if smooth:
        end_values = 0.5 * (1 - np.cos(end_values * np.pi))
    end_values = end_values ** exponent
    if end_values.any():
        end_values *= amount - end_offset
        end_values += end_offset

    multipliers[start_idx : mid_idx + 1] = start_values
    multipliers[mid_idx : end_idx + 1] = end_values
    multipliers[:start_idx] = start_offset
    multipliers[end_idx + 1 :] = end_offset
    multipliers *= 1 - fade
    return multipliers


def _sample_schedule(sigma, sigmas, schedule):
    sched_len = len(schedule)
    if sched_len < 2 or len(sigmas) < 2 or sigma <= 0 or not (sigmas[-1] <= sigma <= sigmas[0]):
        return 0.0
    deltas = (sigmas[:-1] - sigma).abs()
    idx = int(deltas.argmin())
    if (
        (idx == 0 and sigma >= sigmas[0])
        or (idx == sched_len - 1 and sigma <= sigmas[-2])
        or deltas[idx] == 0
    ):
        return schedule[idx].item()
    idxlow, idxhigh = (idx, idx - 1) if sigma > sigmas[idx] else (idx + 1, idx)
    nlow, nhigh = sigmas[idxlow], sigmas[idxhigh]
    if nhigh - nlow == 0:
        return schedule[idxlow]
    ratio = ((sigma - nlow) / (nhigh - nlow)).clamp(0, 1)
    return torch.lerp(schedule[idxlow], schedule[idxhigh], ratio).item()


def _save_mask_preview(mask: torch.Tensor, upscale: int = 8) -> str:
    m = mask[0, 0].detach().cpu().clamp(0, 1).numpy()
    img = (m * 255).astype(np.uint8)
    pil = Image.fromarray(img, mode="L")
    if upscale > 1:
        pil = pil.resize((pil.width * upscale, pil.height * upscale), Image.NEAREST)
    out_dir = folder_paths.get_temp_directory()
    os.makedirs(out_dir, exist_ok=True)
    filename = f"selective_sigma_detailer_mask_{int(time.time() * 1000)}.png"
    path = os.path.join(out_dir, filename)
    pil.save(path, compress_level=1)
    return path


def _compute_density_mask(
    denoised: torch.Tensor,
    variance_kernel: int,
    blur: int,
    threshold: float,
    gamma: float,
) -> torch.Tensor:
    k = variance_kernel | 1
    pad = k // 2
    mean = F.avg_pool2d(denoised, k, stride=1, padding=pad)
    var = F.avg_pool2d((denoised - mean).pow(2), k, stride=1, padding=pad)
    density = var.mean(dim=1, keepdim=True).sqrt()

    if blur > 0:
        bk = blur * 2 + 1
        density = F.avg_pool2d(density, bk, stride=1, padding=blur)

    b = density.shape[0]
    flat = density.view(b, -1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    norm = ((flat - lo) / (hi - lo + 1e-8)).view_as(density)

    if threshold > 0:
        norm = (norm - threshold).clamp(min=0) / max(1e-8, 1.0 - threshold)
    if gamma != 1.0:
        norm = norm.clamp(min=0).pow(gamma)
    return norm.clamp(0, 1)


def selective_sigma_detailer_sampler(
    model,
    x,
    sigmas,
    *,
    ssd_wrapped_sampler,
    ssd_make_schedule,
    ssd_cfg_scale_override,
    ssd_variance_kernel,
    ssd_mask_blur,
    ssd_mask_threshold,
    ssd_mask_gamma,
    ssd_save_mask_preview,
    **kwargs,
):
    if ssd_cfg_scale_override > 0:
        cfg_scale = ssd_cfg_scale_override
    else:
        maybe_cfg = getattr(model.inner_model, "cfg", None)
        cfg_scale = float(maybe_cfg) if isinstance(maybe_cfg, (int, float)) else 1.0

    schedule = torch.tensor(
        ssd_make_schedule(len(sigmas) - 1), dtype=torch.float32, device="cpu"
    )
    sigmas_cpu = sigmas.detach().clone().cpu()
    sigma_max = float(sigmas_cpu[0])
    sigma_min = float(sigmas_cpu[-1]) + 1e-5

    state = {"frozen_mask": None}

    def model_wrapper(x, sigma, **extra_args):
        sigma_float = float(sigma.max().detach().cpu())
        if not (sigma_min <= sigma_float <= sigma_max):
            return model(x, sigma, **extra_args)

        adjustment = _sample_schedule(sigma_float, sigmas_cpu, schedule) * 0.1
        if adjustment == 0.0:
            return model(x, sigma, **extra_args)

        denoised_normal = model(x, sigma, **extra_args)

        if state["frozen_mask"] is None:
            state["frozen_mask"] = _compute_density_mask(
                denoised_normal,
                ssd_variance_kernel,
                ssd_mask_blur,
                ssd_mask_threshold,
                ssd_mask_gamma,
            )
            if ssd_save_mask_preview:
                path = _save_mask_preview(state["frozen_mask"])
                print(f"[SelectiveSigmaDetailer] mask preview saved: {path}")

        mask = state["frozen_mask"]
        if mask.shape[-2:] != denoised_normal.shape[-2:]:
            mask = F.interpolate(
                mask, size=denoised_normal.shape[-2:], mode="bilinear", align_corners=False
            )

        adjusted_sigma = sigma * max(1e-6, 1.0 - adjustment * cfg_scale)
        denoised_detailed = model(x, adjusted_sigma, **extra_args)

        m = mask.to(denoised_normal)
        return denoised_normal * (1 - m) + denoised_detailed * m

    for k in ("inner_model", "sigmas"):
        if hasattr(model, k):
            setattr(model_wrapper, k, getattr(model, k))
    return ssd_wrapped_sampler.sampler_function(
        model_wrapper, x, sigmas, **kwargs, **ssd_wrapped_sampler.extra_options
    )


class SelectiveSigmaDetailerNode:
    DESCRIPTION = (
        "Boosts detail only in regions that are already dense (high local "
        "latent variance) at the moment the schedule activates. Costs 2x "
        "model calls during active steps."
    )
    CATEGORY = "sampling/custom_sampling/samplers"
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "detail_amount": ("FLOAT", {"default": 0.4, "min": -5.0, "max": 5.0, "step": 0.01}),
                "start": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "exponent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "start_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "end_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "fade": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "smooth": ("BOOLEAN", {"default": True}),
                "cfg_scale_override": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                "variance_kernel": ("INT", {"default": 5, "min": 3, "max": 15, "step": 2,
                    "tooltip": "Window size for local variance. Larger = coarser density estimate."}),
                "mask_blur": ("INT", {"default": 2, "min": 0, "max": 16, "step": 1,
                    "tooltip": "Smoothing applied to the mask. Softens transitions to avoid seams."}),
                "mask_threshold": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.99, "step": 0.01,
                    "tooltip": "Mask values below this are pulled to 0. Higher = stricter targeting of dense areas."}),
                "mask_gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05,
                    "tooltip": "Contrast curve on the mask. >1 sharpens, <1 softens."}),
                "save_mask_preview": ("BOOLEAN", {"default": False,
                    "tooltip": "Save the frozen mask as a PNG to ComfyUI's temp directory. Path is printed to console."}),
            },
        }

    @classmethod
    def go(
        cls,
        sampler,
        *,
        detail_amount,
        start,
        end,
        bias,
        exponent,
        start_offset,
        end_offset,
        fade,
        smooth,
        cfg_scale_override,
        variance_kernel,
        mask_blur,
        mask_threshold,
        mask_gamma,
        save_mask_preview,
    ):
        def ssd_make_schedule(steps):
            return _make_schedule(
                steps, start, end, bias, detail_amount, exponent,
                start_offset, end_offset, fade, smooth,
            )

        return (
            KSAMPLER(
                selective_sigma_detailer_sampler,
                extra_options={
                    "ssd_wrapped_sampler": sampler,
                    "ssd_make_schedule": ssd_make_schedule,
                    "ssd_cfg_scale_override": cfg_scale_override,
                    "ssd_variance_kernel": variance_kernel,
                    "ssd_mask_blur": mask_blur,
                    "ssd_mask_threshold": mask_threshold,
                    "ssd_mask_gamma": mask_gamma,
                    "ssd_save_mask_preview": save_mask_preview,
                },
            ),
        )
