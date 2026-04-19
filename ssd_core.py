from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from comfy.samplers import KSAMPLER


def make_schedule(
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


def sample_schedule(sigma, sigmas, schedule):
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


def postprocess_mask(raw, blur, threshold, gamma):
    if blur > 0:
        bk = blur * 2 + 1
        padded = F.pad(raw, (blur, blur, blur, blur), mode="reflect")
        raw = F.avg_pool2d(padded, bk, stride=1, padding=0)
    b = raw.shape[0]
    flat = raw.view(b, -1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    norm = ((flat - lo) / (hi - lo + 1e-8)).view_as(raw)
    if threshold > 0:
        norm = (norm - threshold).clamp(min=0) / max(1e-8, 1.0 - threshold)
    if gamma != 1.0:
        norm = norm.clamp(min=0).pow(gamma)
    return norm.clamp(0, 1)


def local_variance(denoised, kernel):
    k = kernel | 1
    pad = k // 2
    padded = F.pad(denoised, (pad, pad, pad, pad), mode="reflect")
    mean = F.avg_pool2d(padded, k, stride=1, padding=0)
    diff_sq = (denoised - mean).pow(2)
    diff_sq_padded = F.pad(diff_sq, (pad, pad, pad, pad), mode="reflect")
    var = F.avg_pool2d(diff_sq_padded, k, stride=1, padding=0)
    return var.mean(dim=1, keepdim=True).sqrt()


def sobel_magnitude(denoised):
    sx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=denoised.dtype, device=denoised.device,
    ).view(1, 1, 3, 3)
    sy = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=denoised.dtype, device=denoised.device,
    ).view(1, 1, 3, 3)
    c = denoised.shape[1]
    sx = sx.expand(c, 1, 3, 3)
    sy = sy.expand(c, 1, 3, 3)
    padded = F.pad(denoised, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(padded, sx, padding=0, groups=c)
    gy = F.conv2d(padded, sy, padding=0, groups=c)
    return (gx.pow(2) + gy.pow(2)).sqrt().mean(dim=1, keepdim=True)


def mask_to_preview_image(mask, upscale=8):
    m = mask[:, :1].detach().float().cpu().clamp(0, 1)
    if upscale > 1:
        m = F.interpolate(m, scale_factor=upscale, mode="nearest")
    m = m.squeeze(1)
    return m.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()


def build_sampler(wrapped_sampler, make_schedule_fn, cfg_scale_override,
                  mask_fn, mask_params, mask_ref):
    """Wraps `wrapped_sampler` with sigma-shift detail blending.

    `mask_fn(denoised, x, sigma, state, params) -> mask [B,1,H,W]` is invoked
    on every active step (schedule value != 0). The latest mask is written to
    `mask_ref["mask"]` so a downstream preview node can display it after
    sampling finishes.
    """

    def sampler_function(model, x, sigmas, **kwargs):
        if cfg_scale_override > 0:
            cfg_scale = cfg_scale_override
        else:
            maybe_cfg = getattr(model.inner_model, "cfg", None)
            cfg_scale = float(maybe_cfg) if isinstance(maybe_cfg, (int, float)) else 1.0

        schedule = torch.tensor(
            make_schedule_fn(len(sigmas) - 1), dtype=torch.float32, device="cpu"
        )
        sigmas_cpu = sigmas.detach().clone().cpu()
        sigma_max = float(sigmas_cpu[0])
        sigma_min = float(sigmas_cpu[-1]) + 1e-5

        state = {}

        def model_wrapper(x, sigma, **extra_args):
            sigma_float = float(sigma.max().detach().cpu())
            if not (sigma_min <= sigma_float <= sigma_max):
                return model(x, sigma, **extra_args)

            adjustment = sample_schedule(sigma_float, sigmas_cpu, schedule) * 0.1
            if adjustment == 0.0:
                return model(x, sigma, **extra_args)

            denoised_normal = model(x, sigma, **extra_args)
            mask = mask_fn(denoised_normal, x, sigma, state, mask_params)
            if mask is None:
                return denoised_normal
            mask_ref["mask"] = mask.detach()

            if mask.shape[-2:] != denoised_normal.shape[-2:]:
                mask = F.interpolate(
                    mask, size=denoised_normal.shape[-2:],
                    mode="bilinear", align_corners=False,
                )

            adjusted_sigma = sigma * max(1e-6, 1.0 - adjustment * cfg_scale)
            denoised_detailed = model(x, adjusted_sigma, **extra_args)
            m = mask.to(denoised_normal)
            return denoised_normal * (1 - m) + denoised_detailed * m

        for k in ("inner_model", "sigmas"):
            if hasattr(model, k):
                setattr(model_wrapper, k, getattr(model, k))
        return wrapped_sampler.sampler_function(
            model_wrapper, x, sigmas, **kwargs, **wrapped_sampler.extra_options
        )

    return KSAMPLER(sampler_function)


SCHEDULE_INPUTS = {
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
}


MASK_COMMON_INPUTS = {
    "mask_blur": ("INT", {"default": 2, "min": 0, "max": 16, "step": 1,
        "tooltip": "Smoothing applied to the mask. Softens transitions to avoid seams."}),
    "mask_threshold": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.99, "step": 0.01,
        "tooltip": "Mask values below this are pulled to 0. Higher = stricter targeting."}),
    "mask_gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05,
        "tooltip": "Contrast curve on the mask. >1 sharpens, <1 softens."}),
}
