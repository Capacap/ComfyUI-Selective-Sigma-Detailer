from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from comfy.samplers import KSAMPLER


def make_schedule(steps, start, amount):
    multipliers = np.zeros(steps)
    start_idx = max(1, int(round(start * (steps - 1))))
    multipliers[start_idx:] = amount
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


def normalize_mask(raw, clip_percentile):
    b = raw.shape[0]
    flat = raw.view(b, -1)
    if clip_percentile > 0:
        q = torch.tensor(
            [clip_percentile, 1.0 - clip_percentile],
            dtype=flat.dtype, device=flat.device,
        )
        bounds = torch.quantile(flat, q, dim=1)
        lo = bounds[0].unsqueeze(1)
        hi = bounds[1].unsqueeze(1)
    else:
        lo = flat.min(dim=1, keepdim=True).values
        hi = flat.max(dim=1, keepdim=True).values
    return ((flat - lo) / (hi - lo + 1e-8)).clamp(0, 1).view_as(raw)


def mask_to_preview_image(mask, upscale=8):
    m = mask[:, :1].detach().float().cpu().clamp(0, 1)
    if upscale > 1:
        m = F.interpolate(m, scale_factor=upscale, mode="nearest")
    m = m.squeeze(1)
    return m.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()


def build_sampler(wrapped_sampler, make_schedule_fn, mask_fn, mask_params, mask_ref,
                  normalize_by_active_steps=False):
    def sampler_function(model, x, sigmas, **kwargs):
        maybe_cfg = getattr(model.inner_model, "cfg", None)
        cfg_scale = float(maybe_cfg) if isinstance(maybe_cfg, (int, float)) else 1.0

        schedule = torch.tensor(
            make_schedule_fn(len(sigmas) - 1), dtype=torch.float32, device="cpu"
        )
        if normalize_by_active_steps:
            active_count = int((schedule != 0).sum())
            adjustment_scale = 1.0 / max(1, active_count)
        else:
            adjustment_scale = 1.0
        sigmas_cpu = sigmas.detach().clone().cpu()
        sigma_max = float(sigmas_cpu[0])
        sigma_min = float(sigmas_cpu[-1]) + 1e-5

        state = {}

        def model_wrapper(x, sigma, **extra_args):
            sigma_float = float(sigma.max().detach().cpu())
            if not (sigma_min <= sigma_float <= sigma_max):
                return model(x, sigma, **extra_args)

            adjustment = sample_schedule(sigma_float, sigmas_cpu, schedule) * 0.1 * adjustment_scale
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
