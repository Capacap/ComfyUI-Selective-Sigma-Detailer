"""Core helpers for the Selective Sigma Detailer sampler wrapper.

The wrapper runs the model twice per step: once at the normal sigma to get a
reference prediction, and once at a slightly reduced sigma to get a
more-detailed prediction. The two are blended by a mask derived from how much
the prediction changed between the previous step and this one, so "busy"
regions receive extra detail while flat regions are left alone.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from comfy.samplers import KSAMPLER


def make_schedule(steps, start, amount):
    """Build a per-step intensity schedule.

    Returns a length-`steps` array of zeros, with `amount` filled in from
    `start_idx` onward. The first few steps are left at zero because they set
    overall composition; detail injection there tends to warp layout rather
    than sharpen. At least one step is always skipped.
    """
    multipliers = np.zeros(steps)
    start_idx = max(1, int(round(start * (steps - 1))))
    multipliers[start_idx:] = amount
    return multipliers


def sample_schedule(sigma, sigmas, schedule):
    """Look up the schedule value for a given sigma via linear interpolation.

    The sampler may call the model at intermediate sigmas (RK-style solvers,
    ancestral noise), not only at the discrete scheduled sigmas. We map an
    arbitrary sigma back onto the schedule by finding the nearest scheduled
    sigma and lerping toward its neighbor. Returns 0.0 for sigmas outside the
    schedule range, which causes the wrapper to skip the detail pass entirely.
    """
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
    """Min/max stretch each batch element of a raw mask into [0, 1].

    A handful of extreme pixels can dominate the min/max and collapse the rest
    of the mask to near-zero. Clipping both tails at `clip_percentile` before
    the stretch keeps the normalized range dominated by the bulk of the
    distribution instead of a few outliers.
    """
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
    """Convert a latent-space mask to a grayscale BHWC image for preview.

    Takes the first channel, upscales with nearest-neighbor to show the
    latent-grid structure honestly (no bilinear smoothing that would hide
    artifacts), and expands to 3 channels so standard image nodes accept it.
    """
    m = mask[:, :1].detach().float().cpu().clamp(0, 1)
    if upscale > 1:
        m = F.interpolate(m, scale_factor=upscale, mode="nearest")
    m = m.squeeze(1)
    return m.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()


# Calibration anchor: at intensity=16 the per-step sigma shift is 0.1.
# Deliberately NOT divided by the actual step count — the sampler's own
# integration already applies step-size normalization, so dividing again would
# make short runs Nx stronger than long runs. Keep the shift per-step-constant
# across step counts.
_INTENSITY_REFERENCE = 16.0
_REFERENCE_PER_STEP_SHIFT = 0.1


def build_sampler(wrapped_sampler, make_schedule_fn, mask_fn, mask_params, mask_ref):
    """Wrap an existing SAMPLER with the two-pass detail injection.

    Flow per denoiser call:
      1. Run the wrapped model at the original sigma -> `denoised_normal`.
      2. Ask `mask_fn` for a mask derived from `denoised_normal` vs. the
         previous step's prediction (stored in `state`). `mask_fn` may return
         None on the first step (no previous frame yet) to skip detail.
      3. Run the model again at a slightly reduced sigma. A lower sigma tells
         the denoiser to assume less remaining noise, so it commits to
         higher-frequency structure -> `denoised_detailed`.
      4. Blend the two predictions by the mask.

    `mask_ref` is a caller-owned dict used as a one-slot channel for the debug
    preview node. It's cleared on every new sampler invocation so stale masks
    from prior runs don't leak into the preview.
    """
    def sampler_function(model, x, sigmas, **kwargs):
        mask_ref.clear()
        schedule = torch.tensor(
            make_schedule_fn(len(sigmas) - 1), dtype=torch.float32, device="cpu"
        )
        sigmas_cpu = sigmas.detach().clone().cpu()
        sigma_max = float(sigmas_cpu[0])
        # Small epsilon so the final scheduled sigma still counts as in-range
        # against float comparison noise.
        sigma_min = float(sigmas_cpu[-1]) + 1e-5

        # Per-run scratch space for the mask_fn (previous denoised, previous
        # mask for EMA, etc.). Scoped to this sampler_function call so each
        # run starts clean.
        state = {}

        def model_wrapper(x, sigma, **extra_args):
            # Reduce to a single scalar: batched sigma tensors are all equal
            # in practice, but .max() is defensive against solvers that pass
            # per-sample sigmas.
            sigma_float = float(sigma.max().detach().cpu())
            # Out-of-range sigmas come from ancestral noise or solver probes
            # at boundaries — not actual denoising steps on our schedule.
            if not (sigma_min <= sigma_float <= sigma_max):
                return model(x, sigma, **extra_args)

            intensity = sample_schedule(sigma_float, sigmas_cpu, schedule)
            adjustment = intensity * _REFERENCE_PER_STEP_SHIFT / _INTENSITY_REFERENCE
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

            # Clamp the scale floor away from zero so the model never sees
            # sigma=0 (which some denoisers handle as a special case).
            adjusted_sigma = sigma * max(1e-6, 1.0 - adjustment)
            denoised_detailed = model(x, adjusted_sigma, **extra_args)
            m = mask.to(denoised_normal)
            return denoised_normal * (1 - m) + denoised_detailed * m

        # Propagate attributes that downstream samplers (e.g. ancestral
        # variants in comfy) read off the model callable.
        for k in ("inner_model", "sigmas"):
            if hasattr(model, k):
                setattr(model_wrapper, k, getattr(model, k))
        return wrapped_sampler.sampler_function(
            model_wrapper, x, sigmas, **kwargs, **wrapped_sampler.extra_options
        )

    return KSAMPLER(sampler_function)
