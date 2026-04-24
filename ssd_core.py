"""Core helpers for the Selective Sigma Detailer sampler wrapper.

Default flow is two model forwards per active step: one at a slightly
reduced sigma for a more-detailed prediction, then one at the normal sigma
for the reference. The two are blended by a mask derived from how much the
prediction changed between the previous step and this one, so "busy" regions
receive extra detail while flat regions are left alone. When coverage is
saturated (>=1.0) the mask would collapse to all-ones, so the normal pass is
skipped and the step runs a single adjusted-sigma forward. See
`build_sampler` for the rationale behind the detail-first, normal-last
ordering.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from comfy.samplers import KSAMPLER


# Width (as a fraction of the schedule) of the linear taper that eases the
# detail pass down from full strength to zero at the tail. The taper starts
# at `end - _TAIL_TAPER_WIDTH` and finishes at `end`. Kept narrow so the
# active window dominates; the taper's job is to avoid a cliff, not to be a
# second phase. Not user-exposed: a smoothing detail, not a knob worth
# turning.
_TAIL_TAPER_WIDTH = 0.15


def make_schedule(steps, start, end, strength):
    """Build a per-step sigma-reduction schedule.

    `strength` is the peak per-step fraction of sigma removed during the
    detail pass: adjusted_sigma = sigma * (1 - strength). E.g. 0.1 means
    "shave 10% off sigma at peak". Deliberately NOT divided by step count —
    the sampler's own integration already applies step-size normalization,
    so dividing again would make short runs Nx stronger than long runs.

    Schedule shape:
      [0, start_idx)              -> 0 (composition-setting prologue)
      [start_idx, taper_start)    -> full `strength`
      [taper_start, end_idx]      -> linearly decays to 0
      (end_idx, steps)            -> 0 (skipped via adjustment==0)

    Invariants enforced here regardless of `start`/`end` values:
      - at least one clean step at the head (index 0 is always 0)
      - at least one clean step at the tail (index steps-1 is always 0)
    These give the sampler room to set composition before detail kicks in
    and to clean leftover noise after the last detail step.
    """
    multipliers = np.zeros(steps)
    start_idx = max(1, int(round(start * (steps - 1))))
    end_idx = min(steps - 2, int(round(end * (steps - 1))))
    # Degenerate step counts (e.g. 3 steps with start=0.1, end=0.9) may leave
    # no room for any active region after the head/tail clean-step invariants
    # are enforced. Bail out with an all-zero schedule rather than inverting.
    if end_idx <= start_idx:
        return multipliers
    taper_start_idx = max(start_idx + 1, int(round((end - _TAIL_TAPER_WIDTH) * (steps - 1))))
    taper_start_idx = min(taper_start_idx, end_idx)
    multipliers[start_idx:taper_start_idx] = strength
    taper_len = end_idx - taper_start_idx + 1
    if taper_len > 1:
        multipliers[taper_start_idx:end_idx + 1] = np.linspace(strength, 0.0, taper_len)
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


# When the mask's mean activity is below this floor, the detail pass would
# contribute <2% to the blend. Skip the blend and return denoised_normal;
# the detail forward has already run by this point (we need it before the
# mask is available) so this saves blend math, not a forward.
_MIN_MASK_ACTIVITY = 0.02


# CFG++ samplers pull the noise direction from `uncond_denoised` captured via
# a post_cfg_function hook, separate from the returned `denoised`. Our two-
# forward blend mixes `denoised` across two sigmas while the uncond stays
# tied to one, so the implicit guidance signal (denoised - uncond) gains a
# cross-sigma term that amplifies the effective strength. Empirically tuned
# against mid-mask behavior (the typical operating point); over-attenuates
# as the mask saturates toward 1 (where coverage=1 fast path is a better
# choice anyway). Detected per-run by inspecting model_options for the
# CFG++ hook.
_CFG_PP_STRENGTH_ATTENUATION = 0.15


def _has_cfg_pp_hook(extra_args):
    """True if the sampler has installed a CFG++ post_cfg_function hook."""
    model_options = extra_args.get("model_options") or {}
    hooks = model_options.get("sampler_post_cfg_function") or ()
    return len(hooks) > 0


def build_sampler(wrapped_sampler, make_schedule_fn, mask_fn, mask_params, mask_ref):
    """Wrap an existing SAMPLER with the two-pass detail injection.

    Flow per denoiser call:
      1. Run the model at a slightly reduced sigma -> `denoised_detailed`.
         A lower sigma tells the denoiser to assume less remaining noise, so
         it commits to higher-frequency structure.
      2. Run the model at the original sigma -> `denoised_normal`. This call
         is last so that CFG++ samplers, which capture `uncond_denoised` via a
         post_cfg_function hook, end up with the uncond aligned to the sigma
         they actually use in their step formula.
      3. Ask `mask_fn` for a mask from `denoised_normal` vs. the previous
         step's prediction (stored in `state`). Returns None on the first step.
      4. Blend the two predictions by the mask.

    `mask_ref` is a caller-owned dict used as a one-slot channel for the debug
    preview node. It's cleared on every new sampler invocation so stale masks
    from prior runs don't leak into the preview.

    Coverage fast path (checked once per run, not per step):
      coverage >= 1 -> full: the blend would collapse to denoised_detailed, so
                       skip the normal forward and the mask_fn entirely. Just
                       run the adjusted-sigma pass. One forward per step
                       instead of two. Caveat: on CFG++ samplers this leaves
                       uncond_denoised captured at the adjusted sigma while
                       the sampler uses it at the original — same class of
                       sigma mismatch as plain Detail Daemon on CFG++. Fix
                       would be to trail a normal-sigma call, which negates
                       the fast path; left as a known limitation.

    (coverage <= 0 is short-circuited at the node boundary and never reaches
    build_sampler.)
    """
    full_coverage = mask_params.get("coverage", 0.5) >= 1.0

    def sampler_function(model, x, sigmas, **kwargs):
        mask_ref.clear()
        # Populate mask_ref up-front at full coverage so the debug preview
        # reflects the fast-path behavior (mask is effectively all-ones)
        # rather than rendering the empty default.
        if full_coverage:
            mask_ref["mask"] = torch.ones(
                (x.shape[0], 1, x.shape[2], x.shape[3]),
                device=x.device, dtype=x.dtype,
            )
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
        # Per-run bookkeeping for the end-of-run summary.
        stats = {"detail": 0, "skip_activity": 0, "skip_schedule": 0, "skip_first": 0, "skip_range": 0, "full": 0}
        # Detected on first in-range call (we need extra_args to inspect).
        cfg_pp_state = {"checked": False, "active": False}

        def model_wrapper(x, sigma, **extra_args):
            # Reduce to a single scalar: batched sigma tensors are all equal
            # in practice, but .max() is defensive against solvers that pass
            # per-sample sigmas.
            sigma_float = float(sigma.max().detach().cpu())
            # Out-of-range sigmas come from ancestral noise or solver probes
            # at boundaries — not actual denoising steps on our schedule.
            if not (sigma_min <= sigma_float <= sigma_max):
                stats["skip_range"] += 1
                return model(x, sigma, **extra_args)

            adjustment = sample_schedule(sigma_float, sigmas_cpu, schedule)
            if adjustment == 0.0:
                stats["skip_schedule"] += 1
                return model(x, sigma, **extra_args)

            if not cfg_pp_state["checked"]:
                cfg_pp_state["active"] = _has_cfg_pp_hook(extra_args)
                cfg_pp_state["checked"] = True

            # Coverage=1.0: the mask would saturate to all-ones, so the normal
            # pass and mask_fn are both dead weight. Run only the detail pass.
            if full_coverage:
                stats["full"] += 1
                adjusted_sigma = sigma * max(1e-6, 1.0 - adjustment)
                return model(x, adjusted_sigma, **extra_args)

            if cfg_pp_state["active"]:
                adjustment = adjustment * _CFG_PP_STRENGTH_ATTENUATION

            # Detail first, normal last. CFG++ samplers install a
            # post_cfg_function hook that captures `uncond_denoised` into the
            # sampler's closure on every model call, and use it to compute the
            # noise direction `to_d(x, sigma_i, uncond_denoised)` in their
            # step formula. Only the LAST captured value survives, so ending
            # on the original sigma keeps the side channel aligned with the
            # sigma the sampler plugs into its formula. For non-CFG++ samplers
            # the order is irrelevant (they only read the returned `denoised`)
            # so this is safe as the single path. Cost: the activity and
            # first-step short-circuits no longer save a forward, because the
            # mask isn't available until after the normal pass has already
            # run. Acceptable — both skip reasons are rare relative to the
            # active blend steps.
            adjusted_sigma = sigma * max(1e-6, 1.0 - adjustment)
            denoised_detailed = model(x, adjusted_sigma, **extra_args)
            denoised_normal = model(x, sigma, **extra_args)

            mask = mask_fn(denoised_normal, x, sigma, state, mask_params)
            if mask is None:
                stats["skip_first"] += 1
                return denoised_normal
            mask_ref["mask"] = mask.detach()

            if mask.mean().item() < _MIN_MASK_ACTIVITY:
                stats["skip_activity"] += 1
                return denoised_normal
            stats["detail"] += 1

            if mask.shape[-2:] != denoised_normal.shape[-2:]:
                mask = F.interpolate(
                    mask, size=denoised_normal.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
            m = mask.to(denoised_normal)
            return denoised_normal * (1 - m) + denoised_detailed * m

        # Propagate attributes that downstream samplers (e.g. ancestral
        # variants in comfy) read off the model callable.
        for k in ("inner_model", "sigmas"):
            if hasattr(model, k):
                setattr(model_wrapper, k, getattr(model, k))
        result = wrapped_sampler.sampler_function(
            model_wrapper, x, sigmas, **kwargs, **wrapped_sampler.extra_options
        )
        total = sum(stats.values())
        saved = total - stats["detail"]
        cfg_pp_tag = " cfg++attenuated" if cfg_pp_state["active"] else ""
        print(
            f"[SSD] calls={total} detail={stats['detail']} full={stats['full']} "
            f"skip: schedule={stats['skip_schedule']} "
            f"activity={stats['skip_activity']} "
            f"first={stats['skip_first']} range={stats['skip_range']} "
            f"({saved} forwards saved){cfg_pp_tag}"
        )
        return result

    return KSAMPLER(sampler_function)
