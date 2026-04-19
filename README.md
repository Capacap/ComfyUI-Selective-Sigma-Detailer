# ComfyUI-Selective-Sigma-Detailer

A ComfyUI custom sampler that boosts detail only in latent regions that are
already dense at the moment its schedule activates. Smooth regions (skies,
out-of-focus backgrounds, flat surfaces) are left untouched; busy regions
(faces, fabric, foliage) get a sigma-shifted pass that sharpens them.

## How it works

1. Early sampling steps run unmodified so the model can establish composition.
2. On the first step where the schedule becomes active, a **density mask** is
   derived from the model's denoised prediction via local latent variance, and
   then frozen for the rest of the run.
3. For every subsequent active step, two denoises are computed (normal, and
   one at a shifted sigma) and blended per-pixel by the mask.

Cost: 2x model calls on active steps only. Inactive steps are free.

## Node

**Selective Sigma Detailer** (`sampling/custom_sampling/samplers`)

Wraps a `SAMPLER` and returns a new `SAMPLER`. Drop it between your base
sampler (`KSamplerSelect`, etc.) and `SamplerCustom`.

### Schedule parameters

`detail_amount`, `start`, `end`, `bias`, `exponent`, `start_offset`,
`end_offset`, `fade`, `smooth`, `cfg_scale_override` control when and how
strongly the detail pass fires, following the same schedule shape as
Detail Daemon.

### Mask parameters

- `variance_kernel` — window size for local variance (odd, 3–15). Larger is coarser.
- `mask_blur` — smoothing applied after variance. Softens transitions.
- `mask_threshold` — values below this are pulled to 0. Higher = stricter targeting.
- `mask_gamma` — contrast curve on the mask. >1 sharpens, <1 softens.
- `save_mask_preview` — dumps the frozen mask as a PNG to ComfyUI's temp dir.

## Credits

The schedule shaping helpers in `schedule.py` are vendored from
[ComfyUI-Detail-Daemon](https://github.com/Jonseed/ComfyUI-Detail-Daemon) by
Jonseed (MIT). See `LICENSE-THIRD-PARTY`.

## License

MIT.
