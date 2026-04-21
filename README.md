# ComfyUI-Selective-Sigma-Detailer

A ComfyUI custom sampler wrapper in the family of [Detail Daemon](https://github.com/Jonseed/ComfyUI-Detail-Daemon)
and its descendants. It sharpens an image mid-sampling by telling the
denoiser that the remaining noise is smaller than it actually is, which
nudges the model to commit to higher-frequency structure. The difference is
that this wrapper does it selectively: only the regions of the latent where
structure is still forming get the sharpening treatment. Smooth areas are
left at the normal sigma.

Selectivity exists because global sigma modulation has a failure mode. It
sharpens everything, including the regions that were supposed to stay
smooth. Clean skies develop grain, soft bokeh turns crunchy, solid-color
illustrations lose their flatness. Detail Daemon, MultiplySigmas, and
similar tricks all share this behavior. If a composition depends on a
genuinely clean background or a shallow depth of field, a global detailer
works against you. This node preserves those regions by only applying the
sigma shift where the model is already drawing detail.

The mask is built by running the model once at the normal sigma and
comparing the result against the previous step's prediction. Regions where
the prediction is still changing step-to-step count as busy. A second pass
runs at a reduced sigma and gets blended with the first by the mask. This
means the wrapper costs roughly two model forwards per active step instead
of one, a meaningful overhead, made less painful by skipping the first 20%
of steps (composition phase) and tapering off during the last 15%
(structure locked in). A typical 16-step SDXL run ends up paying for about
nine detail passes rather than sixteen. Still more expensive than a plain
sampler or Detail Daemon, worth it specifically when preserving smooth
regions matters to the composition.

The reason two forwards are required is that sigma is a scalar from the
denoiser's perspective. You cannot pass a spatial sigma map in a single
model call without retraining. Running twice and blending by the mask is
the only way to get region-selective behavior out of the existing model.

## Nodes

All nodes live under `sampling/custom_sampling/samplers`.

**Selective Sigma Detailer.** The main node. Wraps a `SAMPLER` and returns a
new `SAMPLER` to drop between `KSamplerSelect` and `SamplerCustom`. Two
parameters: `intensity` (default 16.0) sets the per-step strength of the
sigma shift, where 16.0 corresponds to a 0.1 fractional reduction; negative
values soften instead of sharpen. `coverage` (default 0.5) shifts the mask
threshold. At 0 it disables the detail pass entirely, at 0.5 it uses the
raw normalized mask, at 1.0 it saturates the mask and effectively applies
the shift everywhere (equivalent to running Detail Daemon on the same
schedule).

**Selective Sigma Detailer (Debug).** Same sampler wrapper with the internal
constants (`start`, `ema`, `mask_clip_percentile`) exposed as inputs and a
`mask_ref` output. Use when diagnosing unexpected behavior or experimenting
with different constants. Defaults match the main node.

**Selective Sigma Detailer (Debug Preview).** Takes the `mask_ref` from the
Debug sampler and renders the last captured mask as a preview image.
Latent passthrough is required so ComfyUI runs it after sampling finishes.

Each run prints a stats line to the console showing how many steps paid
for the detail pass, the count of short-circuits by reason, and the total
forwards saved:

```
[SSD] calls=16 detail=9 skip: schedule=6 activity=0 first=1 range=0 (7 forwards saved)
```

## Credits

Schedule gating and the sigma-shift mechanic adapted from
[ComfyUI-Detail-Daemon](https://github.com/Jonseed/ComfyUI-Detail-Daemon)
by Jonseed.

## License

MIT.
