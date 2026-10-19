# imggen-served-edit-divergence — FLUX.2 klein's served edit against its CPU reference

**Open** (2026-10-18, 1.1.1). The edit sd-server serves is 17.2 dB from the
same request on a CPU sd-server, below the image gate's proposed 20 dB
floor; released with this gate open (`decision`, operator, 2026-10-18).

## Charter

`POST /v1/images/edits` on `arcint-imggen-flux` uses the uploaded image as
the **init image** (img2img at strength 0.75) and as FLUX.2 klein's
**reference image** (`code`, sd.cpp `examples/server/routes_openai.cpp`).
Attribute the gap between the B60 and the CPU on that path, and either fix
it or set the tolerance for img2img with the operator.

## Evidence (`measured-here`, 2026-10-18, the 1.1.1 release build)

512², 4 steps, seed 42, the cat-on-a-windowsill source image, the
background-swap prompt, the flux unit's flags, one B60 sd-server and one CPU
sd-server (`--backend cpu`) given the same multipart request:

| comparison | final PSNR | SSIM |
|---|---|---|
| B60 served edit vs the CPU served edit | **17.21 dB** | 0.905 |
| B60 served edit with `--vae-on-cpu` vs the CPU served edit | 17.25 dB | 0.906 |
| B60 served edit with `--vae-on-cpu` vs the all-OpenCL B60 edit | 43.72 dB | |
| the reference-only edit (`sd-cli --ref-image`), B60 vs CPU | 32.43 dB | 0.997 (gate PASS) |

- The difference is local: over most of the frame the two images agree
  within 1-7 levels, and they differ by ~56 in one region (the lower left),
  with no shift between them. Both are valid edits by eye.
- **Not the VAE**: with the VAE on the CPU the B60's image barely moves
  (43.7 dB from the all-OpenCL one) and the gap to the CPU stays.
- `sd-cli --init-img` with `--ref-image` and `--strength 0.75` returned the
  reference-only image unchanged, so the CLI does not reproduce the server's
  path and there is no per-step arm; sd-server writes no previews.

## Next

The DiT against the text encoder (`--clip-on-cpu` puts the encoder on the
CPU), then the img2img start (the noised init latent) on each backend; a
1024² arm. The gate's per-step cosine needs previews from the server path or
a CLI arm that reproduces it.

## Where it lives

`docs/milestone-1.1.1.md` §9.1, `docs/imggen-service.md` (the edit route),
`tools/imggen_gate.py` (`MODEL_TOL`).
