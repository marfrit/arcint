# The image service — units, ports, model names, request formats (1.1.1)

The 1.1.0 image service was one unit (`arcint-imggen`, Z-Image Turbo, port
8085) behind an on-demand front. 1.1.1 adds the models the marketing work
needs; each is its own **on-demand unit next to `arcint-imggen`**, on the
same card, with the same `Conflicts=` (the card's language units, and in
the new units the other image units too: one model at a time on the card)
and the same contract (sd.cpp's `sd-server`). The front and the client side
are the operator's; the units and this contract live here.

## Units

| unit | port | model files | OpenAI routes | other |
|---|---|---|---|---|
| `arcint-imggen` | 8085 | Z-Image Turbo Q8_0 + Qwen3-4B Q8_0 + `ae.safetensors` | `/v1/images/generations` | — |
| `arcint-imggen-qwen` | **8081** | Qwen-Image Q4_K_M + Qwen2.5-VL-7B Q4_K_M + `qwen-image-vae.safetensors` | `/v1/images/generations` | `/sdcpp/v1/upscale` |
| `arcint-imggen-flux` | **8082** | FLUX.2 klein 4B Q8_0 + Qwen3-4B Q8_0 + `flux2-klein-4b-vae.safetensors` | `/v1/images/generations`, `/v1/images/edits` | `/sdcpp/v1/upscale` |

Both serve sd.cpp's own `POST /sdcpp/v1/upscale` (ESRGAN 4x, the print
route). sd-server finds its upscalers only by scanning
`--hires-upscalers-dir` (`code`, `examples/server/runtime.cpp:331`;
`--upscale-model` is read by the CLI alone), and it loads every model file
in that directory to read its scale, so the units point at a directory
holding `RealESRGAN_x4plus.safetensors` and nothing else. The default sizes
are 1024²; each unit's defaults (steps/cfg/sampler) are in its unit file and
per-request tunable. 8081 and 8082 were free on the card host (the language
units use 8080 and 8087-8089, the 1.1.0 pair 8085/8086, the dev windows
8090-8098). Renumbering is the operator's call.

## The request contract

sd-server is sd.cpp's own server; the routes are unchanged from 1.1.0
(`docs/imggen.md`, `contrib/sdcpp/README.md`).

- `POST /v1/images/generations` — JSON body `{model?, prompt, n?, size?
  ("WxH"), response_format:"b64_json", ...}`; response
  `{"created":.., "data":[{"b64_json":..}]}`.
- `POST /v1/images/edits` — `multipart/form-data` with the text field
  `prompt` and one or more `image[]` files (or legacy `image`), an optional
  `mask`, and the fields `n`, `size`, `output_format`. sd-server uses the
  first image twice (`code`, `routes_openai.cpp`): as the **init image**
  (img2img, at the unit's `strength`, sd.cpp's default **0.75**, settable
  per request with `<sd_cpp_extra_args>{"strength":0.9}</sd_cpp_extra_args>`)
  and as FLUX.2 klein's **reference image** (passed to the DiT). So the
  served edit keeps the source's layout less strictly than a reference-only
  edit (`sd-cli --ref-image` without `--init-img`); background swap, object
  removal and photo variants both do. The served form is 17.2 dB from the
  same request on a CPU sd-server, below the gate's proposed floor; 1.1.1
  ships with that gate open (`decision`, operator, 2026-10-18;
  `docs/campaigns/imggen-served-edit-divergence.md`).
- **Seed per request** — append
  `<sd_cpp_extra_args>{"seed":N}</sd_cpp_extra_args>` to the prompt; the
  suffix also carries `steps`, `cfg_scale`, `width`, `height` and other
  sd.cpp generation args the unit's defaults do not already set.
- `POST /sdcpp/v1/upscale` — sd.cpp's native upscale route, JSON body
  `{image: <base64 PNG/JPEG>, upscaler?: "RealESRGAN_x4plus", repeats?: 1-4,
  tile_size?, output_format?, output_compression?}`; the factor is the
  model's (4x a repeat), the result at most 8192 x 8192
  (`code`, `routes_sdcpp.cpp:479-640`); reply `{upscaler, scale, repeats,
  width, height, images: [{b64_json}]}`. 512² to 2048² took 27.9 s on the
  flux unit (`measured-here`).
- `GET /health` (`{"status":"ok"}` once listening), `GET /props` (model,
  TE, VAE, backends, devices, flash-attn/direct-conv modes, defaults,
  build string), `GET /v1/models` (`sd-cpp-local`; front maps it to the
  family name).
- One request at a time a unit: sd-server serialises generation on its
  context; an upscale waits for a running generation.

## The front's mapping (operator's side, proposal)

The 1.1.0 front starts a single unit on the first request. For the new
models it needs a name -> unit/port table, e.g.:

| OpenAI `model` | unit | port |
|---|---|---|
| `z-image-turbo` (default) | `arcint-imggen` | 8085 |
| `qwen-image` | `arcint-imggen-qwen` | 8081 |
| `flux2-klein` (generation and edits) | `arcint-imggen-flux` | 8082 |

The front already reports 503 while a card-window process holds the card
and releases the B60 after 20 idle minutes; each new unit needs the same
"started by the front" bookkeeping, and since starting one image unit stops
another (`Conflicts=`), the front has to treat a switch between image
models like a switch from a language unit. All three run
`/usr/bin/arcint-sd-server` from the **1.1.1** `arcint-imggen` package, which
also installs the three unit templates in `/usr/share/arcint-imggen/`: the
new models need 0059 (Qwen-Image's VAE) and 0060 (the upscaler), which the
1.1.0 package's server does not have.

## After a card window

Before a dev window on the B60, release through the front
(`POST /release` on the card host and `/status`), then stop the owning
language unit; restore the language units by name afterwards. The exact
commands are operator-local (`CLAUDE.local.md`).

## Where it lives

- The unit files: `contrib/systemd/arcint-imggen{,-qwen,-flux}.service`.
- The engine and the routes: `contrib/sdcpp/README.md`, sd.cpp's
  `examples/server/routes_openai.cpp` and `routes_sdcpp.cpp`.
- The model set, licences and staging: `docs/milestone-1.1.1.md` §2.
