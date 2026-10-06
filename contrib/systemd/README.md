# Two cards, two engines: the dev fleet's units as an example

These are the user units one host runs in production, with the operator-local
detail taken out. `packaging/arcint.service` is the *template* the package
installs; these are what it looks like once two cards and two models are on
one machine. Every flag is literal in `ExecStart` on purpose: a unit manager
and a journal can both read the port, the served name and the context there.

| unit | card | model | port |
|---|---|---|---|
| `arcint-agent.service` | GPU.0 (Arc Pro B60, 24 GB) | Qwen3.8-27B dense, this project's AWQ export (`qwen38-b7c1-ov`) with the reconstructed MTP head, MTP on | 8087 |
| `arcint-coder.service` | GPU.1 (Arc A770, 16 GiB) | Qwen3.6-27B-A3B coder, MoE (the b5 export) | 8080 |
| `arcint-qwen38-mtp.service` | GPU.0 | Qwen3.8-27B, Intel's public int4 IR with the reconstructed MTP head | 8088 — an example, not deployed |

Things worth copying rather than re-learning:

- **`Conflicts=`, not arithmetic, keeps a card exclusive.** A resident model
  holds its VRAM for the process lifetime, and a second engine loading beside
  it fails with allocation errors at best. Each unit names the other units
  that want the same card, including retired ones kept as rollback paths —
  starting a retired unit then stops the live one instead of colliding with it.
- **`--served-model-name` names the endpoint, `--model-id` asserts the
  artifact.** A proxy pins its roster to the former; the latter refuses to
  start on the wrong directory.
- **The reservation decides whether the context fits.** `--n-ctx 151552` on
  the 24 GB card and `--n-ctx 131072` on the 16 GiB card are explicit, so they
  are verify-only: the load either admits them (trimming the prefix-cache
  reserve first) or refuses with every reservation term printed — it never
  lowers them silently. Omit `--n-ctx` to adopt the largest depth the card
  admits instead.
- **`--gate-pad 16` is a no-op on a dense model.** It widens the MoE
  shared-expert gate: −13% prefill wall, −5% decode (DESIGN 7.0.2g), a win
  only for prefill-heavy MoE traffic. The dense agent omits it; the coder
  (MoE, but decode-bound one-shot coding traffic) leaves it off.
- **`--paged-kv u8:i4`** on the agent (eight-bit keys, four-bit values) buys
  +28% context at u8's prefill rate on `marfrit-openvino +p6` and later;
  `--paged-kv u8` on the coder is the default precision. Both score 10/10 on
  the acceptance task.
- **`--prefill-chunk 512`** on the agent is a context lever: on a card this
  full the activation reservation grows with the chunk, and 512 is what lets
  151,552 tokens fit with MTP on (DESIGN's fit and served-configuration sections). The coder leaves it at
  the default.
- **`--cache-host-mib 4096`** on the agent enables the host KV tier (§4.4):
  evicted prefix-cache entries are demoted to host RAM instead of being
  discarded, so a returning session restores from host memory (~0.1 s) rather
  than re-prefilling from scratch (~35 s).
- `TimeoutStartSec=20min`: a cold blob cache means minutes of graph compile
  before the port answers, and systemd must not call that a hung start.
- A restart is a full reload of the model; `RestartSec=30` so a crash loop
  does not thrash the card.

Adjust `--model` to where your artifacts are, `--cache-dir` to a writable
place (it is the only thing the process writes), and the `Conflicts=` lines to
the units that actually share a card on your host.

A GGUF-opened unit is the same unit with `--gguf FILE` added and `--model`
naming the served IR of the same architecture as the template; it needs
`marfrit-openvino` at `+p7` or later (patch 0021; the release floor is `+p20`).
None of the units above is served that way: on the 24 GB card the GGUF of the
dense 27B is larger resident than the IR (16.54 against 13.06 GiB) and slower
at decode (FURTHER-READING.md, *Measured*); its case is serving the file's own
quantisation.
