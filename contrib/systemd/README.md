# Two cards, two engines: the dev fleet's units as an example

These are the user units one host runs in production, with the operator-local
detail taken out. `packaging/arcint.service` is the *template* the package
installs; these are what it looks like once two cards and two models are on
one machine. Every flag is literal in `ExecStart` on purpose: a unit manager
and a journal can both read the port, the served name and the context there.

| unit | card | model | port |
|---|---|---|---|
| `arcint-agent.service` | GPU.0 (Arc Pro B60, 24 GB) | Qwen3.8-27B dense, Intel's public int4 IR, MTP on | 8087 |
| `arcint-coder.service` | GPU.1 (Arc A770, 16 GB) | Qwen3.6-27B-A3B coder (the b5 export) | 8080 |
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
- **The reservation decides the context, not the flag.** `--n-ctx 151552` on
  the 24 GB card and `--n-ctx 131072` on the 16 GB card are the flags; the
  actual served context is whatever the reservation arithmetic on that card
  allows, printed at boot, not a defect.
- **`--gate-pad 16` is a no-op on a dense model.** It matters only for MoE
  traffic patterns (the old 35B agent); the dense 27B agent omits it. On the
  coder (also dense, one-shot coding traffic that is decode-bound) it was never
  set. DESIGN 7.0.2g has the break-even for MoE.
- **`--paged-kv u8:i4`** on the agent (half-precision keys, quarter-precision
  values) extends the context window; `--paged-kv u8` on the coder halves the
  KV pages. Both are byte-equal on the acceptance task.
- **`--prefill-chunk 512`** on the agent keeps prompt ingestion chunked so
  chunked-prefill scheduling works; the coder leaves it at the default.
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

A GGUF-opened unit (0.4.0 stage 1) is the same unit with `--gguf FILE` added
and `--model` naming the served IR of the same architecture as the template;
it needs `marfrit-openvino` at `+p7` (patch 0021). None of the units above is
served that way yet: the K-quant path prefills at chunk 256 and decodes at
about half the IR's rate on the 24 GB card (*FURTHER-READING.md*'s *Measured*
section), which is 0.4.1's work.
