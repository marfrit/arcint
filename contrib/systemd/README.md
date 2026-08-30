# Two cards, two engines: the dev fleet's units as an example

These are the user units one host runs in production, with the operator-local
detail taken out. `packaging/arcint.service` is the *template* the package
installs; these are what it looks like once two cards and two models are on
one machine. Every flag is literal in `ExecStart` on purpose: a unit manager
and a journal can both read the port, the served name and the context there.

| unit | card | model | port |
|---|---|---|---|
| `arcint-agent.service` | GPU.0 (Arc Pro B60, 24 GB) | Qwen3.6-35B-A3B, Intel's public int4 IR | 8087 |
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
- **The reservation decides the context, not the flag.** `--n-ctx 262144` on
  the 16 GB card serves 98304 with the chunk clamped to 128; that is the
  reservation arithmetic on that card, printed at boot, not a defect.
- **`--gate-pad 16` on the agent, not on the coder.** Prefill-heavy traffic
  (agents re-sending long conversations) gains 13% on prefill and pays ~5% on
  decode; one-shot coding traffic is decode-bound and leaves it off. DESIGN
  7.0.2g has the break-even.
- **`--paged-kv u8`** halves the KV pages; it is the default and is byte-equal
  on the acceptance task.
- `TimeoutStartSec=20min`: a cold blob cache means minutes of graph compile
  before the port answers, and systemd must not call that a hung start.
- A restart is a full reload of the model; `RestartSec=30` so a crash loop
  does not thrash the card.

Adjust `--model` to where your artifacts are, `--cache-dir` to a writable
place (it is the only thing the process writes), and the `Conflicts=` lines to
the units that actually share a card on your host.
