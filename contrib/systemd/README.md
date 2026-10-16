# Two cards, four services: the dev fleet's units as an example

These are the user units one host runs in production, with the operator-local
detail taken out. `packaging/arcint.service` is the *template* the package
installs; these are what it looks like once two cards and several models are
on one machine. Every flag is literal in `ExecStart` on purpose: a unit
manager and a journal can both read the port, the served name and the
context there. Each unit's header comment carries what was measured with
exactly its flags.

| unit | card | model | port |
|---|---|---|---|
| `arcint-agent.service` | GPU.0 (Arc Pro B60, 24 GB) | Qwen3.8-27B dense, Q4_K_M GGUF, q8_0:q4_0 KV, 245,760 tokens, MTP 5 | 8087 |
| `arcint-writer.service` | GPU.0, on demand | Cydonia 24B v4.3 (Mistral Small 3.2 finetune), imatrix Q4_K_M GGUF, q8_0 KV, 98,304 tokens | 8088 |
| `arcint-flashnext.service` | GPU.0, on demand | Qwen3.8-Flash-Next IQ3_XXS (since 2026-10-15): expert cache (12,000 MiB of slots, the rest in pinned host memory), the CPU misses, the bank stream for prefill, MTP 3 with Strata's stop, 131,072 tokens, q8_0 KV | 8089 |
| `arcint-coder.service` | GPU.1 (Arc A770, 16 GB) | Qwen3.6-27B-A3B coder, the searched `c5f495ac shq8` GGUF, q8_0:q4_0 KV, 98,304 tokens, MTP 4 | 8080 |

The GPU.0 units take turns on one card (`Conflicts=`); on the host they come
from, a unit manager switches between them on demand. Flash-Next's other
quants run from `arcint-flashnext.service` with their own `--gguf` and slot
budget: IQ2_XS at 13,400 MiB of slots (faster, KL 0.294 against IQ3_XXS's
0.218), UD-Q3_K_XL once the host has 96 GB (`docs/llama-engine.md`,
"Flash-Next quant options").

Things worth copying rather than re-learning:

- **Size the VRAM for what a request takes, not for what is free after the
  load.** On the B60 the first request keeps ~0.9 GiB and a 20k prefill
  needs ~3 GiB more for a while; with less room, xe evicts buffers to host
  memory and leaves them there (a verify took 4.5 s instead of ~25 ms). Each
  unit's context and slot budget were set from the peak at its deepest
  admitted prompt. Watch the process's `drm-resident-gtt` in
  `/proc/<pid>/fdinfo`: above the bank's size, something was evicted.
- **`Conflicts=`, not arithmetic, keeps a card exclusive.** A resident model
  holds its VRAM for the process lifetime, and a second process loading
  beside it fails at best and faults the card at worst.
- **`--served-model-name` names the endpoint.** A proxy pins its roster to
  it and reads the context from `/v1/models`; the GGUF behind it can change
  without the clients noticing.
- **`--queue-timeout 30`**: a request for a busy lane waits that long and
  then gets a 503 with the lane's numbers, instead of an immediate refusal.
- **Flash-Next needs its host memory free**: the bank (28.72 GiB for IQ3_XXS)
  is pinned, beside the page cache of its GGUF (47.0 GB for the IQ3_XXS shard 1) and the 27 GiB n-gram
  table read lazily. Serve the GGUF from local NVMe, not a network or ZFS
  pool: the first answers read about 1 GB from storage.
- `TimeoutStartSec=20min` (30 for Flash-Next): loading Flash-Next fills the
  slots and the bank from the file, several minutes.
- A restart is a full reload of the model; `RestartSec=30` so a crash loop
  does not thrash the card.

Adjust `--gguf` to where your files are and the `Conflicts=` lines to the
units that actually share a card on your host.
