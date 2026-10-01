# 0.3.0 — the extension series M7–M14 (closed, tagged 2026-09-05)

| item | what exists |
|---|---|
| M7 fit pass | the paged path's measured reservation; `--n-ctx` omitted auto-fits, given explicitly it is verify-only |
| M8 asymmetric KV | `--paged-kv u8:i4` (patches 0008–0010, 0020); `docs/design-m8-asymmetric-kv.md` |
| M9 expert offload v2 | device-resident slot pool and async batched uploads (patches 0004–0006) |
| M10 sub-4-bit experts | carried by the `sub4bit-vram-kernel` campaign: the checkpoints' native expert formats serve (patches 0043 onward) |
| M11 drafting II | drafter fixes at depth: the MTP layer's KV charged to the reservation, the drafters' rotary kept f32 |
| M12 dispatch pin, tiled exporter | `--pin-dispatch`; exporter `--moe-lowering tiled` |
| M13 vision reserved | `--vision` refused; vision IRs reported at load, never loaded |
| M14 host compute tier | `--moe-cpu-tier` (patches 0011–0013, 0018) |

Open work from the 0.3.0 backlog is one campaign per lever under
`docs/campaigns/`.

Full history: `git show b0447b8:docs/milestone-0.3.0.md`.
