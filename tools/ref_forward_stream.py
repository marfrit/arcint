"""The exact reference forward at FULL depth, streamed: the pin's own
`Qwen4ExpForCausalLM` at the checkpoint's geometry, every weight fed from
the GGUF (dequantised f32, `q4e.gguf_feed`), with the two tensors that do
not fit any host materialised lazily --

  * each layer's routed experts (10 GB f32 per layer) are loaded right
    before that layer's MoE runs and dropped right after (forward hooks on
    `mlp.experts`; the parameters live on the meta device in between);
  * the PLE's n-gram table (320M rows x 160, 191 GiB f32) is never built:
    the pin's `ngram_embedding` is replaced by a gather over the shard's
    own IQ4_NL rows (`q4e.native_blocks.iq4_nl_decode` of the 90-byte
    rows the ids hit), the same decode the serving-shape graph does.

Everything else is the pin, untouched: the GDN, the hyper-connections,
the full attention with its sparse path, the router, the shared expert,
the final mixer and norm, the separate head the checkpoint ships. That
makes it the yardstick without llama.cpp's Q8 activation quantisation
(campaign sub4bit-vram-kernel, 2026-09-18: that quantisation re-routes
tokens through this model's near-tied router, and the served artifact's
KLD against llama.cpp's capture is dominated by it).

Two products:
  --ids 760,6511,314,9338,369 --out DIR [--taps]
      the logits of a prompt (logits.npy [T, V]) and, with --taps, every
      layer's output as L{i}__out.npy [T, hc, H] (llama's l_last-i) plus
      final_norm.npy -- the deep ladder's reference.
  --capture CAPTURE.dat --write-capture REF.dat [--windows 0 1]
      replays the llama.cpp capture's windows (the same token ids) and
      writes a capture of the SAME format from this forward's logits, so
      `tools/kld_served.py --compare` reads the served dumps against it
      unchanged.

Memory: the dense model in f32 (~7 GB) + one layer's experts (10 GB) +
activations; a 128 GB unified-memory host runs the full depth. Rate is
set by the numpy dequant of each layer's experts (~1 min per layer on
8 cores).

  --layers N truncates the model (a smoke test at 2 layers covers the
  PLE); --dtype bf16 halves the weights and is NOT the exact reference.

[2026-09-28] Captures written before this date fed the sparse-attention
indexer's BF16 projections as their raw bytes (gguf_feed cast them to f32).
Their rows at or above position 2,051 do not carry the model's selection;
rows below it are unaffected. Re-capture before reading above the boundary.
"""
import argparse
import datetime
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hot_set_census import write_router_trace  # noqa: E402


def _lazy_table_class(torch, nn):
    class LazyNGramTable(nn.Module):
        """Stands in for the pin's `ngram_embedding` (an nn.Embedding the pin
        would allocate at 191 GiB): rows gathered from the shard's memmap and
        decoded on demand. `.weight` is a meta tensor so the pin's own device
        check leaves the ids where they are."""

        def __init__(self, raw_table, dim):
            super().__init__()
            self._raw = raw_table                          # (rows, 90) u8 memmap
            self._dim = dim
            self._torch = torch
            self.weight = torch.empty(0, device="meta")    # a plain attribute, not a parameter
            self.rows_gathered = 0

        def forward(self, ids):
            return _gather_rows(self, ids)
    return LazyNGramTable


def _gather_rows(self, ids):
    from q4e import native_blocks as nb
    torch = self._torch
    flat = ids.reshape(-1).cpu().numpy().astype(np.int64)
    uniq, inv = np.unique(flat, return_inverse=True)
    rows = np.ascontiguousarray(self._raw[uniq])                 # [n, 90] u8
    codes, scales = nb.iq4_nl_split(rows)
    vals = nb.iq4_nl_decode(codes, scales)                        # [n, 160] f32
    assert vals.shape[1] == self._dim, (vals.shape, self._dim)
    self.rows_gathered += int(len(uniq))
    out = torch.from_numpy(np.ascontiguousarray(vals[inv]))
    return out.reshape(*ids.shape, self._dim).to(ids.device)


def build_model(feed, config, device, dtype, torch, nn, log):
    from transformers.models.qwen4_exp import modeling_qwen4_exp as pin

    # 1. the two allocations that must not happen: experts and the n-gram
    #    table are constructed on the meta device
    orig_experts_init = pin.Qwen4ExpTextExperts.__init__
    orig_ngram_init = pin.Qwen4ExpTextNGramEmbedding.__init__

    def experts_init(self, cfg):
        with torch.device("meta"):
            orig_experts_init(self, cfg)

    def ngram_init(self, cfg, *a, **kw):
        with torch.device("meta"):
            orig_ngram_init(self, cfg, *a, **kw)
        # the index buffers are config arithmetic: rebuild them for real
        self.layer_multipliers = nn.Buffer(pin._build_layer_multipliers(
            self.unigram_vocab_size, self.ngram_size, self.ple_layer_index, self.seed))
        self.ngram_heads_vocab_sizes = nn.Buffer(torch.tensor(self.head_vocab_sizes, dtype=torch.long))
        self.ngram_heads_offsets = nn.Buffer(torch.tensor(self.head_offsets, dtype=torch.long))

    pin.Qwen4ExpTextExperts.__init__ = experts_init
    pin.Qwen4ExpTextNGramEmbedding.__init__ = ngram_init
    try:
        t0 = time.time()
        model = pin.Qwen4ExpForCausalLM(config).eval()
        log(f"[model] constructed in {time.time() - t0:.1f}s (experts and the n-gram table on meta)")
    finally:
        pin.Qwen4ExpTextExperts.__init__ = orig_experts_init
        pin.Qwen4ExpTextNGramEmbedding.__init__ = orig_ngram_init

    # 2. every real parameter from the GGUF
    t0 = time.time()
    fed = 0
    with torch.no_grad():
        for name, p in list(model.named_parameters()):
            if p.is_meta:
                continue
            key = name[len("model."):] if name.startswith("model.") else name
            arr = feed.fitted(key, tuple(p.shape))
            p.data = torch.from_numpy(np.ascontiguousarray(arr)).to(dtype)
            fed += 1
        if feed.has_lm_head():
            head = feed.fitted("lm_head.weight", tuple(model.lm_head.weight.shape))
            model.lm_head.weight = nn.Parameter(torch.from_numpy(np.ascontiguousarray(head)).to(dtype),
                                                requires_grad=False)   # untie: the checkpoint ships its own head
            log("[model] separate lm_head fed (the checkpoint's output.weight; the pin's tie broken)")
        else:
            log("[model] no separate head in the source: the pin's tie stands")
    log(f"[model] {fed} dense parameters fed in {time.time() - t0:.1f}s")

    # 3. the n-gram table: lazy rows for every PLE layer
    n_ple = 0
    for layer in model.model.layers:
        ple = getattr(layer, "ple", None)
        if ple is None:
            continue
        emb = ple.ple_embedding
        dim = emb.ngram_embedding.weight.shape[1]
        emb.ngram_embedding = _lazy_table_class(torch, nn)(feed.raw_table(), int(dim))
        n_ple += 1
    log(f"[model] {n_ple} PLE layer(s) on the lazy n-gram table")

    # 3b. the two runtime corrections of 2026-09-18 (DESIGN 7.0.2bz) that the
    #     pin does not carry: the GDN's output gate is a sigmoid (the pin's
    #     norm defaults to silu) and the 16 key heads serve the 48 value heads
    #     TILED (value head h <- key head h % 16; the pin interleaves, h // 3).
    #     Both are config-driven (`output_gate_type`, `gdn_key_head_map`, the
    #     exporter's keys), applied exactly as tools/ref_forward_real.py does.
    gate_act = getattr(config, "output_gate_type", None)
    n_gate = 0
    if gate_act:
        for layer in model.model.layers:
            gdn = getattr(layer, "linear_attn", None)
            if gdn is not None and hasattr(gdn, "norm") and hasattr(gdn.norm, "activation"):
                gdn.norm.activation = gate_act
                n_gate += 1
    khm = getattr(config, "gdn_key_head_map", None) or "interleave"
    hk, hv = int(config.linear_num_key_heads), int(config.linear_num_value_heads)
    if khm == "tiled" and hv % hk == 0 and hv > hk:
        r = hv // hk
        orig_chunk = pin.torch_chunk_gated_delta_rule

        def tiled_chunk(query, key, value, g, beta, *a, **kw):
            if query.shape[2] == hv:
                query = query[:, :, ::r].repeat(1, 1, r, 1)
                key = key[:, :, ::r].repeat(1, 1, r, 1)
            return orig_chunk(query, key, value, g, beta, *a, **kw)
        pin.torch_chunk_gated_delta_rule = tiled_chunk
    log(f"[model] GDN output gate {gate_act!r} on {n_gate} layer(s); key-head map {khm} "
        f"({hk} key heads, {hv} value heads)")

    # 4. to the device (meta stays meta)
    for name, p in model.named_parameters():
        if not p.is_meta:
            p.data = p.data.to(device)
    for name, b in model.named_buffers():
        if not b.is_meta:
            b.data = b.data.to(device)

    # 5. the expert stream
    stats = {"load_s": 0.0, "layers": 0}

    def make_hooks(i, experts):
        gu_key = f"layers.{i}.mlp.experts.gate_up_proj"
        d_key = f"layers.{i}.mlp.experts.down_proj"
        gu_shape = tuple(experts.gate_up_proj.shape)
        d_shape = tuple(experts.down_proj.shape)

        def pre(module, args):
            t0 = time.time()
            gu = feed.fitted(gu_key, gu_shape)
            d = feed.fitted(d_key, d_shape)
            module.gate_up_proj = nn.Parameter(torch.from_numpy(np.ascontiguousarray(gu)).to(dtype).to(device),
                                               requires_grad=False)
            module.down_proj = nn.Parameter(torch.from_numpy(np.ascontiguousarray(d)).to(dtype).to(device),
                                            requires_grad=False)
            stats["load_s"] += time.time() - t0
            stats["layers"] += 1

        def post(module, args, output):
            module.gate_up_proj = nn.Parameter(torch.empty(gu_shape, device="meta"), requires_grad=False)
            module.down_proj = nn.Parameter(torch.empty(d_shape, device="meta"), requires_grad=False)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        experts.register_forward_pre_hook(pre)
        experts.register_forward_hook(post)

    for i, layer in enumerate(model.model.layers):
        make_hooks(i, layer.mlp.experts)
    return model, stats


def write_capture(path, n_ctx, n_vocab, tokens, rows_u16):
    """A llama.cpp `--kl-divergence-base` file: magic, n_ctx, n_vocab, n_chunk,
    the tokens, then the rows (kld_served.read_capture is the reader)."""
    n_chunk = tokens.shape[0]
    with open(path, "wb") as f:
        f.write(b"_logits_")
        f.write(struct.pack("<iii", n_ctx, n_vocab, n_chunk))
        np.ascontiguousarray(tokens, dtype=np.int32).tofile(f)
        for w in range(n_chunk):
            np.ascontiguousarray(rows_u16[w], dtype=np.uint16).tofile(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--ids", default=None, help="comma-separated token ids (the prompt mode)")
    ap.add_argument("--capture", default=None, help="a llama.cpp --kl-divergence-base capture to replay")
    ap.add_argument("--write-capture", default=None, help="write this forward's capture of the replayed windows")
    ap.add_argument("--windows", type=int, nargs="*", default=None, help="which capture windows (default all)")
    ap.add_argument("--out", default=None, help="directory for logits.npy / taps")
    ap.add_argument("--taps", action="store_true", help="save every layer's output ([T, hc, H]) and the final norm")
    ap.add_argument("--router-trace", default=None,
                    help="write a format-v1 routed-expert trace of every layer's router "
                         "top-k for the prompt of --ids (device-free reference router)")
    ap.add_argument("--layers", type=int, default=None, help="truncate to the first N layers (smoke tests)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="f32", choices=("f32", "bf16"))
    args = ap.parse_args(argv)

    import torch
    from torch import nn
    from q4e import gguf_feed as gf
    from q4e.piecewise_export import real_config

    def log(msg):
        print(msg, flush=True)

    torch.manual_seed(0)
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "f32" else torch.bfloat16
    cfg = real_config()
    if args.layers is not None:
        cfg.num_hidden_layers = args.layers
        cfg.layer_types = list(cfg.layer_types[:args.layers])
        cfg.ple_layer_ids = [i for i in (cfg.ple_layer_ids or []) if i <= args.layers]
    cfg._attn_implementation = "eager"
    if hasattr(cfg, "tie_word_embeddings"):
        cfg.tie_word_embeddings = False
    log(f"[config] {cfg.num_hidden_layers} layers ({sum(1 for t in cfg.layer_types if t != 'linear_attention')} attention), "
        f"PLE at {cfg.ple_layer_ids}, device {device}, dtype {args.dtype}, "
        f"gate {getattr(cfg, 'output_gate_type', '?')}, key-head map {getattr(cfg, 'gdn_key_head_map', '?')}")

    t0 = time.time()
    feed = gf.GgufFeed(args.shards)
    log(f"[feed] over {args.shards} in {time.time() - t0:.1f}s; head declared: {feed.has_lm_head()}")
    model, stats = build_model(feed, cfg, device, dtype, torch, nn, log)

    taps = {}
    if args.taps:
        for i, layer in enumerate(model.model.layers):
            def hook(mod, inp, out, i=i):
                h = out[0] if isinstance(out, (tuple, list)) else out
                taps[f"L{i}__out"] = h.detach().float().cpu()
            layer.register_forward_hook(hook)
        final = getattr(model.model, "norm", None)
        if final is not None:
            final.register_forward_hook(
                lambda m, i, o: taps.__setitem__("final_norm", (o[0] if isinstance(o, (tuple, list)) else o).detach().float().cpu()))
        else:
            log("[taps] the text model has no `norm` attribute; final_norm not tapped")

    router_by_layer = {}
    if args.router_trace:
        def make_router_hook(i):
            def hook(mod, inp, out):
                # the pin's TopKRouter returns (logits, routing_weights,
                # selected_experts); the SparseMoeBlock unpacks output[2].
                if isinstance(out, (tuple, list)) and len(out) >= 3:
                    sel = out[2]
                else:
                    raise RuntimeError(
                        "router hook: expected the TopKRouter's (logits, weights, "
                        f"experts) tuple, got {type(out).__name__}")
                router_by_layer[i] = sel.detach().to(torch.long).cpu().numpy()
            return hook
        n = 0
        for i, layer in enumerate(model.model.layers):
            gate = getattr(getattr(layer, "mlp", None), "gate", None)
            if gate is not None:
                gate.register_forward_hook(make_router_hook(i))
                n += 1
        log(f"[router] {n} router hook(s) registered; trace -> {args.router_trace}")

    def forward(ids):
        T = len(ids)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attn = torch.ones(1, T, dtype=torch.long, device=device)
        stats["load_s"] = 0.0
        stats["layers"] = 0
        t0 = time.time()
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logits = out.logits[0].float()
        log(f"[forward] T={T} in {time.time() - t0:.1f}s (expert loads {stats['load_s']:.1f}s over {stats['layers']} layers); "
            f"logits {tuple(logits.shape)} finite={bool(torch.isfinite(logits).all())} absmax={float(logits.abs().max()):.3f}")
        return logits.cpu()

    if args.ids:
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
        out = Path(args.out or "ref-stream-out")
        out.mkdir(parents=True, exist_ok=True)
        logits = forward(ids)
        np.save(out / "logits.npy", logits.numpy())
        H = cfg.hidden_size
        for k, v in taps.items():
            a = v.numpy()
            if k.startswith("L"):
                a = a.reshape(len(ids), cfg.hc_count, H)
            else:
                a = a.reshape(len(ids), -1)
            np.save(out / f"{k}.npy", a)
            log(f"[tap] {k:<14} {str(a.shape):<16} sum {a.sum():+.6f} absmax {np.abs(a).max():.4f}")
        top = torch.topk(logits[-1], 5).indices.tolist()
        log(f"[done] logits under {out}; last-token top-5 {top}")
        if args.router_trace:
            provenance = {
                "artifact_sha256": "none",
                "source": "reference-f32-router",
                "card": "none",
                "device": str(device),
                "depth": str(len(router_by_layer)),
                "dtype": args.dtype,
                "ids": str(len(ids)),
                "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "tool": "ref_forward_stream.py",
            }
            write_router_trace(args.router_trace, router_by_layer, len(ids), provenance)
            log(f"[router] wrote {args.router_trace}: {len(ids)} tokens x "
                f"{len(router_by_layer)} layers")
        return 0

    if args.capture:
        from kld_served import read_capture, llama_row
        n_ctx, n_vocab, n_chunk, tokens, _rows = read_capture(args.capture)
        which = args.windows if args.windows is not None else list(range(n_chunk))
        first = n_ctx // 2
        n_rows = n_ctx - 1 - first
        nv = 2 * ((n_vocab + 1) // 2) + 4
        rows_out = np.zeros((len(which), n_rows, nv), dtype=np.uint16)
        for wi, w in enumerate(which):
            ids = [int(t) for t in tokens[w]]
            logits = forward(ids).numpy()
            assert logits.shape == (n_ctx, n_vocab), logits.shape
            for i in range(n_rows):
                rows_out[wi, i] = llama_row(logits[first + i])
            log(f"[capture] window {w}: {n_rows} rows written from positions {first}..{n_ctx - 2}")
        if args.write_capture:
            write_capture(args.write_capture, n_ctx, n_vocab, tokens[which], rows_out)
            log(f"[done] capture {args.write_capture}: n_ctx {n_ctx}, {len(which)} window(s), tokens identical to the source")
        return 0

    ap.error("one of --ids or --capture is required")


if __name__ == "__main__":
    sys.exit(main())
