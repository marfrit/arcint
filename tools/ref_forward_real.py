"""A REAL-GEOMETRY reference forward of the first N decoder layers, from the
shipped GGUF, with a tap on every block -- the yardstick the cut ladder of
`boot_serving_shape.py --cut` is read against (campaign serving-shape-logits).

What it is: the pin's own modules (`q4e.ref_backbone.Qwen4ExpTextBackbone`,
which wires `modeling_qwen4_exp`'s leaves) built at the checkpoint's geometry
(`piecewise_export.real_config`), truncated to `--layers`, every weight fed
from the GGUF through `q4e.gguf_feed` (dequantised f32, the same name map the
exporter fills from), run in f32 on the CPU over the given token ids with an
all-valid conv mask. The forward loop is the reference's own (pin 1283-1309),
unrolled here so each block's output can be saved:

    emb            [T, H]      embed_tokens rows
    hc_init        [T, hc, H]  the residual streams before layer 0
    L{i}/mix_attn  [T, H]      attn_hyper_connection's mixed input to the block
    L{i}/attn_out  [T, H]      the GDN (or attention) block's output
    L{i}/combine_attn [T, hc, H]
    L{i}/mix_mlp   [T, H]
    L{i}/ffn_out   [T, H]      the MoE block's output (routed + shared)
    L{i}/out       [T, hc, H]  the residual streams after layer i

each as `<out>/<name>.npy` (slashes become double underscores). A layer that
is a full-attention layer is refused here (this reference is GDN-only; the
attention emitter has its own harness), so `--layers` must stay below the
first attention layer (3). The PLE at layer 1 needs the row ids; those are
computed with `q4e.ngram_ids` exactly as the boot driver computes them.

Reading it: llama.cpp's `llama-eval-callback` prints the same tensors of the
same graph (`hc_mixed-i`, `linear_attn_out-i`, `hc_combine-i`, `ffn_out-i`,
`l_last-i`) with their sums and 3-element corners; the artifact's cut dumps
hold the whole tensor. Three-way: where this reference and llama.cpp agree
and the artifact does not, the emitter or the fill is wrong at that block.

  ref_forward_real.py --shards DIR --ids 760,6511,314,9338,369 --layers 1 --out DIR
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--ids", required=True, help="comma-separated token ids")
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--refold-norms", default="",
                    help="comma-separated pin-key suffixes whose fed gamma gets 1.0 ADDED "
                         "back: the feed undoes the converter's (1 + w) fold since "
                         "2026-09-18 (gguf_feed kind gamma1), so this re-creates the "
                         "AS-STORED experiment that found it (the pin then applies "
                         "(1 + w) on top of the stored (1 + w))")
    ap.add_argument("--a-as-stored", action="store_true",
                    help="feed linear_attn.A_log as the GGUF stores ssm_a, -exp(A_log): "
                         "the feed undoes that fold (kind neglog), so this re-creates the "
                         "AS-STORED experiment that found it")
    ap.add_argument("--k-head-map", default=None, choices=(None, "interleave", "tiled"),
                    help="experiment: how the 16 key heads serve the 48 value heads in "
                         "layer 0's GDN core -- the pin's repeat_interleave (value head h "
                         "<- key head h // 3) or llama.cpp's tiling (h <- h %% 16)")
    ap.add_argument("--gate-act", default=None,
                    help="experiment: override the activation of layer 0's GDN gated "
                         "norm (the pin's `norm.activation`, e.g. sigmoid or silu)")
    args = ap.parse_args(argv)

    import torch
    from q4e import gguf_feed as gf
    from q4e import ngram_ids as nid
    from q4e.piecewise_export import real_config
    from q4e.ref_backbone import Qwen4ExpTextBackbone

    ids = [int(x) for x in args.ids.split(",") if x.strip()]
    T = len(ids)
    cfg = real_config()
    L = int(args.layers)
    if L < 1 or L > len(cfg.layer_types):
        raise SystemExit(f"--layers {L} outside 1..{len(cfg.layer_types)}")
    kinds = list(cfg.layer_types[:L])
    if any(k != "linear_attention" for k in kinds):
        raise SystemExit(f"layers 0..{L - 1} hold a full-attention layer ({kinds}); "
                         "this reference is GDN-only")
    cfg.num_hidden_layers = L
    cfg.layer_types = kinds

    t0 = time.time()
    feed = gf.GgufFeed(args.shards)
    print(f"[feed] over {args.shards} in {time.time() - t0:.1f}s; head declared: "
          f"{feed.has_lm_head()}", flush=True)

    torch.manual_seed(0)
    ref = Qwen4ExpTextBackbone(cfg, declare_lm_head=feed.has_lm_head()).eval()
    sd = ref.state_dict()
    t0 = time.time()
    fed = 0
    for k in list(sd):
        if any(k.endswith(s) for s in gf._DERIVED_SUFFIXES):
            continue
        arr = feed.fitted(k, tuple(sd[k].shape))
        sd[k] = torch.from_numpy(np.ascontiguousarray(arr)).to(sd[k].dtype)
        fed += 1
    refold = tuple(x for x in args.refold_norms.split(",") if x)
    if refold:
        hit = [k for k in sd if k.endswith(refold)]
        for k in hit:
            sd[k] = sd[k] + 1.0
        print(f"[refold] 1.0 added back to {len(hit)} fed gamma(s) (as stored): {hit}", flush=True)
    if args.a_as_stored:
        hit = [k for k in sd if k.endswith("linear_attn.A_log")]
        for k in hit:
            sd[k] = -torch.exp(sd[k])
        print(f"[a-as-stored] A_log <- -exp(A_log) for {len(hit)} vector(s): {hit}", flush=True)
    ref.load_state_dict(sd)
    del sd
    print(f"[feed] {fed} keys fed at the real geometry in {time.time() - t0:.1f}s "
          f"(depth {L}, T={T})", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    taps = {}

    def tap(name, t):
        x = t.detach().float().numpy()
        if x.ndim >= 3 and x.shape[0] == 1 and x.shape[1] == T:
            a = x.reshape(T, *x.shape[2:])
        elif x.ndim == 2 and x.shape[0] == T:
            a = x
        else:
            a = x                                   # e.g. a conv output [C, T'] or [1, C, T']
        a = np.ascontiguousarray(a, dtype=np.float32)
        taps[name] = a
        np.save(out / (name.replace("/", "__") + ".npy"), a)
        print(f"[tap] {name:<20} {str(a.shape):<16} sum {a.sum():+.6f} absmax {np.abs(a).max():.4f}",
              flush=True)

    # inner taps of layer 0's GDN, by forward hooks on the pin's submodules --
    # llama.cpp's names beside each: in_proj_qkv -> linear_attn_qkv_mixed,
    # in_proj_z -> z, in_proj_a -> alpha, in_proj_b -> beta (pre-sigmoid),
    # conv1d -> conv_output_raw (before the silu, over the padded sequence),
    # norm -> final_output (the gated norm, before out_proj), out_proj ->
    # linear_attn_out
    gdn0 = ref.layers[0].linear_attn
    if args.gate_act:
        print(f"[gate-act] layer 0 GDN norm activation {gdn0.norm.activation!r} -> {args.gate_act!r}",
              flush=True)
        gdn0.norm.activation = args.gate_act
    # the conv's output and the core's inputs, through the pin's module-level
    # functions (the forward calls them by global name, so a wrapper installed
    # on the module is what it calls): conv_silu -> llama conv_output_silu-i;
    # the core's q/k/v/g/beta after the head expansion, before the kernel's
    # own L2 norm -> llama's q/k/v_conv_predelta-i, gate-i, beta_sigmoid-i
    import q4e.ref_backbone as _rb
    _pin = _rb._pin
    # the module-level functions are called once per GDN layer, in layer
    # order; the call index names the tap's layer (a reviewer caught the
    # earlier version writing every layer under L0)
    calls = {"conv": 0, "chunk": 0}
    _orig_conv = _pin.causal_conv1d_fn
    def _conv_tap(*a, **kw):
        o = _orig_conv(*a, **kw)
        tap(f"L{calls['conv']}/gdn/conv_silu", o[0] if isinstance(o, tuple) else o)
        calls["conv"] += 1
        return o
    _pin.causal_conv1d_fn = _conv_tap
    khm = args.k_head_map or getattr(cfg, "gdn_key_head_map", None) or "interleave"
    hk, hv = int(cfg.linear_num_key_heads), int(cfg.linear_num_value_heads)
    print(f"[k-head-map] {khm} ({'flag' if args.k_head_map else 'the real config'}; "
          f"{hk} key heads, {hv} value heads)", flush=True)
    _orig_chunk = _pin.torch_chunk_gated_delta_rule
    def _chunk_tap(query, key, value, g, beta, *a, **kw):
        li = calls["chunk"]
        calls["chunk"] += 1
        if khm == "tiled" and query.shape[2] == hv and hv % hk == 0 and hv > hk:
            r = hv // hk
            # undo the pin's interleave (head r*h <- key head h), then tile
            query = query[:, :, ::r].repeat(1, 1, r, 1)
            key = key[:, :, ::r].repeat(1, 1, r, 1)
        tap(f"L{li}/gdn/q_in", query); tap(f"L{li}/gdn/k_in", key); tap(f"L{li}/gdn/v_in", value)
        tap(f"L{li}/gdn/g_in", g); tap(f"L{li}/gdn/beta_in", beta)
        return _orig_chunk(query, key, value, g, beta, *a, **kw)
    _pin.torch_chunk_gated_delta_rule = _chunk_tap
    # the gated norm's INPUT is the delta-rule core's output (llama: attn_output-i)
    gdn0.norm.register_forward_pre_hook(
        lambda m, inp: tap("L0/gdn/core_out", inp[0]))
    for sub in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "conv1d", "norm", "out_proj"):
        mod = getattr(gdn0, sub, None)
        if mod is None:
            continue
        mod.register_forward_hook(
            lambda m, inp, o, sub=sub: tap(f"L0/gdn/{sub}", o[0] if isinstance(o, tuple) else o))

    hc = cfg.hc_count
    H = cfg.hidden_size
    input_ids = torch.tensor([ids], dtype=torch.long)
    conv_mask = torch.ones(1, T)
    with torch.no_grad():
        emb = ref.embed_tokens(input_ids)                                    # [1,T,H]
        tap("emb", emb)
        hidden = emb.repeat(1, 1, hc)                                        # [1,T,hc*H]
        tap("hc_init", hidden.reshape(1, T, hc, H))
        for i, layer in enumerate(ref.layers):
            if layer.ple is not None:
                hidden = hidden + layer.ple(hidden, input_ids, None, conv_mask=conv_mask)
                tap(f"L{i}/ple_out", hidden.reshape(1, T, hc, H))
            h, hyper, inj = layer.attn_hyper_connection(hidden)
            tap(f"L{i}/mix_attn", h)
            g = layer.linear_attn(h, cache_params=None, attention_mask=conv_mask)
            tap(f"L{i}/attn_out", g)
            hidden = hyper + (g.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
            tap(f"L{i}/combine_attn", hidden.reshape(1, T, hc, H))
            h, hyper, inj = layer.mlp_hyper_connection(hidden)
            tap(f"L{i}/mix_mlp", h)
            m = layer.mlp(h)
            tap(f"L{i}/ffn_out", m)
            hidden = hyper + (m.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
            tap(f"L{i}/out", hidden.reshape(1, T, hc, H))
    print(f"[done] {len(taps)} taps under {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
