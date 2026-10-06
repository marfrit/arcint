#!/usr/bin/env python3
"""THE EXPERT STORE: one ext4 file per expert, in the device's own byte order.

Campaign: `docs/campaigns/nvme-direct-expert-tier.md` (0.5.3 LISBON), the
artifact-format step. Design: `docs/design-nvme-direct-expert-tier.md` §3/§7.4.
This tool resolves the *scales/zp store-layout precondition* the B60 probe left
OWED and builds the pinned expert set the load-time fill needs.

Evidence discipline. Every claim below carries `paper` / `code` / `measured-here`
and names its source. Nothing is asserted below what the sources support.

--------------------------------------------------------------------------
1. WHAT ONE EXPERT SLICE IS, AND WHERE ITS BYTES COME FROM
--------------------------------------------------------------------------

One Flash-Next expert layer is 2,457,600 B (`code`:
`src/exec/flash_next_offload.h:45`; re-derived in
`docs/design-qwen-flash-next.md` Fit table):
`3 * hidden * moe_intermediate * bytes_per_weight`
= `3 * 2560 * 640 * 0.5 = 2,457,600 B` -- the three u4 weight matrices gate,
up and down, and NOTHING else. Scales and zero-points are separate tensors in
the plugin's OTD weight file and are NOT part of this slice.

The bytes are produced by the existing fill machinery
(`q4e.expert_fill`): `ExpertFiller` reads the GGUF through `q4e.gguf_feed`,
quantises each expert row to the u4 grouped-affine form the tiled lowering
declares, and packs it with the C++ contract
(`src/core/gguf_repack.h:90`, executed at `src/core/gguf_repack.cpp:225`):
element `idx` is the LOW nibble of byte `idx/2` when `idx` is even, the HIGH
nibble when odd. This module does not re-implement any of that; it consumes it
and lays the result down.

--------------------------------------------------------------------------
2. THE LAYOUT VERDICT: THE SLICE IS THE THREE WEIGHT TENSORS, DEVICE ORDER
--------------------------------------------------------------------------

`code`: the plugin's device slot layout and the OTD weight-file layout are
IDENTICAL for the weights (`[oc][ic]`, row-major, inn contiguous, two codes a
byte with the even index low) and DIFFERENT for scales/zero-points -- the
device wants scales `[group][oc]` and zp `[group][oc/2]`, the file carries
`[oc][group]`, and the plugin's `maybe_transpose_scale_zp` transposes on upload
(`contrib/packaging/marfrit-openvino/patches/0011-…:75-90`, `0006-…:291`).

`code` arithmetic, the reason the slice excludes scales/zp:

    weights            = 2,457,600 B = 600 x 4096 B pages = 4,800 x 512 B LBAs
    scale/zp, serving-shape group 128:
      scales           = (640x20 + 640x20 + 2560x5) x 2 B = 76,800 B
      zero-points      = (640x20 + 640x20 + 2560x5) / 2   = 19,200 B
      weights + scale + zp = 2,553,600 B = 623.4375 pages -> NOT page-aligned

`serving_shape.EXPERT_GROUP_SIZE = 128` is the exporter's group; the plugin's
production OTD group is narrower (32 gate/up, 8 down, `code`:
`patches/0011-…:75-90`), which changes the byte count but not the verdict
(g=64 gives 2,611,200 B = 637.5 pages, still unaligned). arcwell requires
`in_dest_offset` and the transfer length to be page-aligned (`code`:
`USING_ARCWELL.md` §6).

**Verdict (chosen).** The store file is the three packed u4 WEIGHT tensors,
concatenated in the order gate | up | down, each `[oc][ic]` row-major with the
C++ nibble contract. These bytes are byte-identical in the file and on the
device, so a naive full-slice DMA is byte-transparent -- **given the integration
contract that the per-expert device BO is the concatenated record** (the plugin
today keeps three separate per-tensor memories; the contiguous gate|up|down BO
is the D2/D3 integration's layout, stated here so it is not mistaken for an
existing device layout). Scales and zero-points are NOT in the DMA slice: they
stay on the existing host upload path, which already applies
`maybe_transpose_scale_zp`. This is the design note §3's second option ("keep
arcwell to the weight tensors and move the small scale/zp tensors through the
existing host path").

The alignment arithmetic is decisive but not the only conceivable route: one
could pad the record or give scales/zp their own page-aligned files. The chosen
route is the only one that keeps the DMA ONE page-aligned request at the pinned
2,457,600-byte slice WITHOUT changing the plugin's device-slot offsets -- and
the plugin's CPU kernel and existing host path read the FILE `[oc][group]`
order, so keeping scales/zp on that path needs no new indexing (`code`:
`patches/0011-…:79-80`).

The alternative -- a store file that carries scales/zp already in DEVICE order
so a naive DMA would land them too -- is implemented for the case a future
integration wants it (`--with-scale-zp`, and `transpose_scale_zp_to_device`,
which transcribes `maybe_transpose_scale_zp`'s mapping
`src[o*group_count+g] -> dst[g*oc+o]`). It is not the default because it cannot
be one page-aligned request; the writer-level cell exercises the sidecar path.

--------------------------------------------------------------------------
3. GEOMETRY: fallocate, ONE EXTENT, PAGE AND LBA RULES
--------------------------------------------------------------------------

`code` (`USING_ARCWELL.md` §5/§6): one file per expert, `fallocate`d, gives
exactly ONE extent, so an expert is one request at one `in_dest_offset`
(it is still ~3 bios: a bio holds `BIO_MAX_VECS` = 256 pages = 1 MiB).
The file must be page-aligned in length and a whole number of 512 B logical
blocks. 2,457,600 B is 600 pages and 4,800 LBAs (`measured-here`, recon).

`write_expert_file` therefore:
  1. `os.posix_fallocate(fd, 0, size)` FIRST, so the blocks are allocated
     contiguously and the inode's bytes are real on disk, not sparse;
  2. then writes the payload into the preallocated blocks;
  3. refuses a payload whose length is not exactly the declared slice size.
A plain buffered write can leave a hole or interleave fragments; a test asserts
the fallocate call happened (red-first against a truncate/write mutation).

--------------------------------------------------------------------------
4. INPUT: WHICH EXPERTS, CHOSEN BY WHOM
--------------------------------------------------------------------------

Which experts are resident is the RESIDENCY POLICY's business, explicitly OUT
of scope for the campaign (Scope section). This tool only materialises the set
it is handed:

  --pinned PATH    a JSON object `{"<decoder_layer>": [expert_id, ...], ...}`;
                   OR a "hot-set seed v2" text file (patch 0046), whose lines
                   are `<layer_key> <expert> ...` and which needs
                   --layer-keys (a JSON map `{"<decoder_layer>": layer_key}`);
  --all            every expert of every layer (512 x 48 = 24,576 files,
                   60.4 GB -- must actually fit).

At the gate's ratio 86 the pool is `512*(100-86)/100 = 71` slots/layer
(`code`: patches 0041/0047), i.e. 71 x 48 = 3,408 expert files = 8,375,500,800 B
= 8.38 GB. That is the set the load-time fill needs; the exact membership is a
(seed x layer_key) property and is recorded in the store's manifest.

--------------------------------------------------------------------------
5. OUTPUT CONTRACT (what the arcwell client reads at open)
--------------------------------------------------------------------------

The store directory carries `manifest.json`: one record per file with
`layer`, `expert`, `file`, `bytes`, `sha256`, and the layout string. The client
maps each pinned `(layer, expert)` to the file, runs FIEMAP once, and DMA's the
file whole into the device slot at offset 0 of a 64 KiB-rounded BO.

`--dry-run` prints the plan and writes nothing.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

from . import expert_fill as ef

# Flash-Next MoE geometry (`code`: src/exec/flash_next_offload.h:45-48;
# docs/design-qwen-flash-next.md Fit table).
FLASH_NEXT_HIDDEN = 2560
FLASH_NEXT_FF = 640
ROLE_ORDER = ("gate", "up", "down")

# One expert slice, weights only. `code` arithmetic over the geometry above.
EXPERT_SLICE_BYTES = 3 * FLASH_NEXT_HIDDEN * FLASH_NEXT_FF // 2       # 2,457,600

# The serving-shape exporter's group size (`code`: serving_shape.EXPERT_GROUP_SIZE).
DEFAULT_GROUP_SIZE = 128

# The per-role output widths at real geometry.
ROLE_OUT = {"gate": FLASH_NEXT_FF, "up": FLASH_NEXT_FF, "down": FLASH_NEXT_HIDDEN}
ROLE_IN = {"gate": FLASH_NEXT_HIDDEN, "up": FLASH_NEXT_HIDDEN, "down": FLASH_NEXT_FF}

# The layout string the manifest and the tests share: device order, weights
# only. Changing it is a contract change, so it is named once.
LAYOUT = "gate|up|down u4 [oc][ic] row-major, even-linear-index low nibble; scales/zp not in slice"


def role_byte_count(role):
    """Packed u4 bytes for one expert of `role` at real geometry."""
    return ROLE_OUT[role] * ROLE_IN[role] // 2


def slice_byte_count():
    return sum(role_byte_count(r) for r in ROLE_ORDER)


def device_order_weight_record(packed_by_role):
    """Concatenate the three packed u4 weight tensors in device order.

    `packed_by_role` maps role -> a uint8 array whose flattened C order is
    `[out][inn/2]` for one expert. The result is `gate | up | down`, exactly
    the bytes the device slot holds for the weights, so a verbatim DMA is
    byte-transparent.
    """
    parts = []
    for role in ROLE_ORDER:
        arr = np.ascontiguousarray(packed_by_role[role], dtype=np.uint8).ravel()
        want = role_byte_count(role)
        if arr.size != want:
            raise ValueError(
                f"{role}: {arr.size} packed bytes, the device slot holds {want} "
                f"({ROLE_OUT[role]} x {ROLE_IN[role]} u4)")
        parts.append(arr)
    return np.concatenate(parts)


def transpose_scale_to_device(scale):
    """File-order scales `[oc][group_count]` f16 bits -> device `[group][oc]`.

    Transcribes `maybe_transpose_scale_zp`'s mapping
    `src[o*group_count+g] -> dst[g*oc+o]` (`code`: patch 0011 header;
    `moe_otd_runtime.cpp`). `scale` MUST be a 2-D array of raw uint16 f16 bits;
    a float array is refused rather than silently re-emitted at 4 B/element.
    """
    a = np.asarray(scale)
    if a.ndim != 2:
        raise ValueError(f"scales must be 2-D [oc][group_count], got {a.shape}")
    if a.dtype != np.uint16:
        raise ValueError(
            f"transpose_scale_to_device wants raw uint16 f16 bits, got {a.dtype}; "
            f"convert with `.astype(np.float16).view(np.uint16)` first")
    oc, gc = a.shape
    return np.ascontiguousarray(np.transpose(a, (1, 0))).reshape(gc * oc)


def transpose_scale_zp_to_device(scale, zp_codes):
    """Device-order `(scales f16 bits, zp packed bytes)` for one expert.

    `scale` is f32 `[oc][group_count]` as the fill produces it; it is rounded
    to f16 (the artifact's carried precision, `code`: `serving_shape`'s f16
    scale Constant) and handed to `transpose_scale_to_device` as raw bits.
    `zp_codes` is the file-order u4 code matrix `[oc][group_count]`.
    """
    s = np.ascontiguousarray(np.asarray(scale, dtype=np.float32))
    if s.ndim != 2:
        raise ValueError(f"scales must be 2-D [oc][group_count], got {s.shape}")
    s16 = s.astype(np.float16).view(np.uint16)
    return transpose_scale_to_device(s16), transpose_zp_to_device(zp_codes)


def transpose_zp_to_device(zp_codes):
    """File-order zp CODES `[oc][group_count]` -> device `[group][oc/2]` bytes.

    The file packs element `(o,g)` at linear index `o*gc+g`, even index low
    nibble. The device holds it `[group][oc]`, but packed two-per-byte along
    `oc`, i.e. linear index `g*oc+o` with the even index low. So: transpose the
    codes to `[group][oc]`, flatten C order, repack.
    """
    codes = np.asarray(zp_codes)
    if codes.ndim != 2:
        raise ValueError(f"zp codes must be 2-D [oc][group_count], got {codes.shape}")
    oc, gc = codes.shape
    dev = np.ascontiguousarray(np.transpose(codes, (1, 0))).reshape(gc * oc)
    return ef.pack_u4(dev)


def write_expert_file(path, payload, size=None):
    """Create `path` with `fallocate`, then fill it with `payload`.

    `payload` must be exactly `size` bytes when `size` is given (default: the
    payload length). A short or long payload is a caller bug and is refused --
    a silently padded file would fill a device slot with garbage.
    """
    payload = np.ascontiguousarray(payload, dtype=np.uint8).ravel()
    n = int(payload.size if size is None else size)
    if payload.size != n:
        raise ValueError(
            f"{path}: payload is {payload.size} B, the declared slice is {n} B")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        # posix_fallocate allocates real blocks contiguously (ext4 writes an
        # unwritten extent, which fiemap reports as a plain extent). A plain
        # truncate would leave a hole and no extent.
        os.posix_fallocate(fd, 0, n)
        os.lseek(fd, 0, os.SEEK_SET)
        written = os.write(fd, payload.tobytes())
        if written != n:
            raise OSError(f"{path}: short write {written} of {n} B")
        os.fsync(fd)
    finally:
        os.close(fd)
    return path


class ExpertStoreWriter:
    """Write one file per (layer, expert) from the fill machinery.

    `feed` is a `q4e.gguf_feed.GgufFeed` (or any object exposing `_index`
    with reader tensors). `pinned` maps decoder layer -> sorted expert ids.
    `group_size` is the serving-shape group size.
    """

    def __init__(self, feed, store_dir, pinned, group_size=DEFAULT_GROUP_SIZE,
                 with_scale_zp=False, expert_chunk=32, provenance=None):
        self.feed = feed
        self.store_dir = store_dir
        self.pinned = {int(k): [int(e) for e in v]
                       for k, v in pinned.items()}
        self.group_size = int(group_size)
        self.with_scale_zp = bool(with_scale_zp)
        self.expert_chunk = int(expert_chunk)
        # What produced this membership, so the store is reproducible as
        # claimed. Never operator-local paths in the tracked tool's own use;
        # the caller supplies it (seed, pinned source, layer-keys source).
        self.provenance = dict(provenance or {})
        self.manifest = []

    # -- GGUF gather: only the pinned experts dequantise ---------------------
    def _dequant_experts(self, gguf_name, expert_ids):
        """f32 `[E, out, inn]` for the pinned experts of one tensor.

        `gguf_feed.GgufFeed._index` holds the reader tensors; indexing
        `tensor.data` gathers the expert blocks (leading axis) BEFORE
        dequantising, so a 71-of-512 layer costs 71/512 of the full dequant.
        """
        t = self.feed._index[gguf_name]
        raw = np.ascontiguousarray(
            t.data[np.asarray(expert_ids, dtype=np.int64)])
        name = t.tensor_type.name
        if name in ("F32", "F16"):
            return np.ascontiguousarray(np.asarray(raw), dtype=np.float32)
        # BF16 arrives as raw bytes; gguf.quants converts it (gguf_feed._dequant)
        from gguf import quants as _q
        return _q.dequantize(raw, t.tensor_type).astype(np.float32)

    def _source(self, layer, kind):
        """Only the pinned experts of `layer`, dequantised f32 `[E, out, inn]`."""
        ids = self.pinned[layer]
        w = {"gate": "ffn_gate_exps.weight",
             "up": "ffn_up_exps.weight",
             "down": "ffn_down_exps.weight"}[kind]
        gguf_name = f"blk.{layer}.{w}"
        return self._dequant_experts(gguf_name, ids)

    def _role_arrays(self, layer, kind):
        """(packed weights, packed zp, f32 scales) for the pinned experts.

        `packed_w` reshapes per expert [E, out, groups, gs/2]. `packed_zp`
        stays flat: it is the global u4 bitstream over the flattened
        [E, out, groups] order, so one expert's bytes are contiguous but the
        per-expert shape is out*groups/2 BYTES, not [out, groups, 1].
        """
        e = len(self.pinned[layer])
        out, inn = ROLE_OUT[kind], ROLE_IN[kind]
        filler = ef.ExpertFiller(self._source, self.group_size,
                                 expert_chunk=self.expert_chunk)
        packed_w, packed_zp, scales = filler.body(layer, kind, e, out, inn)
        groups = inn // self.group_size
        pw = np.asarray(packed_w, dtype=np.uint8).reshape(
            e, out, groups, self.group_size // 2)
        pzp = np.asarray(packed_zp, dtype=np.uint8).ravel()
        return pw, pzp, np.asarray(scales)

    def _sidecar_for(self, kind, pzp, scales, i):
        """Device-order (scales f16 bits, zp packed bytes) for one expert."""
        oc = ROLE_OUT[kind]
        gc = ROLE_IN[kind] // self.group_size
        per = oc * gc // 2
        codes = ef.unpack_u4(pzp[i * per:(i + 1) * per], oc * gc).reshape(oc, gc)
        return transpose_scale_zp_to_device(scales[i].reshape(oc, gc), codes)

    def plan(self):
        return sum(len(v) for v in self.pinned.values())

    def write(self, progress=False):
        os.makedirs(self.store_dir, exist_ok=True)
        ordinal = 0
        for layer in sorted(self.pinned):
            ids = self.pinned[layer]
            roles = {}
            for kind in ROLE_ORDER:
                roles[kind] = self._role_arrays(layer, kind)
            for i, expert in enumerate(ids):
                rec = device_order_weight_record(
                    {k: roles[k][0][i] for k in ROLE_ORDER})
                if rec.size != EXPERT_SLICE_BYTES:
                    raise ValueError(
                        f"layer {layer} expert {expert}: record is {rec.size} B, "
                        f"expected {EXPERT_SLICE_BYTES}")
                name = expert_file_name(ordinal)
                path = os.path.join(self.store_dir, name)
                write_expert_file(path, rec)
                entry = {
                    "ordinal": ordinal,
                    "layer": layer,
                    "expert": expert,
                    "file": name,
                    "bytes": int(rec.size),
                    "sha256": hashlib.sha256(rec.tobytes()).hexdigest(),
                    "layout": LAYOUT,
                }
                ordinal += 1
                if self.with_scale_zp:
                    side = name.replace(".bin", ".scaleszp")
                    spath = os.path.join(self.store_dir, side)
                    blob = bytearray()
                    for kind in ROLE_ORDER:
                        s, z = self._sidecar_for(
                            kind, roles[kind][1], roles[kind][2], i)
                        blob += s.tobytes() + z.tobytes()
                    write_expert_file(spath,
                                      np.frombuffer(bytes(blob), dtype=np.uint8))
                    entry["sidecar"] = side
                self.manifest.append(entry)
            if progress:
                print(f"layer {layer}: {len(ids)} experts", flush=True)
        self._write_manifest()
        return self.manifest

    def _write_manifest(self):
        manifest = {
            "layout": LAYOUT,
            "slice_bytes": EXPERT_SLICE_BYTES,
            "group_size": self.group_size,
            "roles": list(ROLE_ORDER),
            "provenance": self.provenance,
            "experts": self.manifest,
        }
        with open(os.path.join(self.store_dir, "manifest.json"), "w",
                  encoding="utf-8") as f:
            json.dump(manifest, f, indent=1, sort_keys=True)


def expert_file_name(ordinal):
    """The established store convention: a flat `expert_NNNN.bin`.

    The ordinal is the file's position in the store's own plan order (sorted
    layers, then the pinned expert order); the manifest maps it back to
    `(layer, expert)`. Flat names are what the arcwell reference tools expect
    (`stub/tools/aw_fiemap.c` builds `expert_%04d.bin`).
    """
    return f"expert_{int(ordinal):04d}.bin"


# -- pinned-set inputs --------------------------------------------------------

def load_pinned(path, layer_keys_path=None):
    """A `{layer: [expert, ...]}` mapping from a pinned-set file.

    Accepts either the plain JSON map, or the patch-0046 "hot-set seed v2"
    text format (`<layer_key> <expert> ...`) together with a
    `layer_key -> decoder_layer` map (--layer-keys).
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return json.loads(text)
    if layer_keys_path is None:
        raise ValueError(
            "a seed file is keyed by layer_key; pass --layer-keys "
            "(a JSON map layer_index -> layer_key) to invert it")
    with open(layer_keys_path, "r", encoding="utf-8") as f:
        keys = json.load(f)
    by_key = {int(v): int(k) for k, v in keys.items()}
    pinned = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        lk = int(parts[0])
        layer = by_key[lk]
        experts = [int(x) for x in parts[1:]]
        if layer in pinned:
            raise ValueError(f"layer {layer} (key {lk}) appears twice in {path}")
        pinned[layer] = experts
    return pinned


def splitmix64(x):
    """Patch 0018's splitmix64, transcribed (`code`)."""
    mask = (1 << 64) - 1
    x = (x + 0x9E3779B97F4A7C15) & mask
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return (z ^ (z >> 31)) & mask


def static_partition_resident_experts(seed, layer_key, num_expert, capacity):
    """Patch 0018's static-partition set, transcribed (`code`).

    The `min(capacity, num_expert)` experts with the smallest
    `splitmix64(seed ^ layer_key*0xD6E8FEB86659FD93)` rank, sorted by id.
    """
    mask = (1 << 64) - 1
    key = (layer_key * 0xD6E8FEB86659FD93) & mask
    ranked = sorted(
        (splitmix64(splitmix64((seed ^ key) & mask) ^ e), e)
        for e in range(num_expert))
    return sorted(e for _, e in ranked[:min(capacity, num_expert)])


# -- CLI ----------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", action="append", required=True,
                    help="GGUF shard path/glob (repeatable)")
    ap.add_argument("--store", required=True, help="ext4 store directory")
    ap.add_argument("--pinned", help="pinned-set JSON or hot-set seed v2")
    ap.add_argument("--layer-keys", help="JSON map layer_index -> layer_key")
    ap.add_argument("--all", action="store_true",
                    help="every expert of every layer")
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--seed", type=lambda s: int(s, 0),
                    default=0xF2A17C0DE5EED,
                    help="patch 0018 seed (used when --all and --layer-keys)")
    ap.add_argument("--capacity", type=int, default=71,
                    help="slots/layer for the splitmix64 set")
    ap.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    ap.add_argument("--with-scale-zp", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.pinned:
        pinned = load_pinned(args.pinned, args.layer_keys)
    elif args.layer_keys:
        with open(args.layer_keys, "r", encoding="utf-8") as f:
            keys = json.load(f)
        pinned = {int(k): static_partition_resident_experts(
            args.seed, int(v), args.experts, args.capacity)
            for k, v in keys.items()}
    elif args.all:
        pinned = {l: list(range(args.experts)) for l in range(args.layers)}
    else:
        ap.error("one of --pinned / --layer-keys / --all is required")

    n = sum(len(v) for v in pinned.values())
    print(f"pinned set: {len(pinned)} layers, {n} experts, "
          f"{n * EXPERT_SLICE_BYTES} B at {EXPERT_SLICE_BYTES} B each")
    if args.dry_run:
        for layer in sorted(pinned):
            print(f"  layer {layer:02d}: {len(pinned[layer])} experts")
        return 0

    from . import gguf_feed
    provenance = {
        "source": "pinned" if args.pinned else
                  ("splitmix64" if args.layer_keys else "all"),
        "seed": args.seed,
        "capacity": args.capacity,
        "num_expert": args.experts,
        "pinned_file": args.pinned,
        "layer_keys_file": args.layer_keys,
    }
    # GgufFeed treats a list as explicit shard paths and a string as a
    # directory/glob to expand, so a single --gguf keeps its glob semantics.
    shards = args.gguf[0] if len(args.gguf) == 1 else args.gguf
    feed = gguf_feed.GgufFeed(shards)
    writer = ExpertStoreWriter(feed, args.store, pinned,
                               group_size=args.group_size,
                               with_scale_zp=args.with_scale_zp,
                               provenance=provenance)
    writer.write(progress=True)
    print(f"wrote {len(writer.manifest)} files to {args.store}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
