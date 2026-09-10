#!/usr/bin/env python3
"""Range-sliced fetch of individual tensors from a sharded safetensors checkpoint.

WP8b (0.5.0). A sharded HF checkpoint publishes `model.safetensors.index.json`
(a `weight_map` of tensor-name -> shard file) and each shard is a safetensors
file whose header (a JSON dict of name -> {dtype, shape, data_offsets:[b,e]})
sits at a known place at the front. safetensors data is therefore
RANGE-ADDRESSABLE: to pull one tensor you fetch that shard's small header via an
HTTP Range request, read the tensor's byte offsets from it, and fetch only those
bytes -- never the multi-GB shard. This tool does exactly that for a name
filter, so a 4 GB MTP head is acquired from a 360 GB checkpoint without
downloading the checkpoint.

Idempotent and resumable: a completed tensor (raw file present at the expected
size with a matching sha256 in the sidecar log) is skipped. Every fetch appends
to an acquisition log (URL, byte range, bytes fetched, sha256, timestamp).

safetensors layout (little-endian): [u64 header_len][header JSON of header_len
bytes][data]. A tensor's data_offsets [b, e) are relative to the start of the
data section, so its absolute byte range in the file is
[8 + header_len + b, 8 + header_len + e).

stdlib only (urllib). No torch, no safetensors package, no HF hub.

Usage:
  # size first (headers only, no payload) -- ALWAYS do this before a big fetch:
  fetch_safetensors_tensors.py --repo Qwen/Qwen3.8-Flash-Next --prefix mtp. --manifest
  # then fetch into a staging dir, assembling a single mtp.safetensors:
  fetch_safetensors_tensors.py --repo Qwen/Qwen3.8-Flash-Next --prefix mtp. \
      --out /flash/staging/flash-next-mtp --assemble mtp_head.safetensors
"""
import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time
import urllib.request

HF = "https://huggingface.co"


def resolve_url(repo, revision, shard):
    return f"{HF}/{repo}/resolve/{revision}/{shard}"


def http_get(url, start=None, end=None, retries=4, timeout=60):
    """GET url, optionally Range [start, end] inclusive. Returns bytes. Retries
    on transient errors with backoff. Range requests must come back 206."""
    headers = {"User-Agent": "arcint-fetch-safetensors/1.0"}
    ranged = start is not None
    if ranged:
        headers["Range"] = f"bytes={start}-{end}"
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                code = r.getcode()
                if ranged and code != 206:
                    # server ignored Range and would stream the whole file -- refuse
                    raise RuntimeError(f"Range request returned {code}, not 206 "
                                       f"(server may not support ranges): {url}")
                return r.read()
        except Exception as e:  # noqa: BLE001 -- transient network, retry
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url}\n  last: {last}")


def read_index(repo, revision):
    url = f"{HF}/{repo}/resolve/{revision}/model.safetensors.index.json"
    data = http_get(url)
    idx = json.loads(data)
    return idx.get("weight_map", {}), idx.get("metadata", {})


def read_header(url):
    """Fetch and parse a safetensors file header. Returns (header_dict,
    data_section_start_abs_offset)."""
    n_bytes = http_get(url, 0, 7)
    (header_len,) = struct.unpack("<Q", n_bytes)
    header_json = http_get(url, 8, 8 + header_len - 1)
    header = json.loads(header_json)
    return header, 8 + header_len


ST_DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
                  "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
                  "U32": 4, "I32": 4, "F32": 4,
                  "U64": 8, "I64": 8, "F64": 8}


def tensor_nbytes(shape, dtype):
    n = 1
    for d in shape:
        n *= d
    return n * ST_DTYPE_BYTES.get(dtype, 0)


def gather(repo, revision, name_match, out_dir, assemble, manifest):
    weight_map, meta = read_index(repo, revision)
    names = sorted(n for n in weight_map if name_match(n))
    if not names:
        print("no tensors matched the filter", file=sys.stderr)
        return 2
    # group by shard so each shard header is fetched once
    by_shard = {}
    for n in names:
        by_shard.setdefault(weight_map[n], []).append(n)

    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "acquisition_log.jsonl")
    entries = []  # (name, dtype, shape, nbytes, abs_begin, abs_end, shard, url)
    total = 0
    for shard in sorted(by_shard):
        url = resolve_url(repo, revision, shard)
        header, data_start = read_header(url)
        for n in by_shard[shard]:
            h = header[n]
            b, e = h["data_offsets"]
            nbytes = e - b
            calc = tensor_nbytes(h["shape"], h["dtype"])
            if calc and calc != nbytes:
                print(f"WARN {n}: offset span {nbytes} != shape*dtype {calc}", file=sys.stderr)
            entries.append((n, h["dtype"], h["shape"], nbytes,
                            data_start + b, data_start + e, shard, url))
            total += nbytes

    print(f"repo {repo}@{revision}: {len(entries)} tensor(s) matched, "
          f"total payload {total} B ({total / (1<<20):.1f} MiB, {total / (1<<30):.3f} GiB)")
    for n, dt, shp, nb, _ab, _ae, shard, _u in entries:
        print(f"  {n:<58} {dt:>5} {str(shp):<22} {nb:>13} B  [{shard}]")
    if manifest:
        return 0

    # fetch each tensor's byte range; resumable via sidecar sha log
    done = {}
    if os.path.exists(log_path):
        for ln in open(log_path):
            try:
                r = json.loads(ln)
                done[r["name"]] = r
            except Exception:  # noqa: BLE001
                pass
    logf = open(log_path, "a")
    raw = {}  # name -> bytes (for assembly)
    for n, dt, shp, nb, ab, ae, shard, url in entries:
        safe = n.replace("/", "__")
        raw_path = os.path.join(out_dir, safe + ".raw")
        prev = done.get(n)
        if (prev and os.path.exists(raw_path) and os.path.getsize(raw_path) == nb
                and prev.get("sha256") and prev.get("bytes") == nb):
            with open(raw_path, "rb") as f:
                data = f.read()
            if hashlib.sha256(data).hexdigest() == prev["sha256"]:
                print(f"  skip (done) {n}")
                raw[n] = data
                continue
        data = http_get(url, ab, ae - 1)
        if len(data) != nb:
            raise RuntimeError(f"{n}: fetched {len(data)} B, expected {nb}")
        with open(raw_path, "wb") as f:
            f.write(data)
        sha = hashlib.sha256(data).hexdigest()
        rec = {"name": n, "dtype": dt, "shape": shp, "bytes": nb, "sha256": sha,
               "url": url, "range": [ab, ae - 1], "shard": shard,
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        raw[n] = data
        print(f"  fetched {n}  {nb} B  sha {sha[:16]}...")
    logf.close()

    if assemble:
        # build a single safetensors file with contiguous offsets
        asm_header = {}
        offset = 0
        blobs = []
        for n, dt, shp, nb, *_ in entries:
            asm_header[n] = {"dtype": dt, "shape": shp, "data_offsets": [offset, offset + nb]}
            blobs.append(raw[n]); offset += nb
        hjson = json.dumps(asm_header, separators=(",", ":")).encode("utf-8")
        # safetensors requires 8-byte alignment of the header length
        pad = (-len(hjson)) % 8
        hjson = hjson + b" " * pad
        asm_path = os.path.join(out_dir, assemble)
        h = hashlib.sha256()
        with open(asm_path, "wb") as f:
            prefix = struct.pack("<Q", len(hjson))
            f.write(prefix); h.update(prefix)
            f.write(hjson); h.update(hjson)
            for blob in blobs:
                f.write(blob); h.update(blob)
        print(f"assembled {asm_path}: {len(entries)} tensors, "
              f"{os.path.getsize(asm_path)} B, sha256 {h.hexdigest()}")
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. Qwen/Qwen3.8-Flash-Next")
    ap.add_argument("--revision", default="main")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prefix", help="tensor-name prefix filter (e.g. 'mtp.')")
    g.add_argument("--regex", help="tensor-name regex filter")
    ap.add_argument("--out", default=".", help="output/staging directory")
    ap.add_argument("--assemble", help="assemble matched tensors into this safetensors filename")
    ap.add_argument("--manifest", action="store_true",
                    help="headers only: print sizes and exit, no payload fetch")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.regex:
        rx = re.compile(a.regex)
        match = lambda n: rx.search(n) is not None  # noqa: E731
    else:
        match = lambda n: n.startswith(a.prefix)  # noqa: E731
    return gather(a.repo, a.revision, match, a.out, a.assemble, a.manifest)


if __name__ == "__main__":
    sys.exit(main())
