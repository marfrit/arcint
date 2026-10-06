"""Compare our tensors with llama.cpp's, element for element or by corners.

Two reference formats:

  * a `contrib/llama-eval-dump` directory (index.txt + <name>#<k>.f32): whole
    tensors, the comparison of record;
  * a `llama-eval-callback` log: sums and three-element corners, enough to
    tell agreement from noise but not to localise a head (four printed
    decimals on values near 1e-4).

Our side is a .npy: a boot-driver cut dump (`forward_1.npy`, rows [T, X]) or
a `ref_forward_real.py` tap. llama's data is ne0-fastest, so its numpy
shape is (ne3, ne2, ne1, ne0) squeezed; ours is reshaped to that (a [T,
hc*H] residual becomes [T, hc, H], the same memory order as llama's
{H, hc, T}).

  llama_tap_compare.py dump DUMP_DIR TAG NPY [--per-head] [--reshape]
  llama_tap_compare.py log  LOG TAG NPY        (TAG = name or name#k)
"""
import re
import sys
from pathlib import Path

import numpy as np

HEAD = re.compile(r"^common_debug_cb_eval:\s*(.+?) = \((\w+)\)\s+(\w+)\(.*\) = \{(\d+), (\d+), (\d+), (\d+)\}")


# --- the callback log: corners --------------------------------------------
def parse_log(log):
    out = {}
    lines = Path(log).read_text().splitlines()
    i = 0
    while i < len(lines):
        m = HEAD.match(lines[i])
        if not m:
            i += 1
            continue
        name = m.group(1).strip()
        ne = tuple(int(x) for x in m.groups()[3:7])
        j, body, s = i + 1, [], None
        while j < len(lines) and not HEAD.match(lines[j]):
            t = lines[j].strip()
            if t.startswith("sum = "):
                s = float(t[6:])
                break
            body.append(t)
            j += 1
        rows = [[float(x) for x in re.findall(r"-?\d+\.\d+|-?inf|nan", t)]
                for t in body if t.startswith("[") and "," in t]
        out.setdefault(name, []).append(dict(ne=ne, sum=s, rows=[r for r in rows if r]))
        i = j + 1
    return out


def entry(parsed, spec):
    name, _, k = spec.partition("#")
    return parsed[name][int(k) if k else 0]


def corners(a):
    n0 = a.shape[-1]
    flat = a.reshape(-1, n0)
    return np.concatenate([flat[:, :3], flat[:, -3:]], axis=1) if n0 > 6 else flat


def printed_rows(a3):
    n2, n1, _ = a3.shape
    i2 = range(n2) if n2 <= 6 else [0, 1, 2, n2 - 3, n2 - 2, n2 - 1]
    i1 = range(n1) if n1 <= 6 else [0, 1, 2, n1 - 3, n1 - 2, n1 - 1]
    return np.array([a3[a, b] for a in i2 for b in i1])


def compare_log(log, tag, npy):
    ref = entry(parse_log(log), tag)
    a = np.load(npy).astype(np.float64)
    ne = ref["ne"]
    a3 = a.reshape(ne[2], ne[1], ne[0])
    got = corners(printed_rows(a3))
    want = np.array(ref["rows"], dtype=np.float64)
    n = min(len(got), len(want))
    got, want = got[:n], want[:n]
    d = np.abs(got - want)
    print(f"{tag}: llama ne={ne} sum={ref['sum']:.6f} | ours sum={a3.sum():.6f} | corners {n} rows: "
          f"max|diff| {d.max():.4f} mean|diff| {d.mean():.4f} vs mean|llama| {np.abs(want).mean():.4f}; "
          f"corr {np.corrcoef(got.ravel(), want.ravel())[0, 1]:.4f}")


# --- the dump: whole tensors ------------------------------------------------
def load_index(d):
    idx = {}
    for line in (Path(d) / "index.txt").read_text().splitlines():
        p = line.split()
        if len(p) >= 6 and p[1] != "SKIPPED":
            idx[p[0]] = tuple(int(x) for x in p[1:5])
    return idx


def compare_dump(dump, tag, npy, per_head=False, reshape=False):
    ne = load_index(dump)[tag]
    L = np.fromfile(Path(dump) / f"{tag}.f32", dtype=np.float32)
    L = L.reshape(ne[3], ne[2], ne[1], ne[0]).astype(np.float64)
    while L.ndim > 2 and L.shape[0] == 1:          # drop leading unit axes only (T = 1 survives)
        L = L[0]
    P = np.load(npy).astype(np.float64)
    if P.ndim == 3 and P.shape[0] == 1 and P.shape[1:] == L.shape[::-1]:
        P = P[0].T                                   # a [1, C, T] activation against llama's {C, T}
    elif P.ndim == 2 and P.shape != L.shape and P.shape == L.shape[::-1]:
        P = P.T
    elif P.shape != L.shape and P.size == L.size and reshape:
        P = P.reshape(L.shape)                       # explicit: the caller vouched for the memory order
    if P.shape != L.shape:
        print(f"{tag:<26} shape mismatch: llama {L.shape} ours {P.shape} (pass --reshape to "
              f"reinterpret an equal-size array in llama's memory order)")
        return
    d = np.abs(L - P)
    print(f"{tag:<26} shape {str(L.shape):<16} sum llama {L.sum():+.4f} ours {P.sum():+.4f} | "
          f"max|diff| {d.max():.5f} mean|diff| {d.mean():.6f} mean|llama| {np.abs(L).mean():.6f} "
          f"corr {np.corrcoef(L.ravel(), P.ravel())[0, 1]:.5f}")
    if per_head and L.ndim == 3:
        T, H, _ = L.shape
        rel = np.array([[np.abs(L[t, h] - P[t, h]).max() / max(np.abs(L[t, h]).max(), 1e-9)
                         for h in range(H)] for t in range(T)]).max(0)
        print(f"   heads within 5%: {int((rel < 0.05).sum())}/{H}; worst: " +
              ", ".join(f"h{h}:{rel[h]:.3f}" for h in np.argsort(-rel)[:6]))


def main(argv):
    if len(argv) < 4:
        print(__doc__)
        return 2
    mode, ref, tag, npy = argv[:4]
    if mode == "log":
        compare_log(ref, tag, npy)
    elif mode == "dump":
        compare_dump(ref, tag, npy, per_head="--per-head" in argv, reshape="--reshape" in argv)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
