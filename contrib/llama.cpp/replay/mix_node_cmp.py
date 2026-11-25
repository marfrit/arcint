import sys, numpy as np
def load(prefix):
    rows = []
    for l in open(prefix + ".tsv"):
        f = l.rstrip("\n").split("\t")
        rows.append(dict(idx=int(f[0]), kind=f[1], op=f[2], name=f[3], type=f[4],
                         ne=[int(x) for x in f[5].split(",")], nb=f[6], voffs=f[7], vsrc=f[8], srcs=f[9],
                         off=int(f[10]), bytes=int(f[11])))
    return rows, np.memmap(prefix + ".bin", dtype=np.float32, mode="r")
def data(r, mm):
    return np.asarray(mm[r["off"]//4:(r["off"] + r["bytes"])//4], dtype=np.float64)
def nmse(a, b):
    d = float(((a - b)**2).sum()); n = float((b**2).sum())
    return d/n if n > 0 else (0.0 if d == 0 else float("inf"))

A, am = load(sys.argv[1]); B, bm = load(sys.argv[2])
thr = 1e-10
def named(rows):
    out = {}; seq = []
    for r in rows:
        if r["kind"] != "node" or r["bytes"] == 0: continue
        n = r["name"]
        if n == "" or n.startswith("node_") or n.startswith("leaf_") or " (" in n: continue
        k = out.get(n, 0); out[n] = k + 1
        seq.append(((n, k), r))
    return seq
sa = dict(named(A)); sb = named(B)
print("named nodes with data: %s %d, %s %d" % (sys.argv[1], len(sa), sys.argv[2], len(sb)))
first = None; n_cmp = 0; n_bad = 0; trail = []
for key, rb in sb:
    ra = sa.get(key)
    if ra is None or ra["bytes"] != rb["bytes"]: continue
    e = nmse(data(rb, bm), data(ra, am)); n_cmp += 1
    trail.append((rb["idx"], key[0], rb["op"], rb["ne"], e))
    if e > thr:
        n_bad += 1
        if first is None: first = len(trail) - 1
print("compared %d named nodes, %d above %g" % (n_cmp, n_bad, thr))
if first is not None:
    for t in trail[max(0, first - 6):first + 8]:
        print("  #%d %-22s %-10s %s nmse %.3g" % t)
    print("last nodes:")
    for t in trail[-3:]:
        print("  #%d %-22s %-10s %s nmse %.3g" % t)

# the mixes: the arm's nodes in B against the chain's states views in A, in order
mix = [];
i = 0
while i < len(B):
    r = B[i]
    if r["kind"] == "node" and r["op"] == "DEPTH_MIX":
        srcs = []
        j = i + 1
        while j < len(B) and B[j]["kind"].startswith("src"):
            srcs.append(B[j]); j += 1
        mix.append((r, srcs))
    i += 1
states = []
for i, r in enumerate(A):
    if r["kind"] == "node" and r["op"] == "RMS_NORM" and i + 1 < len(A) and A[i + 1]["kind"] == "src0" and A[i + 1]["ne"][1] >= 2:
        states.append((r, A[i + 1]))
print("arm mixes in %s: %d; chain states views (>= 2 rows) in %s: %d" % (sys.argv[2], len(mix), sys.argv[1], len(states)))
shown = 0
for n, (r, srcs) in enumerate(mix):
    ne0, T, nq = r["ne"][0], r["ne"][1], r["ne"][2]
    k = srcs[0]["ne"][1]
    blk = data(srcs[0], bm).reshape(T, k, ne0); cur = data(srcs[1], bm).reshape(T, ne0)
    q = data(srcs[2], bm).reshape(nq, ne0)
    nw = data(srcs[3], bm).reshape(-1, ne0) if len(srcs) > 3 else None
    out = data(r, bm).reshape(nq, T, ne0)
    # the chain's arithmetic in float64 on the node's own inputs
    v = np.concatenate([blk, cur[:, None, :]], axis=1)                       # T, S, D
    vn = v/np.sqrt((v**2).mean(axis=2, keepdims=True) + 1e-6)
    sc = np.einsum("tsd,qd->tqs", vn, q)/np.sqrt(ne0)
    sc = sc - sc.max(axis=2, keepdims=True); p = np.exp(sc); p /= p.sum(axis=2, keepdims=True)
    ref = np.einsum("tqs,tsd->qtd", p, vn)
    e_self = nmse(out, ref) if nw is None else float("nan")
    line = "mix %3d #%d %-10s k %d nq %d norm %d: op vs chain-math on its own inputs %.3g" % (n, r["idx"], r["name"], k, nq, nw is not None, e_self)
    if n < len(states):
        sr, ss = states[n]
        S = ss["ne"][1]
        if S == k + 1 and ss["ne"][2] == T:
            st = data(ss, am).reshape(T, S, ne0)
            e_blk = nmse(blk, st[:, :k, :]); e_cur = nmse(cur, st[:, k, :])
            rows = [nmse(blk[:, s, :], st[:, s, :]) for s in range(k)]
            line += "; inputs vs the chain run: block rows %.3g (a row: %s), cur %.3g" % (e_blk, " ".join("%.2g" % x for x in rows), e_cur)
            bad = e_blk > thr or e_cur > thr
        else:
            line += "; chain pair has %d rows (no match)" % S; bad = True
    else:
        bad = True
    if bad or (e_self == e_self and e_self > thr) or n < 3:
        if shown < 14:
            print(line); shown += 1
