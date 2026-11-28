#!/usr/bin/env python3
"""0227: flash_attn_dec27.cl a lane at a time, on raw q8_0 / q4_0 cache bytes, against a direct
double-precision attention on the same blocks. No device.

What it checks is the kernel's index arithmetic, which a compile cannot: the nine dword reads of K
and which dword is which block's scale and codes (the DEC27_PAIR table), Q's lane order in the
scratch (dec27_qdim), the V byte reads on the 4-byte grid with the even blocks' splice, the lane to
dim map of the store (V4_ODIM), the flat (head, token) rows of a tile, the mask slots, the split
ranges and the merge. The reads are modelled as the FA port's are documented (flash_attn_port.cl):
the transposed dword read has the key in the lane, the transformed byte read the byte column in the
lane and 32 key rows as its 32 bytes; the DPAS is a row x 16-deep dot product in f32.

The mask is the one read the model takes by ADDRESS, since 2026-11-27. Until then it took the mask
by index (mask[row, key]: what the kernel means), and so could not see what the first build did on
the 24 GB card: its 16-half block read of a run whose pointer is 2 bytes past a 4-byte boundary
(an odd row over an odd key count: the cells are 2 bytes, a row's run starts at row x n_kv cells)
gives the odd lanes the cell two before their own. block_read_us() below is that read as measured
(with it, a double reference stands 4.4e-7 from the card's dumped answers in every row; as written
5.4e-4 .. 3.3e-3). The kernel reads a run off 4 bytes a key at a time (mal[] in the text);
mask_read="block" is the first build, which must fail here at an odd key count as it did there.
A new block read in the kernel needs its pointer's alignment shown here the same way.

It also states the test cases' bound before any card run: the kernel model's NMSE against the
double reference, the CPU backend's arithmetic (Q quantized to q8_0 a 32-dim block, K's codes, V in
f32) against the same reference, the two against each other (what test-backend-ops measures), and
what each red build does to the answer.

    python3 dec27_kernel_model.py            # rc 0: the kernel model within the bound, every red above it
"""
import sys
import numpy as np

D, QR, NB8 = 256, 8, 8
HBK, HBV = 272, 144
M_INIT = np.float32(-3.0e38)
LOG2E = np.float32(1.4426950408889634)
BOUND = 1e-5


def det(shape, seed):
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=shape).astype(np.float32)


# ---- ggml's quantizers (ggml-quants.c quantize_row_q8_0_ref / q4_0_ref) ----
def quant_q8_0(x):                       # x [..., 256] -> bytes [..., 272]
    xb = x.reshape(x.shape[:-1] + (NB8, 32))
    amax = np.abs(xb).max(-1)
    d = (amax / np.float32(127.0)).astype(np.float32)
    idv = np.where(d != 0, np.float32(1.0) / np.where(d != 0, d, 1), 0).astype(np.float32)
    q = np.round(xb * idv[..., None]).astype(np.int8)        # roundf: half away; ties do not occur in the fills
    out = np.zeros(x.shape[:-1] + (NB8, 34), np.uint8)
    out[..., 0:2] = d.astype(np.float16)[..., None].view(np.uint8)
    out[..., 2:] = q.view(np.uint8)
    return out.reshape(x.shape[:-1] + (HBK,))


def quant_q4_0(x):                       # x [..., 256] -> bytes [..., 144]
    xb = x.reshape(x.shape[:-1] + (NB8, 32))
    idx = np.abs(xb).argmax(-1)
    mx = np.take_along_axis(xb, idx[..., None], -1)[..., 0]
    d = (mx / np.float32(-8.0)).astype(np.float32)
    idv = np.where(d != 0, np.float32(1.0) / np.where(d != 0, d, 1), 0).astype(np.float32)
    xi = np.minimum(15, (xb * idv[..., None] + np.float32(8.5)).astype(np.int8)).astype(np.uint8)
    out = np.zeros(x.shape[:-1] + (NB8, 18), np.uint8)
    out[..., 0:2] = d.astype(np.float16)[..., None].view(np.uint8)
    out[..., 2:] = xi[..., :16] | (xi[..., 16:] << 4)
    return out.reshape(x.shape[:-1] + (HBV,))


def dequant_q8_0(b):                     # bytes [..., 272] -> f64 [..., 256]
    bb = b.reshape(b.shape[:-1] + (NB8, 34))
    d = bb[..., 0:2].copy().view(np.float16)[..., 0].astype(np.float64)
    return (bb[..., 2:].view(np.int8).astype(np.float64) * d[..., None]).reshape(b.shape[:-1] + (D,))


def dequant_q4_0(b):
    bb = b.reshape(b.shape[:-1] + (NB8, 18))
    d = bb[..., 0:2].copy().view(np.float16)[..., 0].astype(np.float64)
    qs = bb[..., 2:]
    n = np.concatenate([qs & 15, qs >> 4], -1).astype(np.float64) - 8.0
    return (n * d[..., None]).reshape(b.shape[:-1] + (D,))


# ---- the case: the dense 27B hybrid's layout, the tail keys carrying weight ----
class Case:
    def __init__(self, n_kv, T, n_head_kv=4, G=6, seed=1, masked_cells=(), empty=0, vfloor=False):
        self.n_kv, self.T, self.hkv, self.G = n_kv, T, n_head_kv, G
        nh = n_head_kv * G
        u = np.sign(det((n_head_kv, D), seed + 7)).astype(np.float32)
        q = det((nh, T, D), seed)
        k = det((n_head_kv, n_kv, D), seed + 1)
        v = det((n_head_kv, n_kv, D), seed + 2)
        amp_k = np.float32(0.25) * np.power(np.float32(2.0), 3 * (det((n_head_kv, n_kv, NB8), seed + 3) + 1) / 2)
        amp_v = np.float32(0.25) * np.power(np.float32(2.0), 3 * (det((n_head_kv, n_kv, NB8), seed + 4) + 1) / 2)
        k = (k.reshape(n_head_kv, n_kv, NB8, 32) * amp_k[..., None]).reshape(n_head_kv, n_kv, D)
        v = (v.reshape(n_head_kv, n_kv, NB8, 32) * amp_v[..., None]).reshape(n_head_kv, n_kv, D)
        used = n_kv - empty                           # the cells that hold a token (the cache pads its cells to 256)
        if vfloor:                                    # no dominant key over V blocks of ~1e-6: the V-floor case
            v = v * np.float32(1e-6)
        # the tail: the last 8 keys point along the kv head's direction, which every query row carries
        q = q + np.float32(0.4) * np.repeat(u, G, axis=0)[:, None, :]
        nt = 0 if vfloor else min(8, used)
        # with a logit of about ln(n_kv / 8) - 4 above the other keys', so that at any depth the tail and the rest
        # both hold a good part of a row's weight: a lost tail key and a lost block elsewhere both move the answer
        beta = np.float32(max(0.2, (np.log(max(n_kv, 64) / 8.0) - 4.0) / 6.4))
        if nt:
            k[:, used - nt:used, :] = beta * u[:, None, :] + np.float32(0.3) * det((n_head_kv, nt, D), seed + 5)
        self.q = q.astype(np.float32)
        # the cache as the model's graph has it: a key row holds the kv heads one after another
        self.kpitch, self.vpitch = n_head_kv * HBK, n_head_kv * HBV
        self.kbytes = quant_q8_0(k).transpose(1, 0, 2).reshape(n_kv, self.kpitch).copy()
        self.vbytes = quant_q4_0(v).transpose(1, 0, 2).reshape(n_kv, self.vpitch).copy()
        # the mask as served: row t sees the keys up to its own token (the T rows are the cache's last T cells)
        m = np.zeros((T, n_kv), np.float16)
        for t in range(T):
            m[t, used - T + t + 1:] = -np.inf
        for lo, hi in masked_cells:
            m[:, lo:hi] = -np.inf
        self.mask = m
        self.scale = np.float32(1.0 / np.sqrt(D))

    def kd(self):
        return dequant_q8_0(self.kbytes.reshape(self.n_kv, self.hkv, HBK))

    def vd(self):
        return dequant_q4_0(self.vbytes.reshape(self.n_kv, self.hkv, HBV))


def reference(c, drop_last=False, lost=()):
    """double precision, on the dequantized blocks; out [n_head, T, D]"""
    kd, vd = c.kd(), c.vd()
    out = np.zeros((c.hkv * c.G, c.T, D))
    vis = ~np.isinf(c.mask.astype(np.float64))
    if drop_last:
        vis = vis.copy(); vis[:, c.n_kv - 1] = False
    for lo, hi in lost:
        vis = vis.copy(); vis[:, lo:hi] = False
    for h in range(c.hkv * c.G):
        kvh = h // c.G
        s = (c.q[h].astype(np.float64) @ kd[:, kvh, :].T) * float(c.scale)
        s = np.where(vis, s, -np.inf)
        s -= s.max(1, keepdims=True)
        p = np.exp(s)
        out[h] = (p @ vd[:, kvh, :]) / p.sum(1, keepdims=True)
    return out


def cpu_backend(c):
    """ggml's CPU FLASH_ATTN_EXT on this cache: Q to q8_0 a row, q8_0 x q8_0 dots, V to f32"""
    kb = c.kbytes.reshape(c.n_kv, c.hkv, NB8, 34)
    kdv = kb[..., 0:2].copy().view(np.float16)[..., 0].astype(np.float32)
    kc = kb[..., 2:].view(np.int8).astype(np.int32)
    vd = c.vd().astype(np.float32)
    out = np.zeros((c.hkv * c.G, c.T, D), np.float32)
    for h in range(c.hkv * c.G):
        kvh = h // c.G
        qq = quant_q8_0(c.q[h]).reshape(c.T, NB8, 34)
        qdv = qq[..., 0:2].copy().view(np.float16)[..., 0].astype(np.float32)
        qc = qq[..., 2:].view(np.int8).astype(np.int32)
        isum = np.einsum('tbi,kbi->tkb', qc, kc[:, kvh])
        s = (isum.astype(np.float32) * qdv[:, None, :] * kdv[None, :, kvh, :]).sum(-1, dtype=np.float32) * c.scale
        s = np.where(np.isinf(c.mask.astype(np.float32)), -np.inf, s + c.mask.astype(np.float32))
        s = s - s.max(1, keepdims=True)
        p = np.exp(s, dtype=np.float32)
        out[h] = (p @ vd[:, kvh, :]) / p.sum(1, keepdims=True)
    return out


# ---- the kernel, a sub-group at a time ----
def ORD(h):
    return 4 * ((h & 7) >> 1) + 2 * (h & 1) + (h >> 3)


def qdim(c, h):
    ko = ORD(h)
    e1 = 18 + ko if ko < 12 else (ko - 12 if ko < 14 else ko + 16)
    return 32 * (c // 2) + ((16 * (c & 1) + ko) if ((c // 2) & 1) else (e1 if (c & 1) else 2 + ko))


def V4J(b, l):
    return l if (b & 1) else (l + 2 if l < 14 else l - 14)


def b16(w, xor=0x64806480, mask=0x00FF00FF, shifts=(0, 8)):
    """4 dwords a lane [16, 4] -> 16 halves a lane [16, 16] (as f32 values)"""
    a = ((w >> np.uint32(shifts[0])) & np.uint32(mask))
    b = ((w >> np.uint32(shifts[1])) & np.uint32(mask))
    if xor == 0x64806480:
        a, b = a ^ np.uint32(xor), b ^ np.uint32(xor)
    else:
        a, b = a | np.uint32(xor), b | np.uint32(xor)
    r = np.concatenate([a, b], 1).astype('<u4')
    return r.view('<f2').astype(np.float32)          # [16 lanes, 16 halves]


def dpas(a, b, acc):
    """a [16 lanes = depth, 8 rows] f16 values, b [16 lanes, 16 halves], acc [16 lanes, 8 rows] f32"""
    return (acc + (b.astype(np.float32) @ a.astype(np.float32))).astype(np.float32)


class Reds:
    v_scale = tail = seam = nomax = k_scale2 = pvnorm_ic = False


def block_read_us(flat, idx, byte0=0):
    """intel_sub_group_block_read_us of 16 halves at cell idx of a buffer whose cell 0 is at byte byte0 (mod 4), as
    the 24 GB card does it: on a 4-byte boundary the 16 cells; 2 bytes past one, the even lanes their own cell and
    the odd lanes the cell two before (measured 2026-11-27; a contract violation, not a documented result)."""
    lanes = np.arange(16)
    if (byte0 + 2 * idx) % 4 == 0:
        return flat[idx + lanes]
    return flat[np.maximum(idx + lanes - 2 * (lanes & 1), 0)]


def kernel(c, n_split, red=Reds(), split_blocks=None, pvnorm=True, pv32=False, mask_read="guarded"):
    # pvnorm=True is the kernel: the scale in P normalised a (32-key block, q4_0 block). The two forms that were built,
    # measured and left out of the series stay here as comparisons: pvnorm=False is f16(p 8 d) without the
    # normalisation (it fails the V-floor fill), pv32=True is P V in f32 without the DPAS (right, slower on the card)
    if pv32:
        pvnorm = False
    n_kv, T, G, hkv = c.n_kv, c.T, c.G, c.hkv
    n_tiles = (G * T + 7) // 8
    n_rows = G * T
    blocks = (n_kv + 31) // 32
    if split_blocks is None:
        split_blocks = (blocks + n_split - 1) // n_split
    scale_log2 = np.float32(LOG2E * c.scale)
    kdw = c.kbytes                                    # [n_kv, pitch] bytes
    mask_u = c.mask.view(np.uint16)
    mask_flat = np.ascontiguousarray(mask_u).reshape(-1)          # [rows][n_kv] cells of 2 bytes, cell 0 on 4 bytes
    assert mask_read in ("guarded", "block")
    lanes = np.arange(16)
    vord = np.where(lanes < 8, 2 * lanes, 2 * lanes - 15)
    assert all(vord[h] == ORD(h) for h in range(16))
    out = np.zeros((hkv * G, T, D), np.float32)
    written = np.zeros((hkv * G, T, D), bool)

    def krows(y):                                     # the 16 key rows of a group, clamped as the block read clamps
        return np.minimum(y + lanes, n_kv - 1)

    for kvh in range(hkv):
        kx = (kvh * HBK) % 64                         # the head's first byte from the 64-rounded base (a 64-aligned row)
        kbase = kvh * HBK - kx
        vx = (kvh * HBV) % 64
        vbase = kvh * HBV - vx
        assert kx % 4 == 0 and vx % 4 == 0

        def K8W(w, y):                                # [16 lanes = keys, 8 dwords]
            x0 = kbase + 4 * (kx // 4 + w)
            assert x0 + 32 <= kbase + kx + HBK, "a K read past the head"
            return kdw[krows(y)][:, x0:x0 + 32].copy().view('<u4')

        def V4R(xb, y):                               # [16 lanes = byte columns, 8 dwords = 32 key rows]
            x0 = vbase + vx + xb
            assert (vx + xb) % 4 == 0 and x0 + 16 <= vbase + vx + HBV, "a V read off the grid or past the head"
            rows = np.minimum(y + np.arange(32), n_kv - 1)
            return np.ascontiguousarray(c.vbytes[rows][:, x0:x0 + 16].T).view('<u4')

        for tile in range(n_tiles):
            # the prepass: the tile's Q in K's lane order, and the accumulator starts
            fl = tile * 8 + np.arange(8)
            live = fl < n_rows
            head = kvh * G + np.where(live, fl % G, 0)
            row = np.where(live, fl // G, 0)
            qs = np.zeros((8, D), np.float16)         # [row][16 c + h]
            for cc in range(16):
                for h in range(16):
                    qs[:, 16 * cc + h] = np.where(live, c.q[head, row, qdim(cc, h)], 0).astype(np.float16)
            nq = np.float32(-1152.0) * qs.astype(np.float32).reshape(8, NB8, 32).sum(-1, dtype=np.float32)   # [row][b]
            tk0 = (tile * 8) // G
            rslot = np.minimum(fl // G, T - 1) - tk0
            states = []
            for sidx in range(n_split):
                m = np.full(8, M_INIT, np.float32)
                l = np.zeros((16, 8), np.float32)     # [lane][row]
                o = np.zeros((16, 16, 8), np.float32)  # [chunk][lane][row]
                s_lo = sidx * split_blocks * 32 + (32 if (red.seam and sidx > 0) else 0)
                s_hi = min(n_kv, (sidx + 1) * split_blocks * 32)
                for t0 in range(s_lo, s_hi, 32):
                    sc = np.zeros((2, 16, 8), np.float32)
                    for gb in range(2):
                        y = t0 + gb * 16
                        r = [K8W(w, y) for w in (0, 8, 16, 24, 32, 40, 48, 56, 60)]
                        W = lambda n: r[n // 8][:, n % 8] if n < 64 else r[8][:, n - 60]
                        acc = np.zeros((16, 8), np.float32)
                        for p in range(4):
                            E = [W(17 * p + i) for i in range(17)]
                            kde = (E[0] & np.uint32(0xFFFF)).astype('<u2').view('<f2').astype(np.float32)
                            kdo = (E[8] >> np.uint32(16)).astype('<u2').view('<f2').astype(np.float32)
                            kx0 = (E[0] >> np.uint32(16)) | (E[8] << np.uint32(16))
                            sets = [(2 * p, [E[1], E[2], E[3], E[4]], [E[5], E[6], E[7], kx0], kde),
                                    (2 * p + 1, [E[9], E[10], E[11], E[12]], [E[13], E[14], E[15], E[16]], kdo)]
                            for b, lo, hi, kd in sets:
                                t = np.tile(nq[:, b], (16, 1)).astype(np.float32)
                                t = dpas(qs[:, 32 * b:32 * b + 16].T, b16(np.stack(lo, 1)), t)
                                t = dpas(qs[:, 32 * b + 16:32 * b + 32].T, b16(np.stack(hi, 1)), t)
                                if red.k_scale2:
                                    kd = kd * kd
                                acc = (acc + t * kd[:, None]).astype(np.float32)
                        kk = np.minimum(y + lanes, n_kv - 1)
                        inb = (y + lanes < (n_kv - 1 if red.tail else n_kv))
                        for rr in range(8):
                            mrow = min(tk0 + rslot[rr], T - 1) * n_kv          # the row's first cell
                            mal = (2 * mrow) % 4 == 0 or mask_read == "block"  # the kernel's mal[]; "block": the first build
                            if y + 16 <= n_kv and mal:
                                mr = block_read_us(mask_flat, mrow + y)
                            else:
                                mr = mask_flat[mrow + kk]
                            hidden = ((mr & 0x7C00) == 0x7C00) | (mr == 0xFBFF)
                            mv = np.where(hidden, 0, mr.view(np.float16).astype(np.float32))
                            vis = live[rr] & inb & ~hidden
                            sc[gb, :, rr] = np.where(vis, acc[:, rr] * scale_log2 + LOG2E * mv, M_INIT)
                    mx = np.maximum(sc[0], sc[1]).max(0)
                    mn = np.maximum(m, mx)
                    alpha = np.exp2(m - mn).astype(np.float32)
                    m = mn
                    l *= alpha[None, :]
                    ps = np.zeros((2, 16, 8), np.float32)
                    for gb in range(2):
                        pf = np.where(sc[gb] > 0.5 * M_INIT, np.exp2(np.minimum(sc[gb] - m[None, :], 0)), 0).astype(np.float32)
                        if not pv32:
                            pf = pf.astype(np.float16).astype(np.float32)        # every DPAS form rounds p to f16
                        l += pf
                        ps[gb] = pf if pv32 else pf[vord]                        # DEC27_PV32 keeps P in key order
                    for b in range(NB8):
                        pe = []
                        vb_ = (b + 1) % NB8 if red.v_scale else b
                        dsc = lambda rows: c.vbytes[rows][:, kvh * HBV + 18 * vb_: kvh * HBV + 18 * vb_ + 2].copy().view('<f2')[:, 0].astype(np.float32)
                        cn = ic = np.float32(1.0)
                        if pv32:
                            # DEC27_PV32: no DPAS. w = p d a (key, row) in f32; the nibbles of key k are byte k of a
                            # lane's 32 (dword k / 4, byte k % 4); O += w (nibble - 8), 16 multiply-adds a key
                            vr = V4R(18 * b + (2 if (b & 1) else 4), t0)
                            if (b & 1) == 0:
                                v0 = V4R(18 * b, t0)
                                vr = vr.copy(); vr[14] = v0[2]; vr[15] = v0[3]
                            byts = np.ascontiguousarray(vr).view(np.uint8).reshape(16, 32)       # [lane = quant byte, key]
                            dk = dsc(np.minimum(t0 + np.arange(32), n_kv - 1))
                            if red.pvnorm_ic:
                                dk = np.ones(32, np.float32)
                            w = (np.concatenate([ps[0], ps[1]], 0) * dk[:, None]).astype(np.float32)     # [key, row]
                            alo, ahi = (o[2 * b] * alpha[None, :]).astype(np.float32), (o[2 * b + 1] * alpha[None, :]).astype(np.float32)
                            for key in range(32):
                                alo = (alo + ((byts[:, key] & 15).astype(np.float32) - 8)[:, None] * w[key][None, :]).astype(np.float32)
                                ahi = (ahi + ((byts[:, key] >> 4).astype(np.float32) - 8)[:, None] * w[key][None, :]).astype(np.float32)
                            o[2 * b], o[2 * b + 1] = alo, ahi
                            continue
                        if pvnorm:
                            # DEC27_PVNORM: the binary exponent of the largest |d| among the block's 32 keys, c = 2^-e
                            dmax = np.abs(dsc(np.minimum(t0 + np.arange(32), n_kv - 1))).max()
                            eb = int((np.float32(dmax).view(np.uint32) >> np.uint32(23)) & np.uint32(0xFF))
                            if eb != 0 and eb != 255:
                                cn = np.array((254 - eb) << 23, np.uint32).view(np.float32)[()]; ic = np.array(eb << 23, np.uint32).view(np.float32)[()]
                            if red.pvnorm_ic:
                                ic = np.float32(1.0)
                        for gb in range(2):
                            kr = np.minimum(t0 + gb * 16 + vord, n_kv - 1)
                            dh = dsc(kr)
                            ds = (cn * dh).astype(np.float32) if pvnorm else np.clip(np.float32(8.0) * dh, -65504.0, 65504.0).astype(np.float32)
                            pe.append((ps[gb] * ds[:, None]).astype(np.float16))
                        ones = np.ones((16, 16), np.float32)
                        ns = dpas(pe[1], ones, dpas(pe[0], ones, np.zeros((16, 8), np.float32))) * np.float32(-1032.0)
                        vr = V4R(18 * b + (2 if (b & 1) else 4), t0)
                        if (b & 1) == 0:
                            v0 = V4R(18 * b, t0)
                            vr = vr.copy()
                            vr[14] = v0[2]; vr[15] = v0[3]
                        for nib, cidx in ((0, 2 * b), (1, 2 * b + 1)):
                            sh = (0, 8) if nib == 0 else (4, 12)
                            acc = (o[cidx] * alpha[None, :] * cn + ns).astype(np.float32)
                            acc = dpas(pe[0], b16(vr[:, :4], xor=0x64006400, mask=0x000F000F, shifts=sh), acc)
                            acc = dpas(pe[1], b16(vr[:, 4:], xor=0x64006400, mask=0x000F000F, shifts=sh), acc)
                            o[cidx] = (acc * ic).astype(np.float32)
                states.append((o, m, l.sum(0, dtype=np.float32)))
            # the merge
            M = np.max([s[1] for s in states], 0)
            O = np.zeros((16, 16, 8), np.float32)
            L = np.zeros(8, np.float32)
            for o_, m_, l_ in states:
                w = np.ones(8, np.float32) if red.nomax else np.where(m_ > 0.5 * M_INIT, np.exp2(m_ - M), 0).astype(np.float32)
                O += w[None, None, :] * o_
                L += w * l_
            inv = np.where(L > 0, np.float32(1.0 if (pvnorm or pv32) else 1.0 / 8.0) / np.where(L > 0, L, 1), 0).astype(np.float32)
            for cc in range(16):
                for ln in range(16):
                    dim = 32 * (cc // 2) + 16 * (cc & 1) + V4J(cc // 2, ln)
                    for rr in range(8):
                        if live[rr]:
                            assert not written[head[rr], row[rr], dim], "an output cell written twice"
                            written[head[rr], row[rr], dim] = True
                            out[head[rr], row[rr], dim] = O[cc, ln, rr] * inv[rr]
    assert written.all(), "an output cell not written"
    return out


def nmse(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


def vfloor_table(n=131072, seed=3):
    """The rounding of the scale in P alone, at the V-floor case's size: flat weights (p in (0.5, 1]) over n keys, V
    blocks of a given largest value, q4_0. NMSE against a double sum of: the default form f16(p 8 d); a mutation of
    the PVNORM form (its operand kept to 4 bits instead of 11); the PVNORM form f16(p d c) / c, c a power of two a 32-key
    block; today's kernel's f16(p) x f16((n - 8) d)."""
    rng = np.random.default_rng(seed)
    out = []
    for amp in (1.0, 1e-2, 1e-4, 1e-5, 1e-6):
        v = (rng.uniform(-1, 1, (n, D)) * amp).astype(np.float32)
        ba = np.float32(0.25) * np.power(np.float32(2), 3 * rng.uniform(0, 1, (n, NB8))).astype(np.float32)
        vb = quant_q4_0((v.reshape(n, NB8, 32) * ba[..., None]).reshape(n, D)).reshape(n, NB8, 18)
        d = vb[..., 0:2].copy().view(np.float16)[..., 0].astype(np.float32)
        nib = np.concatenate([vb[..., 2:] & 15, vb[..., 2:] >> 4], -1).astype(np.float32) - 8
        p = np.exp2(rng.uniform(-1, 0, n)); pf = p.astype(np.float16).astype(np.float32)
        exact = (p[:, None, None] * d.astype(np.float64)[..., None] * nib).sum(0) / p.sum()
        x = (pf[:, None] * np.clip(np.float32(8) * d, -65504, 65504)).astype(np.float16).astype(np.float32)
        dflt = (x[..., None] * nib).sum(0, dtype=np.float64) / (8 * pf.sum(dtype=np.float64))
        xn = (pf[:, None] * (d * 1.0)).astype(np.float32)        # the mutation: the PVNORM operand kept to 4 bits
        m, e = np.frexp(xn); coarse = np.ldexp(np.round(m * 16) / 16, e).astype(np.float32)
        mut = (coarse[..., None] * nib).sum(0, dtype=np.float64) / pf.sum(dtype=np.float64)
        dm = np.abs(d).reshape(n // 32, 32, NB8).max(1)
        cc = np.repeat(np.exp2(-np.floor(np.log2(np.where(dm > 0, dm, 1)))).astype(np.float32), 32, axis=0)
        norm = (((pf[:, None] * (d * cc)).astype(np.float16).astype(np.float32) / cc)[..., None] * nib).sum(0, dtype=np.float64) / pf.sum(dtype=np.float64)
        today = (pf[:, None, None] * (d[..., None] * nib).astype(np.float16).astype(np.float32)).sum(0, dtype=np.float64) / p.sum()
        out.append((amp, nmse(dflt, exact), nmse(mut, exact), nmse(norm, exact), nmse(today, exact)))
    return out


def tail_table(n=131072, seed=5, lo=-6, hi=-3):
    """A dominant key over a deep tail, as a served row at depth has it: one key at p = 1, the other n - 1 at p log-uniform
    in 1e-6 .. 1e-3 (their weights together are tens of times the dominant key's), V block maxima 1, 1e-1, 1e-2. The scale
    in P alone, NMSE against a double sum, of: the default form f16(p 8 d); PVNORM f16(p d c) / c; PV32 (p and p d in f32,
    no f16 operand at all); today's kernel f16(p) x f16((n - 8) d). The floor is on p 8 d (and on p), not on d: at block
    maxima of 1e-2 the tail's p 8 d is 1e-8 .. 1e-5, in or under f16's subnormals."""
    rng = np.random.default_rng(seed)
    out = []
    for amp in (1.0, 1e-1, 1e-2):
        v = (rng.uniform(-1, 1, (n, D)) * amp).astype(np.float32)
        ba = np.float32(0.25) * np.power(np.float32(2), 3 * rng.uniform(0, 1, (n, NB8))).astype(np.float32)
        vb = quant_q4_0((v.reshape(n, NB8, 32) * ba[..., None]).reshape(n, D)).reshape(n, NB8, 18)
        d = vb[..., 0:2].copy().view(np.float16)[..., 0].astype(np.float32)
        nib = np.concatenate([vb[..., 2:] & 15, vb[..., 2:] >> 4], -1).astype(np.float32) - 8
        p = np.power(10.0, rng.uniform(lo, hi, n)); p[n // 2] = 1.0
        pf = p.astype(np.float16).astype(np.float32); p32 = p.astype(np.float32)
        exact = (p[:, None, None] * d.astype(np.float64)[..., None] * nib).sum(0) / p.sum()
        x = (pf[:, None] * np.clip(np.float32(8) * d, -65504, 65504)).astype(np.float16).astype(np.float32)
        dflt = (x[..., None] * nib).sum(0, dtype=np.float64) / (8 * pf.sum(dtype=np.float64))
        dm = np.abs(d).reshape(n // 32, 32, NB8).max(1)
        cc = np.repeat(np.exp2(-np.floor(np.log2(np.where(dm > 0, dm, 1)))).astype(np.float32), 32, axis=0)
        norm = (((pf[:, None] * (d * cc)).astype(np.float16).astype(np.float32) / cc)[..., None] * nib).sum(0, dtype=np.float64) / pf.sum(dtype=np.float64)
        pv32 = ((p32[:, None] * d)[..., None] * nib).sum(0, dtype=np.float32).astype(np.float64) / p32.sum(dtype=np.float64)
        today = (pf[:, None, None] * (d[..., None] * nib).astype(np.float16).astype(np.float32)).sum(0, dtype=np.float64) / p32.sum(dtype=np.float64)
        out.append((amp, nmse(dflt, exact), nmse(norm, exact), nmse(pv32, exact), nmse(today, exact), float(p.sum())))
    return out


def main():
    bad = 0
    print(f"bound against the CPU backend, stated before any card run: {BOUND:g}")
    # the lane-accurate model: a partial tail block and tile, 4 / 6 / 8 rows, 1 and several splits, G 6 and G 8
    for (n_kv, T, G, hkv, p) in ((1013, 6, 6, 4, 1), (1013, 6, 6, 4, 3), (1013, 4, 6, 4, 2), (1013, 8, 6, 4, 2),
                                 (1013, 5, 6, 4, 2), (1013, 7, 6, 4, 4), (1013, 6, 8, 2, 2), (2085, 6, 6, 4, 4)):
        c = Case(n_kv, T, n_head_kv=hkv, G=G)
        ref, cpu = reference(c), cpu_backend(c)
        ker = kernel(c, p)
        e_k, e_c, e_kc = nmse(ker, ref), nmse(cpu, ref), nmse(ker, cpu)
        ok = e_k <= BOUND and e_kc <= BOUND
        bad += not ok
        print(f"n_kv {n_kv} T {T} G {G} hkv {hkv} split {p}: kernel/ref {e_k:.3g}  cpu/ref {e_c:.3g}  kernel/cpu {e_kc:.3g}  {'ok' if ok else 'FAIL'}")
        if (n_kv, T, G, p) in ((1013, 6, 6, 3), (2085, 6, 6, 4)):
            for name in ("v_scale", "tail", "seam", "nomax", "k_scale2"):
                r = Reds(); setattr(r, name, True)
                e = nmse(kernel(c, p, red=r), cpu)
                caught = e > 10 * BOUND
                bad += not caught
                print(f"    red {name}: kernel/cpu {e:.3g}  {'caught (> 10 x bound)' if caught else 'NOT CAUGHT'}")
    # the mask's block read by address: a run off 4 bytes (an odd row over an odd key count). The kernel's guard must
    # be right at every key count; the first build's read (every whole group a block) must FAIL at the odd ones, as it
    # did on the card (1,013 keys: 7.0e-4 there), and pass at the even ones (every run on 4 bytes)
    for n_kv, T, p in ((1013, 6, 2), (1009, 6, 2), (37, 6, 1), (1013, 5, 2), (1012, 6, 2), (1024, 6, 2)):
        c = Case(n_kv, T)
        cpu = cpu_backend(c)
        e_g, e_b = nmse(kernel(c, p), cpu), nmse(kernel(c, p, mask_read="block"), cpu)
        ok = e_g <= BOUND and ((e_b > 10 * BOUND) if n_kv & 1 else (e_b <= BOUND))
        bad += not ok
        print(f"n_kv {n_kv} T {T} split {p}, the mask run by address: the kernel {e_g:.3g}; the first build's block read {e_b:.3g} "
              f"({'an odd key count: must be above 10 x bound' if n_kv & 1 else 'an even key count: every run on 4 bytes'})  {'ok' if ok else 'FAIL'}")
    # an empty split and a fully hidden stretch: a state that weighs 0
    c = Case(1013, 6, masked_cells=((352, 704),))
    e = nmse(kernel(c, 3, split_blocks=11), reference(c))      # split 1 = keys 352 .. 703, all hidden
    ok = e <= BOUND; bad += not ok
    print(f"a split with nothing visible: kernel/ref {e:.3g}  {'ok' if ok else 'FAIL'}")
    # the served shapes with a split that sees nothing (the review's F1): 13,056 keys at 6 rows on 160 compute units is 25
    # splits of 17 blocks, the 25th past the call's 408 blocks; 8,960 cells holding 8,705 tokens, the last split's 256
    # keys hidden from every row but the last. Each must be right, and must be caught by the lost-seam red
    for n_kv, p, empty in ((13056, 25, 0), (8960, 17, 255)):
        c = Case(n_kv, 6, empty=empty)
        ref = reference(c)
        e = nmse(kernel(c, p), ref)
        r = Reds(); r.seam = True
        er = nmse(kernel(c, p, red=r), ref)
        ok = e <= BOUND and er > 10 * BOUND
        bad += not ok
        print(f"n_kv {n_kv} (empty {empty}) split {p} x {((n_kv + 31) // 32 + p - 1) // p} blocks: kernel/ref {e:.3g}, red seam {er:.3g}  {'ok' if ok else 'FAIL'}")
    # the PVNORM and PV32 forms, lane level: the cases' fill, and the V-floor fill at a size the lane model walks
    for vfl in (False, True):
        c = Case(2085, 6, vfloor=vfl)
        ref = reference(c)
        e_d, e_n, e_3 = nmse(kernel(c, 4, pvnorm=False), ref), nmse(kernel(c, 4), ref), nmse(kernel(c, 4, pv32=True), ref)
        r = Reds(); r.pvnorm_ic = True
        e_r, e_r3 = nmse(kernel(c, 4, red=r, pvnorm=True), ref), nmse(kernel(c, 4, red=r, pv32=True), ref)
        ok = e_n <= BOUND and e_3 <= BOUND and not (e_r <= 10 * BOUND) and not (e_r3 <= 10 * BOUND) and ((e_d > 10 * BOUND) if vfl else (e_d <= BOUND))      # a NaN counts as caught
        bad += not ok
        print(f"n_kv 2085 {'V-floor fill' if vfl else 'the cases fill'}: the kernel {e_n:.3g}; left out of the series: f16(p 8 d) {e_d:.3g}, P V in f32 {e_3:.3g}; red 6: the kernel without 1/c {e_r:.3g}, the f32 form without the scale {e_r3:.3g}  "
              f"{'ok' if ok else 'FAIL'}" + (" (f16(p 8 d) FAILS the V-floor fill: why it is not the kernel)" if vfl else ""))
    # the V-floor case at its own size, the rounding alone
    print("the scale in P at 131,072 keys, flat weights (NMSE): |v| block maximum, default f16(p 8 d), PVNORM at 4 bits (a mutation), PVNORM, today's f16(p) f16(v)")
    for amp, e_d, e_m, e_n, e_t in vfloor_table():
        print(f"    {amp:7.0e}   {e_d:9.2e}   {e_m:9.2e}   {e_n:9.2e}   {e_t:9.2e}")
        if amp == 1e-6:
            ok = e_d > 10 * 2e-5 and e_n <= 2e-5 and e_m > 10 * 2e-5; bad += not ok
            print(f"    the V-floor case (1e-6): default above 10 x 2e-5 (it FAILS the case), PVNORM within 2e-5, PVNORM at 4 bits above 10 x 2e-5 (the case can fail): {'ok' if ok else 'FAIL'}")
        if amp == 1.0:
            ok = e_d <= 2e-5; bad += not ok
    # a dominant key over a deep tail (the delta review's D1): every form
    print("a dominant key over a tail of p 1e-6 .. 1e-3 at 131,072 keys (NMSE): |v| block maximum, default f16(p 8 d), PVNORM, PV32 (f32), today's f16(p) f16(v)")
    for amp, e_d, e_n, e_3, e_t, psum in tail_table():
        print(f"    {amp:7.0e}   {e_d:9.2e}   {e_n:9.2e}   {e_3:9.2e}   {e_t:9.2e}   (the tail's weight: {psum - 1:.1f} x the dominant key's)")
        ok = e_3 <= 1e-9; bad += not ok
        if not ok:
            print("    PV32 is not exact here: FAIL")
    print("the same with a thinner tail, p 1e-7 .. 1e-5 (p itself at f16's floor: every form that rounds p to f16 shares it, today's kernel too)")
    for amp, e_d, e_n, e_3, e_t, psum in tail_table(lo=-7, hi=-5):
        print(f"    {amp:7.0e}   {e_d:9.2e}   {e_n:9.2e}   {e_3:9.2e}   {e_t:9.2e}   (the tail's weight: {psum - 1:.2f} x the dominant key's)")
        ok = e_3 <= 1e-9; bad += not ok
    # the depth cases, by the direct model: what the tail and seam reds do to the answer at the served depths
    for n_kv, p in ((4096, 8), (32768, 32), (77056, 32), (131072, 32)):
        c = Case(n_kv, 6)
        ref = reference(c)
        sb = ((n_kv + 31) // 32 + p - 1) // p
        e_tail = nmse(reference(c, drop_last=True), ref)
        e_seam = nmse(reference(c, lost=[(s * sb * 32, s * sb * 32 + 32) for s in range(1, p)]), ref)
        ok = e_tail > 10 * BOUND and e_seam > 10 * BOUND
        bad += not ok
        print(f"n_kv {n_kv} split {p} (direct): a dropped last key {e_tail:.3g}, a lost seam block a split {e_seam:.3g}  {'both > 10 x bound' if ok else 'A RED BELOW 10 x THE BOUND'}")
    print("DEC27-MODEL:", "PASS" if bad == 0 else f"FAIL ({bad})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
