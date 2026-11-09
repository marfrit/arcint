# Emulates the index math of the 0117 Q2_K kernels (the planes, the tile GEMM's and GET_ROWS' decode,
# mv_q2_K's lane map with its min term, the few-column kernel's half codes against swapped activations,
# the 2D GEMM's b operands in P4 order) against ggml's dequantize_row_q2_K.
# Pure python, small sizes. Usage: python3 -I tools/q2k_layout_emul.py [--red]   (exit 0: PASS). Not the
# kernels: a transcription of their arithmetic.
import random, struct, sys

RED = "--red" in sys.argv[1:]
random.seed(2)
BLK = 84

def half_to_float(h):
    return struct.unpack('<e', struct.pack('<H', h))[0]

def ref_dequant(blk):
    # ggml-quants.c dequantize_row_q2_K, one block: scales[16] | qs[64] | d | dmin
    sc = blk[0:16]; q = blk[16:80]
    d = half_to_float(blk[80] | (blk[81] << 8)); mn = half_to_float(blk[82] | (blk[83] << 8))
    y = []; is_ = 0; qo = 0
    for n in range(2):
        shift = 0
        for j in range(4):
            s = sc[is_]; is_ += 1
            dl = d * (s & 0xF); ml = mn * (s >> 4)
            for l in range(16): y.append(dl * ((q[qo + l] >> shift) & 3) - ml)
            s = sc[is_]; is_ += 1
            dl = d * (s & 0xF); ml = mn * (s >> 4)
            for l in range(16): y.append(dl * ((q[qo + l + 16] >> shift) & 3) - ml)
            shift += 2
        qo += 32
    return y

FIELDS = [(80, 4), (0, 16), (16, 64)]   # ggml_cl_lb_fields, Q2_K: dm | scales | qs

def to_planes(blocks):
    out = bytearray()
    for off, size in FIELDS:
        for b in blocks:
            out += bytes(b[off:off + size])
    return out

def u32(S, off):
    assert off % 4 == 0, off
    return struct.unpack_from('<I', S, off)[0]

def dm_of(S, bi):
    return half_to_float(struct.unpack_from('<H', S, 4 * bi)[0]), half_to_float(struct.unpack_from('<H', S, 4 * bi + 2)[0])

def sm(b):
    return ((b >> 4), (b & 0xF)) if RED else ((b & 0xF), (b >> 4))

def sbyte(S, nblk, bi, k):
    return S[nblk * 4 + bi * 16 + k]

def v32(S, nblk, bi, p):
    # mul_mm_kq_f16.cl q2k_v32 and kernel_get_rows_q2_K
    n, j = p >> 2, p & 3
    d, mn = dm_of(S, bi)
    out = []
    for i in range(8):
        w = u32(S, nblk * 20 + bi * 64 + 32 * n + 4 * i)
        sc, m = sm(sbyte(S, nblk, bi, 8 * n + 2 * j + (1 if i >= 4 else 0)))
        for k in range(4):
            out.append(d * sc * ((w >> (8 * k + 2 * j)) & 3) - mn * m)
    return out

def mv_row(S, nblk, nb, r, x):
    acc = 0.0
    for lane in range(16):
        p = lane & 7; n = p >> 2; t0 = (p & 3) * 8; is0 = 8 * n + (t0 >> 4)
        for ib in range(lane >> 3, nb, 2):
            b32 = ib * 8 + 4 * n; b = r * nb + ib
            q = [u32(S, nblk * 20 + b * 64 + 32 * n + t0 + 4 * k) for k in range(2)]
            d, mn = dm_of(S, b)
            for j in range(4):
                w = [(q[k] >> (8 * e + 2 * j)) & 3 for k in range(2) for e in range(4)]
                xs = x[(b32 + j) * 32 + t0:(b32 + j) * 32 + t0 + 8]
                sc, m = sm(sbyte(S, nblk, b, is0 + 2 * j))
                acc += d * sc * sum(a * c for a, c in zip(w, xs)) - mn * m * sum(xs)
    return acc

def few_row(S, nblk, nb, r, x):
    acc = 0.0
    for ib in range(nb):
        b = r * nb + ib
        D, M = dm_of(S, b)
        wq = [u32(S, nblk * 20 + b * 64 + 4 * k) for k in range(16)]
        for n in range(2):
            for j in range(4):
                for hh in range(2):
                    s = 8 * n + 2 * j + hh
                    w = wq[8 * n + 4 * hh: 8 * n + 4 * hh + 4]
                    halves = []
                    for u in range(4):
                        r0 = ((w[u] >> (2 * j)) & 0x00030003) | 0x64006400
                        r1 = ((w[u] >> (8 + 2 * j)) & 0x00030003) | 0x64006400
                        for rr in (r0, r1):
                            halves += [half_to_float(rr & 0xFFFF), half_to_float(rr >> 16)]
                    xv = x[ib * 256 + 16 * s: ib * 256 + 16 * s + 16]
                    staged = []
                    for i in range(0, 16, 4):
                        staged += [xv[i], xv[i + 2], xv[i + 1], xv[i + 3]]
                    S_ = sum(xv)
                    t = -1024.0 * S_ + sum(a * c for a, c in zip(halves, staged))
                    sc, m = sm(sbyte(S, nblk, b, s))
                    acc += t * D * sc - S_ * M * m
    return acc

def gemm2d_dequant(S, nblk, bi):
    P4 = lambda s: (s & ~3) | ((s & 1) << 1) | ((s >> 1) & 1)
    D, M = dm_of(S, bi)
    out = [None] * 256
    for nn in range(2):
        qw = [u32(S, nblk * 20 + bi * 64 + 32 * nn + 4 * i) for i in range(8)]
        for j in range(4):
            sb, is_ = 4 * nn + j, 8 * nn + 2 * j
            sc0, m0 = sm(sbyte(S, nblk, bi, is_)); sc1, m1 = sm(sbyte(S, nblk, bi, is_ + 1))
            halves = []
            for i in range(8):
                w = qw[i] >> (2 * j)
                dl, nml = (D * sc0, -M * m0) if i < 4 else (D * sc1, -M * m1)
                for pw in (((w & 0x00030003) | 0x64006400), (((w >> 8) & 0x00030003) | 0x64006400)):
                    halves += [(half_to_float(pw & 0xFFFF) - 1024) * dl + nml, (half_to_float(pw >> 16) - 1024) * dl + nml]
            for slot in range(32):
                out[32 * sb + 16 * (slot // 16) + P4(slot % 16)] = halves[slot]
    return out

def main():
    ne01, nb = 6, 4
    nblk = ne01 * nb
    blocks = []
    for i in range(nblk):
        b = bytearray(random.getrandbits(8) for _ in range(BLK))
        for o, v in ((80, random.uniform(0, 0.05)), (82, random.uniform(0, 0.05))):
            h = struct.unpack('<H', struct.pack('<e', v))[0]
            b[o] = h & 0xFF; b[o + 1] = h >> 8
        blocks.append(b)
    S = to_planes(blocks)
    ref = [ref_dequant(b) for b in blocks]
    close = lambda a, b: abs(a - b) <= 1e-12 * max(1.0, abs(b))
    bad = sum(1 for bi in range(nblk) for a, b in zip(sum((v32(S, nblk, bi, p) for p in range(8)), []), ref[bi]) if not close(a, b))
    print("dequant: %d of %d weights differ" % (bad, nblk * 256))
    bad2 = sum(1 for bi in range(nblk) for a, b in zip(gemm2d_dequant(S, nblk, bi), ref[bi]) if not close(a, b))
    print("2D GEMM b operands: %d of %d weights differ" % (bad2, nblk * 256))
    x = [random.uniform(-1, 1) for _ in range(nb * 256)]
    worst_mv = worst_few = 0.0
    for r in range(ne01):
        w = sum((ref[r * nb + ib] for ib in range(nb)), [])
        exact = sum(a * b for a, b in zip(w, x))
        scale = max(1e-9, abs(exact))
        worst_mv = max(worst_mv, abs(mv_row(S, nblk, nb, r, x) - exact) / scale)
        worst_few = max(worst_few, abs(few_row(S, nblk, nb, r, x) - exact) / scale)
    print("matvec lane map: worst relative error %.3g" % worst_mv)
    print("few-column codes: worst relative error %.3g" % worst_few)
    ok = bad == 0 and bad2 == 0 and worst_mv < 1e-9 and worst_few < 1e-6
    print("RESULT", "PASS" if ok else "FAIL", "(red)" if RED else "")
    sys.exit(0 if ok else 1)

main()
