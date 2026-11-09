# Emulates the index math of the 0116 Q3_K kernels (planes, lb_expand / get_rows, mv_q3_K's lane map,
# the few-column kernel's half codes and swapped activations, the 2D GEMM's b operands in P4 order) against ggml's dequantize_row_q3_K.
# Pure python, small sizes. Usage: python3 -I tools/q3k_layout_emul.py [--red]   (exit 0: PASS). Not the kernels: a transcription of their arithmetic.
import random, struct, sys

RED = "--red" in sys.argv[1:]
random.seed(1)
BLK = 110

def half_to_float(h):
    return struct.unpack('<e', struct.pack('<H', h))[0]

def ref_dequant(blk):
    # ggml-quants.c dequantize_row_q3_K, one block
    hm = blk[0:32]; q = blk[32:96]; sc12 = blk[96:108]
    d_all = half_to_float(blk[108] | (blk[109] << 8))
    aux = list(struct.unpack('<3I', bytes(sc12))) + [0]
    kmask1, kmask2 = 0x03030303, 0x0f0f0f0f
    tmp = aux[2]
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)
    scales = []
    for a in aux:
        for k in range(4):
            v = (a >> (8 * k)) & 0xFF
            scales.append(v - 256 if v > 127 else v)
    y = []
    is_ = 0; m = 1; qo = 0
    for n in range(2):
        shift = 0
        for j in range(4):
            dl = d_all * (scales[is_] - 32); is_ += 1
            for l in range(16):
                y.append(dl * (((q[qo + l] >> shift) & 3) - (0 if (hm[l] & m) else 4)))
            dl = d_all * (scales[is_] - 32); is_ += 1
            for l in range(16):
                y.append(dl * (((q[qo + l + 16] >> shift) & 3) - (0 if (hm[l + 16] & m) else 4)))
            shift += 2
            m <<= 1
        qo += 32
    return y

FIELDS = [(108, 2), (0, 32), (32, 64), (96, 12)]   # ggml_cl_lb_fields, Q3_K

def to_planes(blocks):
    nblk = len(blocks)
    out = bytearray()
    for off, size in FIELDS:
        for b in blocks:
            out += bytes(b[off:off + size])
    assert len(out) == nblk * BLK
    return out

def from_planes(S, nblk):
    blocks = []
    for i in range(nblk):
        b = bytearray(BLK)
        plane = 0
        for off, size in FIELDS:
            b[off:off + size] = S[plane + i * size: plane + (i + 1) * size]
            plane += nblk * size
        blocks.append(b)
    return blocks

def u32(S, off):
    assert off % 4 == 0, off   # the kernels' 32-bit loads must be aligned
    return struct.unpack_from('<I', S, off)[0]

def kbyte(w, k):
    return (w[k // 4] >> (8 * (k & 3))) & 0xFF

def q3k_scale(w, is_):
    lo = (kbyte(w, is_) & 0xF) if is_ < 8 else (kbyte(w, is_ - 8) >> 4)
    hi = 0 if RED else (kbyte(w, 8 + (is_ & 3)) >> (2 * (is_ >> 2))) & 3
    return (lo | (hi << 4)) - 32

def q3k_w4(q, nh, j, hb):
    return (((q >> (2 * j)) & 0x03030303) | ((((nh >> hb) & 0x01010101) * 0xFC) & 0xFFFFFFFF)) & 0xFFFFFFFF

def sbytes(w):
    return [((w >> (8 * k)) & 0xFF) - 256 if ((w >> (8 * k)) & 0xFF) > 127 else ((w >> (8 * k)) & 0xFF) for k in range(4)]

def lb_expand(S, nblk, bi, p):
    # mul_mm_kq_f16.cl lb_expand TYPE 12 and kernel_get_rows_q3_K
    n, j = p >> 2, p & 3
    d = half_to_float(struct.unpack_from('<H', S, bi * 2)[0])
    qw = [u32(S, nblk * 34 + bi * 64 + 32 * n + 4 * i) for i in range(8)]
    nh = [(~u32(S, nblk * 2 + bi * 32 + 4 * i)) & 0xFFFFFFFF for i in range(8)]
    sw = [u32(S, nblk * 98 + bi * 12 + 4 * i) for i in range(3)]
    s0 = d * q3k_scale(sw, 8 * n + 2 * j); s1 = d * q3k_scale(sw, 8 * n + 2 * j + 1)
    v = []
    for i in range(8):
        c = sbytes(q3k_w4(qw[i], nh[i], j, 4 * n + j))
        s = s0 if i < 4 else s1
        v += [s * x for x in c]
    return v

def mv_row(S, nblk, nb, r, x):
    # mv_q3_K: sum over the 16 lanes of one row (float activations instead of the q8_1 terms)
    acc = 0.0
    for lane in range(16):
        p = lane & 7; n = p >> 2; t0 = (p & 3) * 8; is0 = 8 * n + (t0 >> 4)
        for ib in range(lane >> 3, nb, 2):
            b32 = ib * 8 + 4 * n
            b = r * nb + ib
            q = [u32(S, nblk * 34 + b * 64 + 32 * n + t0 + 4 * k) for k in range(2)]
            nh = [(~u32(S, nblk * 2 + b * 32 + t0 + 4 * k)) & 0xFFFFFFFF for k in range(2)]
            sw = [u32(S, nblk * 98 + b * 12 + 4 * k) for k in range(3)]
            d = half_to_float(struct.unpack_from('<H', S, b * 2)[0])
            for j in range(4):
                w = sbytes(q3k_w4(q[0], nh[0], j, 4 * n + j)) + sbytes(q3k_w4(q[1], nh[1], j, 4 * n + j))
                xs = x[(b32 + j) * 32 + t0:(b32 + j) * 32 + t0 + 8]
                acc += d * q3k_scale(sw, is0 + 2 * j) * sum(a * bb for a, bb in zip(w, xs))
    return acc

def few_row(S, nblk, nb, r, x):
    # kernel_mul_mm_few_q3_K_f16: per k16 step, the b operand's halves 1024 + u against the staged
    # activation (middle two of every four swapped), the chain from -1028 S
    acc = 0.0
    for ib in range(nb):
        b = r * nb + ib
        D = half_to_float(struct.unpack_from('<H', S, b * 2)[0])
        sw = [u32(S, nblk * 98 + b * 12 + 4 * k) for k in range(3)]
        hq = [u32(S, nblk * 2 + b * 32 + 4 * k) for k in range(8)]
        wq = [u32(S, nblk * 34 + b * 64 + 4 * k) for k in range(16)]
        for n in range(2):
            for j in range(4):
                for hh in range(2):
                    s = 8 * n + 2 * j + hh
                    w = wq[8 * n + 4 * hh: 8 * n + 4 * hh + 4]
                    h = hq[4 * hh: 4 * hh + 4]
                    halves = []
                    for u in range(4):
                        r0 = ((w[u] >> (2 * j)) & 0x00030003) | (((h[u] >> (4 * n + j)) & 0x00010001) << 2) | 0x64006400
                        r1 = ((w[u] >> (8 + 2 * j)) & 0x00030003) | (((h[u] >> (8 + 4 * n + j)) & 0x00010001) << 2) | 0x64006400
                        for rr in (r0, r1):
                            halves += [half_to_float(rr & 0xFFFF), half_to_float(rr >> 16)]
                    k0 = ib * 256 + 16 * s
                    xv = x[k0:k0 + 16]
                    staged = []
                    for i in range(0, 16, 4):
                        staged += [xv[i], xv[i + 2], xv[i + 1], xv[i + 3]]
                    S_ = sum(xv)
                    t = -1028.0 * S_ + sum(a * bb for a, bb in zip(halves, staged))
                    acc += t * D * q3k_scale(sw, s)
    return acc

def q3k_scale_2d(w, is_):
    # mul_mm_kq_2d.cl q3k_scale_2d (its own byte selection)
    v8 = (w[0] if is_ < 4 else w[1]) if is_ < 8 else (w[0] if is_ < 12 else w[1])
    lo = (v8 >> (8 * (is_ & 3))) & 0xF if is_ < 8 else (v8 >> (8 * (is_ & 3) + 4)) & 0xF
    hi = 0 if RED else (w[2] >> (8 * (is_ & 3) + 2 * (is_ >> 2))) & 3
    return (lo | (hi << 4)) - 32

def gemm2d_dequant(S, nblk, bi):
    # kernel_mul_mm_kq2d_q3_K: the weights its b operands carry, put back at their k (slot s of a
    # 16-wide operand holds k = P4(s)); exact arithmetic for (u - 4) * D * sc
    P4 = lambda s: (s & ~3) | ((s & 1) << 1) | ((s >> 1) & 1)
    D = half_to_float(struct.unpack_from('<H', S, bi * 2)[0])
    sw = [u32(S, nblk * 98 + bi * 12 + 4 * k) for k in range(3)]
    hmw = [u32(S, nblk * 2 + bi * 32 + 4 * i) for i in range(8)]
    out = [None] * 256
    for nn in range(2):
        qw = [u32(S, nblk * 34 + bi * 64 + 32 * nn + 4 * i) for i in range(8)]
        for j in range(4):
            sb, is_, hb = 4 * nn + j, 8 * nn + 2 * j, 4 * nn + j
            s0, s1 = D * q3k_scale_2d(sw, is_), D * q3k_scale_2d(sw, is_ + 1)
            halves = []
            for i in range(8):
                w = qw[i] >> (2 * j); h = hmw[i] >> hb
                for pw in (((w & 0x00030003) | ((h & 0x00010001) << 2) | 0x64006400),
                           (((w >> 8) & 0x00030003) | (((h >> 8) & 0x00010001) << 2) | 0x64006400)):
                    sc = s0 if i < 4 else s1
                    halves += [(half_to_float(pw & 0xFFFF) - 1028) * sc, (half_to_float(pw >> 16) - 1028) * sc]
            for slot in range(32):
                out[32 * sb + 16 * (slot // 16) + P4(slot % 16)] = halves[slot]
    return out

def main():
    ne01, nb = 6, 4            # 6 rows of 4 blocks: an even block count
    nblk = ne01 * nb
    blocks = []
    for i in range(nblk):
        b = bytearray(random.getrandbits(8) for _ in range(BLK))
        dh = struct.unpack('<H', struct.pack('<e', random.uniform(-0.05, 0.05)))[0]
        b[108] = dh & 0xFF; b[109] = dh >> 8
        blocks.append(b)
    S = to_planes(blocks)
    assert from_planes(S, nblk) == blocks
    ref = [ref_dequant(b) for b in blocks]
    # dequant (tile GEMM, GET_ROWS)
    bad = 0
    for bi in range(nblk):
        got = []
        for p in range(8):
            got += lb_expand(S, nblk, bi, p)
        bad += sum(1 for a, b in zip(got, ref[bi]) if a != b)
    print("dequant: %d of %d weights differ" % (bad, nblk * 256))
    bad2 = 0
    for bi in range(nblk):
        got = gemm2d_dequant(S, nblk, bi)
        bad2 += sum(1 for a, b in zip(got, ref[bi]) if abs(a - b) > 1e-12 * max(1.0, abs(b)))
    print("2D GEMM b operands: %d of %d weights differ" % (bad2, nblk * 256))
    x = [random.uniform(-1, 1) for _ in range(nb * 256)]
    worst_mv = worst_few = 0.0
    for r in range(ne01):
        w = sum((ref[r * nb + ib] for ib in range(nb)), [])
        exact = sum(a * b for a, b in zip(w, x))
        mv = mv_row(S, nblk, nb, r, x)
        fw = few_row(S, nblk, nb, r, x)
        scale = max(1e-9, abs(exact))
        worst_mv = max(worst_mv, abs(mv - exact) / scale)
        worst_few = max(worst_few, abs(fw - exact) / scale)
    print("matvec lane map: worst relative error %.3g" % worst_mv)
    print("few-column codes: worst relative error %.3g" % worst_few)
    ok = bad == 0 and bad2 == 0 and worst_mv < 1e-9 and worst_few < 1e-6
    print("RESULT", "PASS" if ok else "FAIL", "(red)" if RED else "")
    sys.exit(0 if ok else 1)

main()
