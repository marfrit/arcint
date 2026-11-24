import sys
p = sys.argv[1]
s = open(p).read()
old = '''    // 0198's original rule: a contiguous order. Red-proven by port_gate_test.cpp on the
    // export's prefill lines - the model's tensors are permuted views.
    const int64_t q_elem = (int64_t) q->nb[0];
    const int64_t k_elem = (int64_t) k->nb[0];
    const int64_t v_elem = (int64_t) v->nb[0];
    if (q->nb[1] != q->ne[0]*q_elem || q->nb[2] != q->ne[1]*q->nb[1] ||
        k->nb[1] != k->ne[0]*k_elem || k->nb[2] != k->ne[1]*k->nb[1] ||
        v->nb[1] != v->ne[0]*v_elem || v->nb[2] != v->ne[1]*v->nb[1]) REFUSE("layout");
'''
new = '''    // 0198's rule after the the first card window decline: the strides the kernel and the prepass index,
    // not a contiguous order. The model's FLASH_ATTN_EXT inputs are permuted views (q nb1
    // 12288 / nb2 512, K/V nb1 2048 / nb2 256), which is exactly what these terms allow.
    const int64_t D = v->ne[0];
    // q: flash_attn_q_to_f16 reads vload8 at head*nb2 + row*nb1; a row's D floats span D*4
    // bytes from the row base, so nb1 and nb2 must cover that and keep the vload8 alignment.
    if ((int64_t) q->nb[0] != 4 || q->nb[1] < (size_t) (D * 4) || q->nb[1] % 32 != 0 ||
        q->nb[2] < (size_t) (D * 4) || q->nb[2] % 32 != 0) REFUSE("layout-q");
    // k/v: the 2D block reads, surface (D*2-1, n_kv-1, nb[1]-1) and base kvh*nb[2]
    const int64_t kD = k->ne[0];
    if ((int64_t) k->nb[0] != 2 || k->nb[1] < (size_t) (kD * 2) || k->nb[1] % 16 != 0 ||
        k->nb[2] < (size_t) (kD * 2) || k->nb[2] % 64 != 0) REFUSE("layout-k");
    if ((int64_t) v->nb[0] != 2 || v->nb[1] < (size_t) (D * 2) || v->nb[1] % 16 != 0 ||
        v->nb[2] < (size_t) (D * 2) || v->nb[2] % 64 != 0) REFUSE("layout-v");
'''
assert old in s
s = s.replace(old, new)
# per-tensor stride terms for mask, table, sink, dst
s = s.replace('''    if (mask->ne[0] < n_kv || mask->ne[1] < rows) REFUSE("mask");''','''    if (mask->ne[0] < n_kv || mask->ne[1] < rows || mask->nb[1] < (size_t) (n_kv * 2)) REFUSE("mask");''')
s = s.replace('''    if (sinks_q->type != GGML_TYPE_F32 || sinks_q->ne[0] != n_head || sinks_q->ne[1] < rows ||
        sinks_q->ne[2] != 1) REFUSE("sinks");''','''    if (sinks_q->type != GGML_TYPE_F32 || sinks_q->ne[0] != n_head || sinks_q->ne[1] < rows ||
        sinks_q->ne[2] != 1 || (int64_t) sinks_q->nb[0] != 4 || sinks_q->nb[1] != (size_t) (n_head * 4)) REFUSE("sinks");''')
s = s.replace('''    if (table->type != GGML_TYPE_F32 || table->ne[0] <= 0 || table->ne[1] != n_head ||
        table->ne[2] < rows || table->ne[3] != 1) REFUSE("bias-table");''','''    if (table->type != GGML_TYPE_F32 || table->ne[0] <= 0 || table->ne[1] != n_head ||
        table->ne[2] < rows || table->ne[3] != 1) REFUSE("bias-table");
    const int64_t n_bins = table->ne[0];
    if ((int64_t) table->nb[0] != 4 || table->nb[1] != (size_t) (n_bins * 4) ||
        table->nb[2] != (size_t) (n_head * n_bins * 4)) REFUSE("bias-table-strides");''')
s = s.replace('''    if (kpos->type != GGML_TYPE_I32 || qpos->type != GGML_TYPE_I32 ||
        kpos->ne[0] < n_kv || qpos->ne[0] < rows) REFUSE("positions");''','''    if (kpos->type != GGML_TYPE_I32 || qpos->type != GGML_TYPE_I32 || (int64_t) kpos->nb[0] != 4 ||
        (int64_t) qpos->nb[0] != 4 || kpos->ne[0] < n_kv || qpos->ne[0] < rows) REFUSE("positions");''')
s = s.replace('''    if (cflag->type != GGML_TYPE_I8 || rflag->type != GGML_TYPE_I8 ||
        cflag->ne[0] < (n_kv + 15) / 16 || rflag->ne[0] < (rows + 7) / 8) REFUSE("flags");''','''    if (cflag->type != GGML_TYPE_I8 || rflag->type != GGML_TYPE_I8 ||
        (int64_t) cflag->nb[0] != 1 || (int64_t) rflag->nb[0] != 1 ||
        cflag->ne[0] < (n_kv + 15) / 16 || rflag->ne[0] < (rows + 7) / 8) REFUSE("flags");''')
s = s.replace('''    if (dst == NULL || dst->type != GGML_TYPE_F32 || dst->ne[0] != v->ne[0] ||
        dst->ne[1] != n_head || dst->ne[2] < rows) REFUSE("dst");''','''    if (dst == NULL || dst->type != GGML_TYPE_F32 || dst->ne[0] != v->ne[0] ||
        dst->ne[1] != n_head || dst->ne[2] < rows || dst->nb[1] < (size_t) (v->ne[0] * 4) ||
        dst->nb[2] < (size_t) (v->ne[0] * 4)) REFUSE("dst");''')
open(p, 'w').write(s)
print('v2 applied')
