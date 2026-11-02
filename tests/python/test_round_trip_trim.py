"""Patch 0083's device-free cell: round-trip discipline's three cuts and their
switch.

LLAMA_SYNC_TRIM defaults to 0 (all bits off, byte-identical to before): bit 1
skips a context synchronize when this context queued nothing since the last
one; bit 2 keeps the KV rotation matrices on the cache's device instead of
uploading two graph inputs per graph; bit 4 reads a batch's backend-sampled ids
with one copy instead of one small read per row. The queue and per-read costs
are the fork's measured-here numbers on its stack, not measured here; the card
arms (served A/B, LLAMA_SYNC_TRIM=0/3) are run by Claude in an announced
window.

Red mutations, each run and seen failing before it was restored:
  (a) the default 0 -> 3: the default-off cell fails;
  (b) the trim predicate's `!sync_pending &&` dropped: the predicate cell fails;
  (c) the device-rotation return dropped from build_input_v_rot: the rotation
      cell fails;
  (d) the sampled-id concat's bit 4 -> 8: the one-read cell fails.
Standard library only (unittest); the recipe's `ctest -L unit` runs it.
"""
import os
import unittest
from pathlib import Path


def _llama_root():
    raw = os.environ.get("LLAMA_SRC")
    if not raw:
        raise unittest.SkipTest("LLAMA_SRC is unset: the patched llama tree is not available")
    root = Path(raw)
    for cand in (root, root.parent):
        if (cand / "src" / "llama-context.cpp").exists():
            return cand
    raise unittest.SkipTest("LLAMA_SRC has no src/llama-context.cpp")


class RoundTripTrim(unittest.TestCase):
    def setUp(self):
        root = self.root = _llama_root()
        self.ctx = (root / "src" / "llama-context.cpp").read_text()
        self.hdr = (root / "src" / "llama-context.h").read_text()
        self.kv = (root / "src" / "llama-kv-cache.cpp").read_text()
        self.kvh = (root / "src" / "llama-kv-cache.h").read_text()
        self.graph = (root / "src" / "llama-graph.cpp").read_text()
        self.graphh = (root / "src" / "llama-graph.h").read_text()

    def test_switch_defaults_off(self):
        self.assertIn('getenv("LLAMA_SYNC_TRIM");', self.ctx)
        self.assertRegex(self.ctx, r"return e \? atoi\(e\) : 0;")
        self.assertIn("int llama_sync_trim();", (self.root / "src" / "llama-impl.h").read_text())

    def test_sync_trim_predicate(self):
        self.assertIn("(llama_sync_trim() & 1) && !sync_pending && !cparams.pipeline_parallel", self.ctx)
        self.assertIn("bool sync_pending       = true;", self.hdr)
        # the queuing sites: encode, decode, graph_compute, opt_epoch_iter
        self.assertGreaterEqual(self.ctx.count("sync_pending = true;"), 4)
        self.assertIn("sync_pending = false;", self.ctx)

    def test_rotations_kept_on_device(self):
        self.assertIn("(llama_sync_trim() & 2) && ctxs_bufs.size() == 1", self.kv)
        self.assertIn("ggml_backend_alloc_ctx_tensors_from_buft(attn_rot_dev_ctx.get(), buft)", self.kv)
        self.assertIn("std::unordered_map<int64_t, ggml_tensor *> attn_rot_dev;", self.kvh)
        # both builders return the device copy before allocating a host input
        k = self.kv.split("llama_kv_cache::build_input_k_rot", 1)[1].split("llama_kv_cache::build_input_v_rot", 1)[0]
        v = self.kv.split("llama_kv_cache::build_input_v_rot", 1)[1]
        for body in (k, v):
            self.assertIn("attn_rot_dev.find(", body)
            self.assertRegex(body, r"return it->second;")
        # both setters skip a non-input tensor (the device copy, already filled)
        self.assertEqual(self.kv.count("if (!(dst->flags & GGML_TENSOR_FLAG_INPUT))"), 2)

    def test_one_sampled_id_read(self):
        self.assertIn("(llama_sync_trim() & 4) && n_rows > 1", self.graph)
        self.assertIn('ggml_concat(ctx0, ids, res->t_sampled[i], 0)', self.graph)
        self.assertIn("ggml_tensor *              t_sampled_ids = nullptr;", self.graphh)
        self.assertIn("res->t_sampled_ids != nullptr && sampling.sampled.has_data()", self.ctx)
        self.assertIn("ggml_backend_tensor_get_async(b, t, sampling.sampled.data + n_outputs_prev", self.ctx)
        # the old per-row read stays as the fallback (switch off is byte-identical)
        self.assertIn("copy_tensor_async_rows(res->t_sampled,    sampling.sampled", self.ctx)


if __name__ == "__main__":
    unittest.main()
