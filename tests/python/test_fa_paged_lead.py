"""Patches 0129x/0129y/0129z's device-free gate: the dead next-graph lead's cause,
its named early returns, and the verify-only request totals.

Window 292 (`measured-here`, 65k B60): OVERLAP+DEBUG printed 2,240 "commands (lead)"
lines and not one "next-graph lead" line; every request-end summary read "0 reused, 0
re-copied"; and the VERIFY=1 arm with OVERLAP unset printed no totals at all.

Red mutations, each run and seen failing before it was restored:
  (a) the tail call in the split-decode branch dropped (the `return` kept, the
      `ggml_cl_fa_paged_overlap_after` call removed): the branch cell fails -- that is
      window 292's code, and it is why the lead was dead;
  (b) the `runs < 0` early return made silent (its WARN removed): the named-fallbacks
      cell fails (CLAUDE.md: no silent fallbacks);
  (c) the roll-up gated on `overlap_on()` only (the verify switch dropped): the
      verify-only cell fails -- window 292's A-uverify arm.
Source-text assertions, not numerics.
"""
import os
import unittest
from pathlib import Path

OPENCL = "ggml/src/ggml-opencl/ggml-opencl.cpp"


def _llama_root():
    raw = os.environ.get("LLAMA_SRC")
    if not raw:
        raise unittest.SkipTest("LLAMA_SRC is unset: the patched llama tree is not available")
    root = Path(raw)
    for cand in (root, root.parent):
        if (cand / OPENCL).exists():
            return cand
    raise unittest.SkipTest("LLAMA_SRC has no ggml-opencl.cpp: it lacks patch 0119")


class FaPagedLead(unittest.TestCase):

    def setUp(self):
        self.root = _llama_root()
        self.t = {f: (self.root / f).read_text() for f in (OPENCL,)}

    def test_the_split_branch_calls_the_tail(self):
        t = self.t[OPENCL]
        i = t.find("this branch returns here, before the tail call below")
        self.assertTrue(i >= 0, "0129x's note is absent")
        j = t.rfind("const size_t merge_wg", 0, i)
        self.assertTrue(j >= 0)
        block = t[j:t.find("        return;\n", i)]
        self.assertIn("if (paged_overlap_copied && paged_overlap_parity >= 0) {", block,
                      "the split branch must call the tail before it returns")
        self.assertIn("ggml_cl_fa_paged_overlap_after(backend_ctx, paged_overlap_parity);", block)
        # the served decode takes this branch; the tail call below it is unreachable there
        self.assertIn("backend_ctx->enqueue_ndrange_kernel(k_merge, 3, merge_gws, merge_lws, dst);", block,
                      "the call must follow both kernels")

    def test_every_early_return_names_itself(self):
        t = self.t[OPENCL]
        i = t.find("static void ggml_cl_fa_paged_prefetch_head(ggml_backend_opencl_context * ctx) {")
        self.assertTrue(i >= 0)
        block = t[i:t.find("static bool ggml_cl_fa_paged_prefetch_same", i)]
        for reason in ("no paged FA in the graph's chain",
                       "ring slot is not cached",
                       "plan is not USM host memory",
                       "enqueue_pages -1"):
            self.assertIn(reason, block, f"the early return '{reason}' is silent")
        self.assertIn("bool pre_head_dead_warned = false;", t, "the WARNs are one-shot a process")
        self.assertIn("o.pre_head_dead_warned = true;", block)

    def test_the_summary_carries_issued_and_prints_for_verify_only(self):
        t = self.t[OPENCL]
        self.assertIn("next-graph lead: %d issued, %d reused, %d re-copied", t)
        self.assertIn("int issued = 0, fell_back = 0, reused = 0, recopied = 0, pre_issued = 0;", t)
        self.assertIn("g_fa_paged_totals.pre_issued += o.pre_issued;", t)
        i = t.find("if (ggml_cl_fa_paged_overlap_on() || ggml_cl_fa_paged_verify_on() || ggml_cl_fa_paged_timeline()) {")
        self.assertTrue(i >= 0, "0129z's roll-up gate is absent (0130p adds the timeline's counters)")
        block = t[i:t.find("backend_ctx->pair_perm_t = nullptr;", i)]
        self.assertIn("g_fa_paged_totals.v_nodes    += o.v_nodes;", block,
                      "the verify totals roll up without OVERLAP")
        self.assertIn("o.pre_issued = o.pre_reused = o.pre_recopied = 0;", block)
        self.assertIn("static bool ggml_cl_fa_paged_verify_on();   // 0129z", t,
                      "the roll-up needs the verify switch declared above graph_compute")


if __name__ == "__main__":
    unittest.main()


class FaPagedPostFA(unittest.TestCase):
    """Patch 0130d's device-free gate: VERIFY=3 hashes the same rows after the FA, before
    the consumed marker, reads both back non-blocking, compares pre against post, and the
    red switch NOCONSUME drops exactly the consumed wait.

    Red mutations, each run and seen failing before it was restored:
      (a) the post hash moved after the consumed marker in the split-decode branch: the
          ordering cell fails -- a copy that lands during the FA is then read after it;
      (b) the readback's CL_FALSE made CL_TRUE: the non-blocking cell fails (the graph
          would wait at every node's post hash);
      (c) NOCONSUME applied to the prefetch's consumed wait too: the exactly-one-wait cell
          fails -- the red arm must drop only `overlap_issue`'s wait.
    """

    def setUp(self):
        self.root = _llama_root()
        self.t = (self.root / OPENCL).read_text()

    def test_the_post_hash_follows_the_fa_and_precedes_the_marker(self):
        t = self.t
        post = "ggml_cl_fa_paged_verify3_post(backend_ctx, true);"   # 0130g adds the branch flag
        # the split-decode branch: after k_merge, before the consumed marker
        i = t.find("this branch returns here, before the tail call below")
        self.assertTrue(i >= 0)
        j = t.rfind("const size_t merge_wg", 0, i)
        self.assertTrue(j >= 0)
        block = t[j:t.find("        return;\n", i)]
        k = block.find("backend_ctx->enqueue_ndrange_kernel(k_merge, 3, merge_gws, merge_lws, dst);")
        p = block.find(post)
        a = block.find("ggml_cl_fa_paged_overlap_after(backend_ctx, paged_overlap_parity);")
        self.assertTrue(k >= 0 and p >= 0 and a >= 0, "the split branch lacks kernel, post or marker")
        self.assertGreater(p, k, "the post hash must follow the FA's kernels")
        self.assertLess(p, a, "the post hash must precede the consumed marker")
        # the tail path: after the tail kernels, before its consumed marker
        i = t.find("// 0130d: VERIFY=3's post hash, after the tail kernels and before the consumed marker")
        self.assertTrue(i >= 0, "the tail's post site is absent")
        j = t.find("if (paged_overlap_copied && paged_overlap_parity >= 0) {", i)
        tailblock = t[i:j]
        self.assertIn("ggml_cl_fa_paged_verify3_post(backend_ctx, false);", tailblock)
        self.assertLess(t.find(post, i), j, "the tail's post hash must precede its marker")
        self.assertGreater(t.find("// 0129n: this layer's kernels are enqueued", 0), i,
                           "the 0129n marker comment precedes the tail's after call")

    def test_the_pair_is_read_back_non_blocking(self):
        t = self.t
        self.assertIn("clEnqueueReadBuffer(ctx->queue, out, CL_FALSE, 0, host.size()*sizeof(uint32_t),", t,
                      "the readback must be CL_FALSE")
        self.assertIn("host.data(), 0, nullptr, ev));", t, "the readback must keep its event")
        i = t.find("static void ggml_cl_fa_paged_verify3_collect(")
        self.assertTrue(i >= 0)
        block = t[i:t.find("static void ggml_cl_fa_kv_stream_pages", i)]
        self.assertIn("clWaitForEvents(1, &e.ev[kv][0])", block)
        self.assertIn("clWaitForEvents(1, &e.ev[kv][1])", block)
        self.assertIn("e.dev[kv][0][p*nc + c] != e.dev[kv][1][p*nc + c]", block,
                      "the comparison must be pre against post per (page, chunk)")
        # collected at the next graph's start and flushed at the request's end
        self.assertIn("if (ggml_cl_fa_paged_verify_mode() == 3 && g_fa_paged_verify3_ctx)", t)
        self.assertIn("ggml_cl_fa_paged_verify3_collect(g_fa_paged_verify3_ctx);", t)

    def test_a_change_warns_once(self):
        t = self.t
        self.assertIn('"ggml_opencl: paged FA post-FA changed: node %d page %d chunk %d\\n"', t)
        self.assertIn("int  v_post_warned = 0;", t, "the WARN is one-shot a process")
        i = t.find("if (bad > 0 && !o.v_post_warned) {")
        self.assertTrue(i >= 0, "the change WARN is absent or ungated")

    def test_the_summary_carries_the_post_totals(self):
        t = self.t
        self.assertIn("long long v_post_changes = 0;", t)
        self.assertIn("post-FA changes %lld", t)
        self.assertIn("g_fa_paged_totals.v_post_changes += o.v_post_changes;", t)
        i = t.find("if (ggml_cl_fa_paged_overlap_on() || ggml_cl_fa_paged_verify_on() || ggml_cl_fa_paged_timeline()) {")
        self.assertTrue(i >= 0)
        block = t[i:t.find("backend_ctx->pair_perm_t = nullptr;", i)]
        self.assertIn("o.v_post_changes = 0;", block)

    def test_noconsume_drops_exactly_the_consumed_wait(self):
        t = self.t
        self.assertEqual(t.count("ggml_cl_fa_paged_noconsume()"), 2,
                         "the switch must be defined once and called exactly once")
        self.assertIn("const bool nocons = ggml_cl_fa_paged_noconsume();", t,
                      "the drop switch is read once and the drop is the only effect")
        self.assertIn("if (o.consumed[parity] && !nocons) wl[nw++] = o.consumed[parity];", t)
        # the cross-step prefetch's wait is not the drop
        self.assertIn("cl_uint  nw = (o.consumed[p] != nullptr && !nofix) ? 1 : 0;", t,
                      "NOCONSUME must not touch the prefetch's consumed wait (NOFIX does)")


class FaPagedPostSelftest(unittest.TestCase):
    """Patch 0130g's device-free gate: GGML_OPENCL_FA_PAGED_VERIFY_SELFTEST=3 writes one byte
    into the K slot's first streamed row after the FA and before the post hash, so the collect
    and compare path must fire ("post-FA changes 1").

    Red mutations, each run and seen failing before it was restored:
      (a) the switch accepting any nonzero value (`atoi(e) != 0`): the exactly-three cell
          fails -- VERIFY_SELFTEST also names other modes;
      (b) the fill moved into the pre hash's site (`verify3_begin`): the placement cell fails
          -- the pre hash then hashes the written byte too and nothing differs;
      (c) the once-a-request reset removed: the one-write cell fails.
    """

    def setUp(self):
        self.root = _llama_root()
        self.t = (self.root / OPENCL).read_text()

    def test_the_switch_is_mode_3_only(self):
        t = self.t
        i = t.find('getenv("GGML_OPENCL_FA_PAGED_VERIFY_SELFTEST")')
        self.assertTrue(i >= 0, "the self-test switch is absent")
        block = t[i:t.find("\n", t.find("return on;", i))]
        self.assertIn("atoi(e) == 3", block,
                      "the self-test must run only for VERIFY_SELFTEST=3")

    def test_one_byte_after_the_fa_before_the_post_hash(self):
        t = self.t
        i = t.find("static void ggml_cl_fa_paged_verify3_post(")
        j = t.find("static void ggml_cl_fa_paged_verify3_collect(", i)
        block = t[i:j]
        fill = block.find("clEnqueueFillBuffer(ctx->queue, e.buf[0], &pat, 1, off, 1, 0, nullptr, nullptr)")
        hash = block.find("ggml_cl_fa_paged_verify3_hash_read(ctx, e.buf[kv]")
        self.assertGreater(fill, 0, "the fill is absent from the post site")
        self.assertGreater(hash, fill, "the fill must precede the post hash")
        self.assertIn("const cl_ulong off = e.base[0] + (cl_ulong) e.rows[0] * e.stride[0];", block,
                      "the write must land in the first streamed row's chunk 0")
        self.assertIn("decode && ggml_cl_fa_paged_selftest3()", block,
                      "the write must be the decode branch's, not the prompt tail's")

    def test_one_node_a_request_and_its_warn(self):
        t = self.t
        self.assertIn("!o.v_self3_done && e.node == 0", t,
                      "one streamed node a request: the first decode node only")
        self.assertIn("o.v_self3_done = 1;", t)
        self.assertIn("g_fa_paged_verify3_ctx->fa_paged.v_self3_done = 0;", t,
                      "the one-a-request flag resets at the request's end")
        self.assertIn('"ggml_opencl: paged FA post-FA self-test: node %d page %d chunk 0 one byte written after the FA\\n"', t)


class FaPagedForcedRace(unittest.TestCase):
    """Patch 0130h's device-free gate: the busy-wait between the FA's split kernels and the
    merge, on the compute queue, so it lengthens FA j's hold of its slot and adds nothing
    before FA j starts; the NOCONSUME drop is in `issue` only, and the cross-step prefetch's
    consumed wait stays -- that is why window 294's red arm did not land.

    Red mutations, each run and seen failing before it was restored:
      (a) the delay enqueued before the split kernel's dispatch: the ordering cell fails --
          it would add time before FA j starts, not during its hold;
      (b) the call removed from the split branch: the ordering cell fails;
      (c) the prefetch's consumed wait made conditional on !noconsume: the kept-wait cell
          fails -- that would make NOCONSUME land in the wrong place.
    """

    def setUp(self):
        self.root = _llama_root()
        self.t = (self.root / OPENCL).read_text()

    def test_the_switch_and_the_spin(self):
        t = self.t
        self.assertIn('getenv("GGML_OPENCL_FA_PAGED_DELAY")', t, "the delay switch is absent")
        self.assertIn("static const long   n = e != nullptr ? atol(e) : 0;", t,
                      "the delay is an iteration count")
        body = t[t.find("static void ggml_cl_fa_paged_delay("):]
        self.assertIn("if (n <= 0) return;", body, "unset or 0 must not delay")
        self.assertIn("fa_paged_delay(__global uint * sink, uint n)", t, "the spin kernel is absent")
        self.assertIn("clEnqueueNDRangeKernel(ctx->queue, o.delay_kernel, 1, nullptr, gws, nullptr, 0, nullptr, nullptr));", t,
                      "the spin runs on the compute queue, one work-item")

    def test_the_delay_sits_between_the_split_kernels_and_the_merge(self):
        t = self.t
        split = t.find("backend_ctx->enqueue_ndrange_kernel(k_split, 3, fd_gws, fd_lws, dst);")
        call  = t.find("ggml_cl_fa_paged_delay(backend_ctx);")
        merge = t.find("cl_kernel k_merge = backend_ctx->fa.f32_merge.at(dk_dv);")
        self.assertTrue(split >= 0 and call >= 0 and merge >= 0)
        self.assertGreater(call, split, "the delay must follow the FA's split kernels")
        self.assertLess(call, merge, "the delay must precede the merge")
        self.assertIn("// 0130h: the forced race's busy-wait, after the split kernels and before the merge", t)

    def test_noconsume_stays_in_issue_and_the_prefetch_keeps_its_wait(self):
        t = self.t
        self.assertEqual(t.count("ggml_cl_fa_paged_noconsume()"), 2,
                         "NOCONSUME is defined once and called exactly once, in issue")
        self.assertIn("cl_uint  nw = (o.consumed[p] != nullptr && !nofix) ? 1 : 0;", t,
                      "the cross-step prefetch's consumed wait is kept unset; NOFIX drops it")

class FaPagedTimelineAndNoFix(unittest.TestCase):
    """Patches 0130p/0130q/0130r's device-free gate: the timeline's marks and margin, the
    NOFIX drop at 0130c's site, and every development switch's startup line and per-request
    count.

    Red mutations, each run and seen failing before it was restored:
      (a) the FA split start stamp moved after k_merge: the ordering cell fails -- the margin
          would be taken against the merge, not against the FA's read;
      (b) NOFIX applied to `issue`'s consumed wait instead of the prefetch's: the site cell
          fails -- the red must be at 0130c, not the intra-graph wait;
      (c) the request-end dev-switch line removed: the counted cell fails -- window 296's
          defect was a log that could not say the red arm had run red.
    """

    def setUp(self):
        self.root = _llama_root()
        self.t = (self.root / OPENCL).read_text()

    def test_compute_queue_profiling_at_runtime_0130u(self):
        # 0130u: the compute queue takes CL_QUEUE_PROFILING_ENABLE when the timeline switch is
        # set at runtime, so a timeline build need not define GGML_OPENCL_PROFILING (whose
        # per-kernel recording drains the host every 2048 kernels). Red: the |= line removed.
        t = self.t
        i = t.find('getenv("GGML_OPENCL_FA_PAGED_TIMELINE");\n        if (tl != nullptr && atoi(tl) != 0) {')
        self.assertGreater(i, 0, "0130u: no runtime timeline check at the compute queue's creation")
        j = t.find("backend_ctx->queue = clCreateCommandQueue(context, device, command_queue_props", i)
        self.assertGreater(j, i, "0130u: the check must precede the compute queue's creation")
        self.assertIn("command_queue_props |= CL_QUEUE_PROFILING_ENABLE;\n            GGML_LOG_INFO", t[i:j],
                      "0130u: the switch must set the compute queue's profiling property and say so")

    def test_the_switch_and_the_two_profiling_sources(self):
        t = self.t
        self.assertIn('getenv("GGML_OPENCL_FA_PAGED_TIMELINE")', t, "the timeline switch is absent")
        self.assertIn("const cl_command_queue_properties cqp = ggml_cl_fa_paged_timeline() ? CL_QUEUE_PROFILING_ENABLE : 0;", t,
                      "the copy queue is profiling-enabled only when the switch is on")
        self.assertIn("clGetEventProfilingInfo(e.ev, CL_PROFILING_COMMAND_START", t)
        self.assertIn("build with GGML_OPENCL_PROFILING", t,
                      "a build without compute-queue profiling must say so, not print zeros")
        self.assertIn("ggml_cl_fa_paged_tl_finish(g_fa_paged_ctx)", t,
                      "the request's end must print and release the stamps")

    def test_the_marks_and_the_margin(self):
        t = self.t
        split = t.find("backend_ctx->enqueue_ndrange_kernel(k_split, 3, fd_gws, fd_lws, dst);")
        start = t.find("ggml_cl_fa_paged_tl_mark(backend_ctx, backend_ctx->queue, 2, paged_overlap_parity, 0, nullptr);")
        end   = t.find("ggml_cl_fa_paged_tl_mark(backend_ctx, backend_ctx->queue, 3, paged_overlap_parity, 0, nullptr);")
        merge = t.find("backend_ctx->enqueue_ndrange_kernel(k_merge, 3, merge_gws, merge_lws, dst);")
        mend  = t.find("ggml_cl_fa_paged_tl_mark(backend_ctx, backend_ctx->queue, 4, paged_overlap_parity, 0, nullptr);")
        post  = t.find("ggml_cl_fa_paged_verify3_post(backend_ctx, true);")
        pst   = t.find("ggml_cl_fa_paged_tl_mark(backend_ctx, backend_ctx->queue, 6, paged_overlap_parity, 0, nullptr);")
        self.assertTrue(min(split, start, end, merge, mend, post, pst) >= 0, "a stamp is absent")
        self.assertLess(start, split, "the FA split start stamp must precede the split kernel")
        self.assertGreater(end, split, "the FA split end stamp must follow it")
        self.assertLess(end, merge, "the split end must precede the merge")
        self.assertGreater(mend, merge, "the merge stamp must follow it")
        self.assertLess(pst, post, "the post start stamp must precede the post hash")
        self.assertIn("ggml_cl_fa_paged_tl_keep(ctx, 5, parity, o.consumed[parity]);", t,
                      "the consumed marker is stamped")
        # the copy start marker carries the copy's waits
        self.assertIn("ggml_cl_fa_paged_tl_mark(ctx, o.q, 0, parity, nw, wl);", t)
        self.assertIn("ggml_cl_fa_paged_tl_mark(ctx, o.q, 7, p, nw, wl);", t)
        # the margin: copy k+2 against FA k's split end, minimum and negative count
        self.assertIn("const long long m = (long long) cs[k + 1] - (long long) fe[k];", t)
        self.assertIn("negative %d", t, "the margin line must say whether any pair landed negative")

    def test_nofix_is_the_prefetch_wait_only(self):
        t = self.t
        self.assertIn('getenv("GGML_OPENCL_FA_PAGED_NOFIX")', t, "the NOFIX switch is absent")
        self.assertIn("const bool nofix = ggml_cl_fa_paged_nofix();", t)
        self.assertIn("cl_uint  nw = (o.consumed[p] != nullptr && !nofix) ? 1 : 0;", t,
                      "NOFIX must drop the cross-step prefetch's consumed wait (0130c's site)")
        self.assertEqual(t.count("ggml_cl_fa_paged_nofix()"), 2,
                         "NOFIX is defined once and called exactly once, in the prefetch")
        # and not issue's wait: NOCONSUME keeps that meaning, defined once and called once
        self.assertEqual(t.count("ggml_cl_fa_paged_noconsume()"), 2,
                         "NOCONSUME stays defined once and called once, in issue")

    def test_every_switch_says_its_value_and_the_summary_counts(self):
        t = self.t
        for s in ("GGML_OPENCL_FA_PAGED_NOCONSUME=1", "GGML_OPENCL_FA_PAGED_NOFIX=1",
                  "GGML_OPENCL_FA_PAGED_DELAY=%ld", "GGML_OPENCL_FA_PAGED_VERIFY=%d",
                  "GGML_OPENCL_FA_PAGED_VERIFY_SELFTEST=3", "GGML_OPENCL_FA_PAGED_TIMELINE=1"):
            self.assertIn("paged FA switch " + s, t, "the startup line for %s is absent" % s)
        self.assertIn("paged FA dev switches (request end): NOCONSUME consumed waits dropped %lld; ",
                      t, "the request end must count the switch arms")
        self.assertIn("NOFIX cross-step waits dropped %lld; delay kernels %lld; self-test writes %lld", t)
        self.assertIn("++o.delay_kernels;", t, "delay kernels are counted")
        self.assertIn("++o.self3_writes;", t, "the injected write is counted")
        self.assertIn("++o.nc_drops;", t)
        self.assertIn("++o.nofix_drops;", t)
