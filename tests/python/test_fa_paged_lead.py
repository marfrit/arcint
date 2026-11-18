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
        r = t.find("        return;", i)
        block = t[j:r]
        # 0131n: the after-step is the dispatch's guard (paged_after), which runs at this return,
        # after both kernels; before 0131n it was an explicit call in this block
        guarded = ("~paged_after_guard()" in t and "return;   // 0131n: paged_after runs the after-step here" in t[r:r + 80])
        explicit = ("if (paged_overlap_copied && paged_overlap_parity >= 0) {" in block
                    and "ggml_cl_fa_paged_overlap_after(backend_ctx, paged_overlap_parity);" in block)
        self.assertTrue(guarded or explicit, "the split branch must run the tail before it returns")
        # the served decode takes this branch; the tail call below it is unreachable there
        self.assertIn("backend_ctx->enqueue_ndrange_kernel(k_merge, 3, merge_gws, merge_lws, dst, tl_now ? &ev_merge : nullptr);", block,
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
        k = block.find("backend_ctx->enqueue_ndrange_kernel(k_merge, 3, merge_gws, merge_lws, dst, tl_now ? &ev_merge : nullptr);")
        p = block.find(post)
        a = block.find("ggml_cl_fa_paged_overlap_after(backend_ctx, paged_overlap_parity);")
        guarded = "~paged_after_guard()" in t
        if a < 0 and guarded:
            a = len(block)   # 0131n: the guard runs the after-step at this branch's return
        self.assertTrue(k >= 0 and p >= 0 and a >= 0, "the split branch lacks kernel, post or marker")
        self.assertGreater(p, k, "the post hash must follow the FA's kernels")
        self.assertLess(p, a, "the post hash must precede the consumed marker")
        # the tail path: after the tail kernels, before its consumed marker
        i = t.find("// 0130d: VERIFY=3's post hash, after the tail kernels and before the consumed marker")
        self.assertTrue(i >= 0, "the tail's post site is absent")
        j = t.find("if (paged_overlap_copied && paged_overlap_parity >= 0) {", i)
        if guarded:
            j = t.find("\n}\n", i)   # 0131n: the guard runs at the function's end
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
        split = t.find("backend_ctx->enqueue_ndrange_kernel(k_split, 3, fd_gws, fd_lws, dst, tl_now ? &ev_split : nullptr);")
        call  = t.find("if (!ggml_cl_fa_paged_delay_at_splits()) ggml_cl_fa_paged_delay(backend_ctx);")
        merge = t.find("cl_kernel k_merge = backend_ctx->fa.f32_merge.at(dk_dv);")
        self.assertTrue(split >= 0 and call >= 0 and merge >= 0)
        self.assertGreater(call, split, "the delay must follow the FA's split kernels")
        self.assertLess(call, merge, "the delay must precede the merge")
        self.assertIn("// 0130h: the forced race's busy-wait, after the split kernels and before the merge;", t)
        # 0130y: with DELAY_AT=splits the same spin goes out BEFORE the split kernels instead
        early = t.find("if (ggml_cl_fa_paged_delay_at_splits()) ggml_cl_fa_paged_delay(backend_ctx);   // 0130y: before the readers")
        self.assertGreater(early, 0)
        self.assertLess(early, split, "0130y: the AT=splits spin must precede the split kernels")

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

    def test_common_kv_resident_flag_reaches_the_context_0131i(self):
        # 0131i: the common tools (llama-perplexity's KL) take --kv-resident MIB into
        # llama_context_params.kv_resident_size (bytes). Red: the flag parsed but not mapped.
        root = self.root
        arg = (root / "common/arg.cpp").read_text()
        com = (root / "common/common.cpp").read_text()
        self.assertIn('{"--kv-resident"}, "MIB"', arg, "0131i: no --kv-resident flag")
        self.assertIn("params.kv_resident_mib = (uint32_t) value;", arg)
        self.assertIn("cparams.kv_resident_size  = params.kv_resident_mib * 1024u * 1024u;", com,
                      "0131i: the flag must reach the context parameters")

    def test_timeline_markers_absent_with_the_switch_off_0131b(self):
        # 0131b removed 0130p's always-enqueued markers; 0131c removed the timeline's markers
        # altogether (they serialized the queues and, with the switch off, cost the overlap its
        # gain). The cell now asserts the absence, which subsumes 0131b's guard.
        t = self.t
        self.assertNotIn("ggml_cl_fa_paged_tl_mark", t)

    def test_timeline_marker_wait_list_null_when_empty_0130v(self):
        # 0130v (window 298's -57 abort) guarded a marker's wait list. 0131c removed the
        # timeline's markers; the hash kernel's wait list is the only new one and it is
        # guarded the same way, so the rule survives in its new place.
        t = self.t
        self.assertIn("const cl_uint nw = ws != nullptr ? 1 : 0;   // 0131d: never a host wait, only the copy's event", t)
        self.assertIn("cl_event wl[1] = { ws != nullptr ? *ws : nullptr };", t)
        self.assertNotIn("CL_CHECK(clEnqueueMarkerWithWaitList(q, nw, nw ? wl : nullptr, &ev))", t)

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
        self.assertIn("carry no profiling (the compute queue needs", t,
                      "a build without compute-queue profiling must say so, not print zeros")
        self.assertIn("ggml_cl_fa_paged_tl_finish(g_fa_paged_ctx)", t,
                      "the request's end must print and release the stamps")

    def test_the_stamps_and_the_margin(self):
        # 0131c: the stamps are the commands' own events, so their order is the enqueue order:
        # k_split's event is stamped after it is enqueued, k_merge's after k_merge, the post
        # hash's inside verify3_post. The margin takes the split's END (role 3) against the
        # copy's first write START (role 0).
        t = self.t
        split = t.find("backend_ctx->enqueue_ndrange_kernel(k_split, 3, fd_gws, fd_lws, dst, tl_now ? &ev_split : nullptr);")
        start = t.find("ggml_cl_fa_paged_tl_keep(backend_ctx, 2, paged_overlap_parity, fa_slot, ev_split);")
        end   = t.find("ggml_cl_fa_paged_tl_keep(backend_ctx, 3, paged_overlap_parity, fa_slot, ev_split);")
        merge = t.find("backend_ctx->enqueue_ndrange_kernel(k_merge, 3, merge_gws, merge_lws, dst, tl_now ? &ev_merge : nullptr);")
        mend  = t.find("ggml_cl_fa_paged_tl_keep(backend_ctx, 4, paged_overlap_parity, fa_slot, ev_merge);")
        post  = t.find("ggml_cl_fa_paged_verify3_post(backend_ctx, true);")
        pst   = t.find("ggml_cl_fa_paged_tl_keep(ctx, 6, e.node, e.slot, kev);")
        self.assertTrue(min(split, start, end, merge, mend, post, pst) >= 0, "a stamp is absent")
        self.assertGreater(start, split, "0131c: the split start stamp follows its enqueue")
        self.assertGreater(end, start, "0131c: the split end stamp is the same event, after the start")
        self.assertLess(end, merge, "the split's stamps must precede the merge's")
        self.assertGreater(mend, merge, "0131c: the merge stamp follows its enqueue")
        self.assertLess(pst, post, "0131c: the post start stamp is inside verify3_post")
        self.assertIn("ggml_cl_fa_paged_tl_keep(ctx, 5, ni, parity, o.consumed[parity]);", t,
                      "the consumed marker is stamped (role 5, the marker the copy waits anyway)")
        self.assertIn("const long long st = x.role == 2 || x.role == 6 ? (long long) x.st : (long long) x.en;", t)
        self.assertIn("if (x.role == 0) { cs.push_back((long long) x.st - (long long) base_en);", t)
        self.assertIn("const long long m = cs[k + 1] - fe[k];", t)
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


class FaPagedTimelineSlotHostDelay(unittest.TestCase):
    """Patches 0130w/0130x/0130y's gate: the events carry node and ring slot separately,
    the host submit time comes with the stamps, and DELAY_AT=splits places the spin before
    the split kernels -- the only placement that widens the ring's read window.

    Red mutations, each run and seen failing before it was restored:
      (a) an FA stamp back to the node index in the slot field: the slot cell fails;
      (b) the split-branch delay back to after the split kernels with AT=splits set: the
          placement cell fails;
      (c) the submit time dropped from the kept stamps: the host-time cell fails.
    """

    def setUp(self):
        self.root = _llama_root()
        self.t = (self.root / OPENCL).read_text()

    def test_the_fa_stamps_carry_the_ring_slot(self):
        t = self.t
        self.assertIn("struct fa_paged_tl_ev { int role; int node; int slot; cl_event ev; long long host; };", t)
        self.assertIn("const int fa_slot = (paged_overlap_parity >= 0 &&", t,
                      "the FA stamps must look the slot up from the chain node")
        self.assertIn("ggml_cl_fa_paged_overlap_slot(backend_ctx, backend_ctx->fa_paged.chain[paged_overlap_parity]) : -1;", t)
        # 0131c: the stamps come from k_split's and k_merge's events and the post hash kernel's
        for role in ("2", "3", "4"):
            self.assertIn("ggml_cl_fa_paged_tl_keep(backend_ctx, %s, paged_overlap_parity, fa_slot," % role, t,
                          "the FA stamp %s must carry node and slot" % role)
        self.assertIn("ggml_cl_fa_paged_tl_keep(ctx, 6, e.node, e.slot, kev);", t,
                      "the post-start stamp must carry node and slot")
        self.assertIn("slot %d node %d copy start %lld end %lld submit %lld", t)

    def test_the_host_submit_time(self):
        t = self.t
        self.assertIn("clock_gettime(CLOCK_MONOTONIC, &ts);", t, "the host stamps need the monotonic clock")
        self.assertIn("clGetDeviceAndHostTimer(ctx->device, &devb, &hostb)", t,
                      "the base must take the device and host clocks together")
        self.assertIn("o.tl.push_back({ role, node, slot, ev, ggml_cl_fa_paged_tl_host_ns() });", t)
        self.assertIn("ggml_cl_fa_paged_tl_host(backend_ctx, 8, paged_overlap_parity, fa_slot);", t)
        self.assertIn("ggml_cl_fa_paged_tl_host(backend_ctx, 9, paged_overlap_parity, fa_slot);", t)
        self.assertIn("start-submit %lld", t, "a copy's device start is printed against its submit time")

    def test_delay_at_splits_stays_before_the_readers(self):
        t = self.t
        self.assertIn('getenv("GGML_OPENCL_FA_PAGED_DELAY_AT")', t)
        self.assertIn('strcmp(e, "splits") == 0', t, "only `splits` moves the spin")
        self.assertIn("if (ggml_cl_fa_paged_delay_at_splits()) ggml_cl_fa_paged_delay(backend_ctx);   // 0130y: before the readers", t)
        i = t.find("// 0130y: before the readers")
        j = t.find("backend_ctx->enqueue_ndrange_kernel(k_split, 3, fd_gws, fd_lws, dst, tl_now ? &ev_split : nullptr);")
        self.assertGreater(j, i, "the AT=splits spin must precede the split kernels")
        k = t.find("if (!ggml_cl_fa_paged_delay_at_splits()) ggml_cl_fa_paged_delay(backend_ctx);")
        self.assertGreater(k, j, "the default placement stays after the split kernels")
        self.assertIn("paged FA switch GGML_OPENCL_FA_PAGED_DELAY_AT=splits", t)


class FaPagedTimelineNoMarkerAndPostRead(unittest.TestCase):
    """Patches 0131c/0131d: the timeline stamps with the commands' own events (no marker of
    its own), and VERIFY=4 hashes the slot's rows without a host wait, comparing pre-read
    against the reader's hash and the reader's hash against a re-hash that waits the slot's
    next copy by event.

    Red mutations, each run and seen failing before it was restored:
      (a) a marker back in the recording start: the no-marker cell fails;
      (b) the copy's stamps taken from a marker instead of the write commands' events: the
          copy-event cell fails;
      (c) the re-hash enqueued without the copy's ready event: the ready-wait cell fails;
      (d) VERIFY=4's pairs collected at the graph start again: the collect-site cell fails.
    """

    def setUp(self):
        self.root = _llama_root()
        self.t = (self.root / OPENCL).read_text()

    def test_no_timeline_marker_of_its_own(self):
        t = self.t
        self.assertNotIn("ggml_cl_fa_paged_tl_mark", t,
                         "0131c: the timeline must stamp with command events, not markers of its own")
        self.assertNotIn("tl_mark(", t)
        # the markers left in the overlap are the copy's readiness, the after-inputs order and
        # the consumed marker that the next copy waits anyway -- none of them the timeline's
        self.assertIn("CL_CHECK(clEnqueueMarkerWithWaitList(o.q, 0, nullptr, &o.ready[parity]));"
                      "   // the slot's readiness; the FA waits it", t)
        self.assertIn("CL_CHECK(clEnqueueMarkerWithWaitList(ctx->queue, 0, nullptr, &o.consumed[parity]));", t)
        self.assertIn("ggml_cl_fa_paged_tl_keep(ctx, 5, ni, parity, o.consumed[parity]);", t)
        # the recording start enqueues nothing at all; the base is derived from the stamps
        i = t.find("static void ggml_cl_fa_paged_overlap_graph(")
        self.assertGreater(i, 0)
        j = t.find("// start a recording", i)
        k = t.find("\nstatic void ", j + 10)   # the next definition ends the recording-start block
        self.assertGreater(j, i)
        seg = t[j:k]
        self.assertNotIn("clEnqueueMarker", seg,
                         "0131c: the recording start must enqueue no command")
        self.assertIn("base the earliest device start among the stamps", t)

    def test_the_stamps_are_the_commands_events(self):
        t = self.t
        self.assertIn("cl_event * ev_out = nullptr", t, "enqueue_ndrange_kernel must hand out its event")
        self.assertIn("cl_event * ev_first = nullptr, cl_event * ev_last = nullptr", t,
                      "enqueue_pages must hand out its first and last write's event")
        # the split's one event is both stamps; the merge's END is its stamp
        self.assertIn("ggml_cl_fa_paged_tl_keep(backend_ctx, 2, paged_overlap_parity, fa_slot, ev_split);", t)
        self.assertIn("ggml_cl_fa_paged_tl_keep(backend_ctx, 3, paged_overlap_parity, fa_slot, ev_split);", t)
        self.assertIn("ggml_cl_fa_paged_tl_keep(backend_ctx, 4, paged_overlap_parity, fa_slot, ev_merge);", t)
        self.assertEqual(t.count("tl ? &ev_first : nullptr, tl ? &ev_last : nullptr);"), 2,
                         "0131c: both copy sites ask enqueue_pages for the writes' events")
        self.assertIn("ggml_cl_fa_paged_tl_keep(ctx, 0, ni, parity, ev_first);", t)
        self.assertIn("ggml_cl_fa_paged_tl_keep(ctx, 1, ni, parity, ev_last);", t)
        self.assertIn("ggml_cl_fa_paged_tl_keep(ctx, 7, 0, p, ev_first);", t)
        # role 2 prints the event's START, role 3 its END (both from one event)
        self.assertIn("const long long st = x.role == 2 || x.role == 6 ? (long long) x.st : (long long) x.en;", t)

    def test_verify4_does_not_block(self):
        t = self.t
        self.assertIn("ggml_cl_fa_paged_verify4_rehash(ctx, parity, o.ready[parity]);", t)
        self.assertIn("ggml_cl_fa_paged_verify4_rehash(ctx, p, o.pre_ready);", t)
        self.assertIn("const cl_uint nw = ws != nullptr ? 1 : 0;   // 0131d: never a host wait, only the copy's event", t)
        # stage 1: the reader's hash against the state it started from; stage 2: that hash
        # against the re-hash behind the slot's next copy
        self.assertIn("ggml_opencl: paged FA post-read changed: node %d page %d chunk %d", t)
        self.assertIn("pre-read vs read %lld mismatches, \"", t)
        self.assertIn("\"read vs re-hash %lld, re-hash missing on %d pairs, stage-1 self-test writes %lld", t)
        self.assertIn('getenv("GGML_OPENCL_FA_PAGED_VERIFY_SELFTEST")', t)
        self.assertIn("== 4;", t, "SELFTEST=4 is the stage-1 red arm")
        self.assertIn("ggml_cl_fa_paged_verify4_selftest(backend_ctx, paged_overlap_parity);", t)
        # the stage-1 write sits between the pre hash and the reader: after the AT=splits spin
        # and before k_split
        i = t.find("ggml_cl_fa_paged_verify4_selftest(backend_ctx, paged_overlap_parity);")
        j = t.find("backend_ctx->enqueue_ndrange_kernel(k_split, 3, fd_gws, fd_lws, dst, tl_now ? &ev_split : nullptr);")
        self.assertGreater(j, i, "0131d: the self-test write must precede the split kernel")

    def test_verify4_collects_only_at_the_request_end(self):
        t = self.t
        self.assertIn("if (ggml_cl_fa_paged_verify_mode() == 3 && g_fa_paged_verify3_ctx)", t,
                      "0131d: a VERIFY=4 pair must not be waited at a graph's start")
        self.assertNotIn("verify_mode() == 3 || ggml_cl_fa_paged_verify_mode() == 4 && g_fa_paged_verify3_ctx", t)


if __name__ == "__main__":
    unittest.main()
