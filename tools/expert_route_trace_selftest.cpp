// Writes a synthetic ERT1 routing trace with the serializer patch 0069 adds
// (src/llama-expert-route-trace.{h,cpp} of the llama tree the build pins), for
// the device-free round trip: tests/python/test_expert_route_trace.py compiles
// this against that tree and reads the bytes back with
// tools/expert_cache_replay.py. The committed fixture
// tests/python/fixtures/expert-route-trace-v2.ert1 is this program's output,
// so the round trip also runs without the tree (the test then compares the
// fixture with the replay's own writer, byte for byte).
//
// It also checks the graph-kind decision: a prompt's 4-token checkpoint tail is
// a PROMPT record, not a VERIFY, and an embedding batch (no token ids) is a
// PROMPT. Red mutation: drop `!all_outputs` from the decision -- the tail
// becomes VERIFY and the run below exits 3.
//
// Usage: expert_route_trace_selftest <out-path>
// The llama tree's src/ directory comes from EXPERT_ROUTE_TRACE_SRC (the test
// passes it); nothing here names a host or a path.
//
// Two more cells, round 7, driven by env so the fixture stays this program's
// plain output:
//  EXPERT_ROUTE_SELFTEST_WRITE_FAIL=1     close the trace's fd, then write: one
//      WARN, tracing stops (opened 0, no sidecar), later records are no-ops.
//      Red: drop `opened = false;` in the writer's error path -- the program
//      prints `write-fail opened 1`.
//  EXPERT_ROUTE_SELFTEST_PENDING_ORDER=1  a verify's real order: after_graph
//      prepares (no rotation yet), the accept line is written, the MTP DRAFT
//      records push the cap over, advance writes the pending record and must
//      not rotate -- so the line and the record stay in one file. The next
//      after_graph rotates. EXPERT_ROUTE_SELFTEST_ROTATE_ON_PENDING=1 is the
//      red arm: rotate at advance, the record lands in the new file and the old
//      one keeps the line with no graph (the replay's unwritten last graph).
//  EXPERT_ROUTE_SELFTEST_CROSS_CACHE=1  a layer split: two caches' records for
//      the same graph are pending; neither cache's prepare rotates until BOTH
//      writes have landed, and the rotation's LAYERS record carries every
//      cache's residents. Red mutation (seen failing): `rotate_now` tests this
//      cache's flag instead of the process-wide pending count -- the first
//      prepare rotates while the other cache's record waits.
#include "llama-expert-route-trace.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <out-path>\n", argv[0]);
        return 2;
    }
    // the decision the cache uses: prompt tail 4 tokens in a prompt call
    if (llama_expert_route_graph_kind(4, 8, false, true) != LLAMA_EXPERT_ROUTE_PROMPT) {
        std::fprintf(stderr, "a 4-token prompt tail classified %d, not PROMPT\n",
                     (int) llama_expert_route_graph_kind(4, 8, false, true));
        return 3;
    }
    if (llama_expert_route_graph_kind(4, 8, true, true) != LLAMA_EXPERT_ROUTE_VERIFY)
        return 4;
    if (llama_expert_route_graph_kind(1, 8, true, true) != LLAMA_EXPERT_ROUTE_DECODE)
        return 5;
    if (llama_expert_route_graph_kind(4, 8, true, false) != LLAMA_EXPERT_ROUTE_PROMPT)
        return 6;   // an embedding batch has no token ids

    // the process's one writer, as the cache uses it: the accept sidecar is
    // named from this instance. A missing out path is a DIR and is created;
    // an existing non-directory is a file base. No append either way.
    auto & t = llama_expert_route_trace::instance();
    if (!t.open(argv[1], 6, 32, 4)) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }
    static int ord = 0;
    // one line a decode graph, as arcint writes it beside the current file
    auto accept = [&](int pos, int rows, int acc) -> bool {
        const std::string name = llama_expert_route_trace_sidecar();
        if (name.empty()) {
            return false;
        }
        std::FILE * f = std::fopen(name.c_str(), "a");
        if (f == nullptr) {
            return false;
        }
        ++ord;
        std::fprintf(f, "verify %d pos %d rows %d accepted %d\n", ord, pos, rows, acc);
        std::fflush(f);
        std::fclose(f);
        return true;
    };
    const bool line_before = std::getenv("EXPERT_ROUTE_SELFTEST_LINE_BEFORE_ROTATE") != nullptr;
    const bool force_rotate = std::getenv("EXPERT_ROUTE_SELFTEST_ROTATE_AT_LINE") != nullptr;
    // two layers, five slots each, the last slot the zeros slot (-1)
    std::vector<int> il = {2, 5};
    std::vector<int> ns = {5, 5};
    std::vector<std::vector<int32_t>> res = {{0, 3, 7, 11, -1}, {1, 2, 3, 4, -1}};
    // as the cache does it: at the cap, open the next file (its own header)
    // and write the residents that file starts from. With the default cap this
    // never runs, so the fixture below is unchanged.
    auto emit = [&](auto fn) {
        if (t.at_cap()) {
            if (!t.rotate_next()) {
                std::fprintf(stderr, "rotation failed\n");
                std::exit(7);
            }
            t.event_layers(il, ns, res);
        }
        fn();
    };
    // the production order of a rotation at the target's after_graph: the
    // rotation and the new file's LAYERS come BEFORE the accept line, then the
    // graph record lands in that same file. The red mutation writes the line
    // first, so a rotation splits line and graph. `force_rotate` makes the test
    // independent of where the byte cap falls.
    auto emit_line = [&](auto fn, int pos, int rows, int acc) {
        if (force_rotate) {
            if (line_before) accept(pos, rows, acc);
            if (!t.rotate_next()) {
                std::fprintf(stderr, "rotation failed\n");
                std::exit(7);
            }
            t.event_layers(il, ns, res);
            fn();
            if (!line_before) accept(pos, rows, acc);
            return;
        }
        if (line_before) accept(pos, rows, acc);
        emit(fn);
        if (!line_before) accept(pos, rows, acc);
    };
    emit([&] { t.event_layers(il, ns, res); });

    // The cache's rotation step on a graph record: the pure decision, then the
    // next file and this file's LAYERS snapshot -- exactly the three lines
    // llama-expert-cache.cpp's trace_prepare runs.
    auto prepare_rotation = [&]() -> bool {
        if (!llama_expert_route_trace_rotate_now(t)) {
            return false;
        }
        if (!t.rotate_next()) {
            std::fprintf(stderr, "rotation failed\n");
            std::exit(7);
        }
        t.event_layers_all();
        return true;
    };

    // a write failure: the tracing must stop with one WARN, not fail the run and
    // not write again (a closed fd is the cheap, deterministic form of a full
    // disk; the writer's rule is the same for either errno)
    if (std::getenv("EXPERT_ROUTE_SELFTEST_WRITE_FAIL") != nullptr) {
        ::close(t.fd);
        t.event_layers(il, ns, res);   // the write fails here: one WARN, opened false
        t.event_layers(il, ns, res);   // no second WARN, nothing written
        const bool opened  = t.opened;
        const bool sidecar = !llama_expert_route_trace_sidecar().empty();
        t.close();
        std::printf("write-fail opened %d sidecar %d\n", opened ? 1 : 0, sidecar ? 1 : 0);
        return 0;
    }

    // a verify's real order at a rotation: the DRAFT records of the MTP block
    // arrive between after_graph (which took the rotation) and the advance that
    // writes the verify's record. The line is already in this file, so advance
    // must not rotate. The cap (0.00014 MB, ~146 bytes) is crossed by the third
    // DRAFT record, after the line.
    if (std::getenv("EXPERT_ROUTE_SELFTEST_PENDING_ORDER") != nullptr) {
        t.set_residents(1, il, ns, res);   // the block below writes a full snapshot
        prepare_rotation();   // after_graph: cap not reached, nothing pending
        t.pending_graph(true);   // the record now waits for advance
        accept(200, 3, 1);         // arcint writes the verify's line here
        std::vector<std::vector<int32_t>> dids = {{1, 2, 3, 4, 5, 6, 7, 8}};
        for (int d = 0; d < 4; ++d) {
            t.event_graph(LLAMA_EXPERT_ROUTE_DRAFT, 300 + d, 2, std::vector<int32_t>{4000, 4001},
                          std::vector<int>{6}, dids);   // no prepare: a DRAFT never rotates
        }
        const bool red = std::getenv("EXPERT_ROUTE_SELFTEST_ROTATE_ON_PENDING") != nullptr;
        if (red) {
            // the round-6/7 defect: prepare at advance, ignoring the pending count
            if (t.at_cap()) {
                if (!t.rotate_next()) {
                    std::fprintf(stderr, "rotation failed\n");
                    std::exit(7);
                }
                t.event_layers_all();
            }
        } else {
            prepare_rotation();    // advance: blocked by the process-wide pending count
        }
        t.event_graph(LLAMA_EXPERT_ROUTE_VERIFY, 200, 3, std::vector<int32_t>{2000, 2001, 2002},
                      il, std::vector<std::vector<int32_t>>{{5, 6, 7, 8, 9, 10}, {11, 12, 13, 14, 15, 16}});
        t.pending_graph(false);   // advance wrote the record
        prepare_rotation();   // the next after_graph rotates
        t.close();
        return 0;
    }

    // A layer split: two caches hold a range of layers, each writes its record
    // for the same graph, and neither cache's prepare may rotate until both
    // records are written. The rotation then writes one LAYERS record over
    // both cards' residents, so the new file replays alone.
    if (std::getenv("EXPERT_ROUTE_SELFTEST_CROSS_CACHE") != nullptr) {
        std::vector<int> il_b = {7, 9};
        std::vector<int> ns_b = {5, 5};
        std::vector<std::vector<int32_t>> res_b = {{8, 9, 10, 11, -1}, {12, 13, 14, 15, -1}};
        t.set_residents(1, il, ns, res);
        t.set_residents(2, il_b, ns_b, res_b);
        t.event_layers_all();   // the first file's own snapshot: all four layers

        prepare_rotation();     // cache A's after_graph: under the cap
        t.pending_graph(true);  // A's record waits for its advance
        prepare_rotation();     // cache B's after_graph: A is pending, no rotation
        if (!t.opened || t.seq != 0) {
            std::fprintf(stderr, "cache B rotated away from A's pending record\n");
            return 8;
        }
        t.pending_graph(true);  // B's record waits too
        // B's advance writes its record, A is still pending: still no rotation
        t.event_graph(LLAMA_EXPERT_ROUTE_VERIFY, 200, 3, std::vector<int32_t>{2000, 2001, 2002},
                      il_b, std::vector<std::vector<int32_t>>{{8, 9, 10, 11, 12, 13}, {12, 13, 14, 15, 16, 17}});
        t.pending_graph(false);
        prepare_rotation();
        if (!t.opened || t.seq != 0) {
            std::fprintf(stderr, "cache B rotated while A's record was pending\n");
            return 9;
        }
        // A's advance writes its record: the next prepare rotates
        t.event_graph(LLAMA_EXPERT_ROUTE_VERIFY, 200, 3, std::vector<int32_t>{2000, 2001, 2002},
                      il, std::vector<std::vector<int32_t>>{{5, 6, 7, 8, 9, 10}, {11, 12, 13, 14, 15, 16}});
        t.pending_graph(false);
        prepare_rotation();
        if (!t.opened || t.seq != 1) {
            std::fprintf(stderr, "the rotation did not happen after both records\n");
            return 10;
        }
        t.close();
        return 0;
    }
    // a prompt batch at position 0: 3 tokens, top-2 a token (rank order kept)
    emit([&] { t.event_graph(LLAMA_EXPERT_ROUTE_PROMPT, 0, 3, std::vector<int32_t>{100, 101, 102},
                  il, std::vector<std::vector<int32_t>>{{0, 3, 7, 11, 2, 9}, {1, 2, 3, 4, 5, 6}}); });
    // a prompt's 4-token checkpoint tail at position 10: PROMPT, not VERIFY
    emit([&] { t.event_graph(LLAMA_EXPERT_ROUTE_PROMPT, 10, 4, std::vector<int32_t>{103, 104, 105, 106},
                  il, std::vector<std::vector<int32_t>>{{0, 0, 0, 0, 1, 1, 1, 1},
                                                        {2, 2, 2, 2, 3, 3, 3, 3}}); });
    // a verify at position 200: 3 tokens (two drafts), the same layers
    emit_line([&] { t.event_graph(LLAMA_EXPERT_ROUTE_VERIFY, 200, 3, std::vector<int32_t>{2000, 2001, 2002},
                  il, std::vector<std::vector<int32_t>>{{5, 6, 7, 8, 9, 10}, {11, 12, 13, 14, 15, 16}}); },
              200, 3, 1);
    // a table change: layer 2, expert 5 into a slot, expert 0 out; layer 5, expert 11 in
    std::vector<llama_expert_route_trace::change> ch = {{2, 5, 1}, {2, 0, 0}, {5, 11, 1}};
    res[0][0] = 5; res[1][4] = 11;   // what the cache's slots hold after the swaps
    emit([&] { t.event_tables(ch); });
    // a decode at position 203: 1 token. EXPERT_ROUTE_SELFTEST_SKIP_LAST omits
    // it: the red case of "the last graph is written at the request's end" --
    // the accept sidecar then has a line with no graph.
    if (std::getenv("EXPERT_ROUTE_SELFTEST_SKIP_LAST") == nullptr) {
        emit_line([&] { t.event_graph(LLAMA_EXPERT_ROUTE_DECODE, 203, 1, std::vector<int32_t>{3000},
                      il, std::vector<std::vector<int32_t>>{{5, 6}, {11, 12}}); }, 203, 1, 0);
    } else {
        // the line exists, the graph does not: the process's unwritten last
        // graph (the red case of the request-end flush)
        accept(203, 1, 0);
    }
    // an MTP draft context's own decode: INPUT tokens (step 0's accepted target
    // token, then the previous draft), its one MoE block (il = n_layer = 6, not
    // one of the cache's layers), top-4 a token
    emit([&] { t.event_graph(LLAMA_EXPERT_ROUTE_DRAFT, 204, 2, std::vector<int32_t>{4000, 4001},
                  std::vector<int>{6}, std::vector<std::vector<int32_t>>{{1, 2, 3, 4, 5, 6, 7, 8}}); });
    t.close();
    return 0;
}
