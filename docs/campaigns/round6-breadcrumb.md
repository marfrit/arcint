# Round 6 breadcrumb (kv-i5, private working note; delete when merged)

HEAD 7125ada8. Recipe EXIT 0 (44 tests). Patches 0130p/q/r/s/t on 82c7d449.

- 0130p: TIMELINE (ggml-opencl.cpp): tl helpers before overlap_graph, markers carry the
  next command's waits; copy queue profiling only when switch on; finish at request end;
  margin `copy(k+2).start - FA(k).split end`, cross-step prefetch start vs last FA split end.
- 0130q: NOFIX drops the prefetch's consumed wait (prefetch_head); startup lines for
  NOCONSUME/NOFIX/DELAY/VERIFY/SELFTEST/TIMELINE; request-end dev-switch counts line.
- 0130r: tests/python/test_fa_paged_lead.py `FaPagedTimelineAndNoFix`; three mutations
  seen failing (split start stamp after merge; NOFIX added to issue; summary line renamed).
  Older cells updated: noconsume read-once line, roll-up gate with timeline, prefetch line.
- 0130s: design note entry in docs/campaigns/kv-streaming-design.md.
- 0130t: README entries 0130o..0130r.
- Key code answer: VERIFY=3 collect (verify3_collect at graph start, clWaitForEvents on both
  readbacks) serializes host lead -> the cross-step prefetch is ordered behind the last FA,
  so window 296's NOFIX red cannot land. Use TIMELINE margin (no VERIFY) or a second-queue
  post hash (not built).
- Card commands are in the intercom message to arcint (8:57 CET). Not done: card runs.
- Patched tree used for LLAMA_SRC: /tmp/fa6/llama.cpp-3e4dfa8f39f904e681737d9f512a6e6338d848aa
