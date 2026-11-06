#pragma once

// An opt-in, binary routing trace for the expert cache (campaign
// expert-foresight; stage 0 built it as a diagnostic, the operator made it a
// production option on 2026-10-29). `LLAMA_EXPERT_ROUTE_TRACE=<dir or path>`
// (arcint's `--llama-expert-route-trace DIR`) turns it on; unset it costs one
// getenv at start-up and nothing else: no allocation, no read-back, no write.
//
// A file per process, rotated at a size cap: `f = dir` gives
// `expert-route-<pid>-<unix-time>.000.ert1`, `.001.ert1`, ...; `f` an existing
// non-directory is a file BASE, not the first file: a file that already sits
// there is skipped (O_EXCL) and the sequence starts at `f.001`, then `.002`.
// Every file is
// opened with O_EXCL and a fresh name is chosen, never appended: a restarted
// unit with a reused pid, or a path whose file already exists, gets a new base
// or the next sequence, so no second header ever lands in an old file. A DIR
// that does not exist is created; if that fails the trace is refused with a
// WARN, never treated as a file path. Each file carries its own header and,
// after a rotation, a LAYERS record of the cache's residents at that moment, so
// every file replays alone; nothing is ever deleted by arcint. Header with the
// model's geometry, then one record a event, little endian, no trailer. A run
// that dies mid-write leaves a torn last record; a reader stops at it (never a
// partial one), and resynchronises on a later "ERT1" magic so an old file with
// an appended header still replays.
//
// Size: a prompt record is n_tokens x top-k x n_layer x 4 B, about 1.9 KB a
// prompt token, so a 100k prompt is about 190 MB. DIR belongs on a dataset with
// room, not the root filesystem.
//
// A trace with token ids is the prompt and the answer in another form: whoever
// turns it on stores every request's content on the serving host.
//
//   header  "ERT1" u32 version u32 n_layer u32 n_expert u32 n_used
//           version 2 adds a graph's first position; version 1 had none
//   event   u8 kind, then
//     0 LAYERS  u32 n, per: i32 il, i32 n_slots, n_slots i32 resident (slot ->
//               expert, -1 for the trailing zeros slot) -- a cache's layers as
//               loaded, so a replay starts from the run's own resident set
//     1 PROMPT  a prompt batch of any size, including the 4-token tail a
//               checkpoint break leaves (the cache classifies by the call:
//               a prompt call does not output every row; a decode call does)
//     2 DECODE  a one-token decode
//     3 VERIFY  a decode verify (2..decode_max tokens): tokens[1..] are the
//               drafts, tokens[0] the token before them; which of the drafts a
//               sampler accepted is NOT in this file -- the engine's cache
//               cannot see the sampler. arcint writes it beside the trace
//               (`<path>.accept`: "verify <ordinal> pos <p> rows <n> accepted
//               <a>", one a decode graph, a single lane; the replay aligns by
//               p, a graph's first position, never by order)
//     4 DRAFT   an MTP draft context's own decode: the MTP block's input
//               tokens (step 0 the accepted target token, then the previous
//               draft -- NOT the drafted tokens) and its own routed ids
//               (Flash-Next's MTP block is a full MoE layer,
//               llama-model.cpp/models/qwen4exp.cpp; it runs without an
//               expert cache, so llama-context.cpp reads its
//               selected_experts tensor when this trace is on)
//     all four graph kinds: i32 first_pos (version 2; -1 when the ubatch has
//      no positions), u32 n_tokens, i32 token[n_tokens] (-1 for an embedding
//      row), u32 n_layers, per: i32 il, u32 count, i32 ids[count]
//      (token-major, the router's top-k in rank order, exactly the ids_out the
//      cache already reads)
//     5 TABLES  u32 n_changes, per: i32 il, i32 expert, u8 resident (1: in a
//               slot, 0: not) -- every residency change the cache applies, so a
//               replay rebuilds the resident set instead of guessing it
//
// The writer is dependency-free on purpose: a device-free test compiles this
// file alone and reads its bytes back with the replay's reader.

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

// ERT1 graph kinds: the cache's decision from a call's shape, kept pure so a
// device-free test can exercise it (a prompt call outputs fewer rows than it
// has tokens; a decode or verify call outputs every one; a prompt's 4-token
// checkpoint tail is a prompt).
enum llama_expert_route_kind : uint8_t {
    LLAMA_EXPERT_ROUTE_LAYERS = 0, LLAMA_EXPERT_ROUTE_PROMPT = 1,
    LLAMA_EXPERT_ROUTE_DECODE = 2, LLAMA_EXPERT_ROUTE_VERIFY = 3,
    LLAMA_EXPERT_ROUTE_DRAFT = 4, LLAMA_EXPERT_ROUTE_TABLES = 5,
};

uint8_t llama_expert_route_graph_kind(uint32_t n_tokens, uint32_t decode_max, bool all_outputs, bool has_tokens);

// The variable is read once: the graph builder asks whether to mark the
// MTP block's routed ids as an output (a no-op when the trace is off).
bool llama_expert_route_trace_enabled();

// The accept sidecar beside the CURRENT trace file (`<file>.accept`), so it
// rotates with it; empty when the trace is off. A writer appends one line a
// accepted graph and reopens the sidecar when this name changes.
std::string llama_expert_route_trace_sidecar();

struct llama_expert_route_trace {
    struct change { int32_t il, expert; uint8_t resident; };
    int  fd      = -1;
    bool opened  = false;
    bool warned  = false;       // one WARN a run when a write or an open fails
    std::string path;           // `base` for the file names below
    bool        dir_mode = false;
    uint32_t    seq      = 0;   // the file in this process's sequence
    uint64_t    bytes_in_file = 0;
    uint64_t    max_bytes = 0;  // rotate past this; LLAMA_EXPERT_ROUTE_TRACE_MAX_MB (default 256)
    uint32_t    n_layer = 0, n_expert = 0, n_used = 0;
    std::vector<uint8_t> buf;   // the record being built
    // the writer is shared by the cache's decode path and an MTP context's
    // synchronize (lanes), so every public record call holds this one lock
    std::recursive_mutex mu;
    // process-wide, not one cache: with a cache a card (a layer split), cache A
    // must not rotate while cache B's graph record for the same graph is still
    // pending -- its accept line is already in A's file
    int pending_graphs = 0;
    // every cache's residents at its last TABLES record, so a rotation writes
    // one LAYERS record that holds every card's layers and every file replays
    // alone under a layer split
    struct layer_set {
        intptr_t key;   // the cache, stable for its life
        std::vector<int> il, n_slots;
        std::vector<std::vector<int32_t>> residents;
    };
    std::vector<layer_set> layer_sets;

    // Opens the first file with O_EXCL (never appends), writes the header.
    bool open(const char * p, uint32_t n_layer, uint32_t n_expert, uint32_t n_used);
    // At the cap: close this file, open the next fresh one (O_EXCL) and write
    // its header. The caller follows with event_layers_all.
    bool at_cap() const { return opened && bytes_in_file >= max_bytes; }
    bool rotate_next();
    void close();

    void put_u8(uint8_t v)  { buf.push_back(v); }
    void put_u32(uint32_t v);
    void put_i32(int32_t v);
    void flush();   // one write() a record, resumed across short writes; the
                    // file is opened O_WRONLY|O_EXCL (no O_APPEND)

    void event_layers(const std::vector<int> & il, const std::vector<int> & n_slots,
                      const std::vector<std::vector<int32_t>> & residents);
    // A cache's current residents, kept for the next rotation's snapshot.
    void set_residents(intptr_t key, const std::vector<int> & il, const std::vector<int> & n_slots,
                       const std::vector<std::vector<int32_t>> & residents);
    // One LAYERS record over every registered cache, in registration order.
    void event_layers_all();
    // Hold/release the process-wide pending count around a graph record.
    void pending_graph(bool pending);
    void event_graph(uint8_t kind, int32_t first_pos, uint32_t n_tokens, const std::vector<int32_t> & tokens,
                     const std::vector<int> & il, const std::vector<std::vector<int32_t>> & ids);
    void event_tables(const std::vector<change> & changed);

    // The process's one trace: opened on the first cache's init, off when the
    // variable is unset or the file cannot be opened (a warning a run).
    static llama_expert_route_trace & instance();
};

// The cache's rotation decision, kept pure so a device-free test can drive the
// cache's real call order: a rotation is taken at `after_graph`, before arcint
// writes the verify's accept line. While ANY cache's graph record is pending
// its accept line is already in this file, so a rotation at the next advance,
// the request-end flush or an MTP DRAFT record would split the line from its
// record; it waits for the next `after_graph` (round 7's defect, made
// process-wide in round 8: a layer split has a cache a card).
bool llama_expert_route_trace_rotate_now(const llama_expert_route_trace & t);
