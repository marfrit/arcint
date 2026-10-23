// clean-turns step 2: the tool-call grammar (core/toolgrammar.h) and the
// sampler's constraint hook (core/sampler.h TokenConstraint), device-free.
//
// Three layers:
//   - the generator's output names what was declared (every build);
//   - the sampler drives a constraint as the reference's common_sampler_sample
//     does, including in the verify walk, where a rejected draft must never
//     reach it (every build, a synthetic constraint);
//   - the GBNF accepts the well-formed call and refuses the malformed ones,
//     run through libllama's own grammar engine on text (llama-grammar.h's
//     test entry points, with no vocabulary -- the way upstream's
//     tests/test-grammar-integration.cpp drives it). Only in a libllama build.
//
// No call tag is written out here: every one comes from tool_call_tags(), the
// table the parser reads.
#include "api/handlers.h"
#include "core/sampler.h"
#include "core/toolcall.h"
#include "core/toolgrammar.h"
#include "exec/backend.h"
#include "exec/verify_walk.h"
#include "harness.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef ARCINT_LLAMA
#include "llama-grammar.h"
#endif

using namespace lgc;

namespace {

std::vector<ToolSpec> two_tools() {
    ToolSpec weather;
    weather.name       = "get_weather";
    weather.parameters = nlohmann::json::parse(R"({"type":"object","properties":{
        "city":{"type":"string"},
        "units":{"type":"string","enum":["celsius","fahrenheit"]}},
        "required":["city"]})");
    ToolSpec query;
    query.name       = "run_query";
    query.parameters = nlohmann::json::parse(R"({"type":"object","properties":{
        "sql":{"type":"string"},
        "limit":{"type":"integer"},
        "meta":{"type":"object"}},
        "required":["sql","limit"]})");
    return { weather, query };
}

std::string s(std::string_view v) { return std::string(v); }

// One call in the coder's form, from the tag table: name, then (param, value)
// pairs in the order given.
[[maybe_unused]] std::string call(const std::string& name, const std::vector<std::pair<std::string, std::string>>& args) {
    const ToolCallTags& t   = tool_call_tags();
    std::string         out = s(t.open) + "\n" + s(t.fn_open) + name + ">\n";
    for (const auto& [k, v] : args) out += s(t.param_open) + k + ">\n" + v + "\n" + s(t.param_end) + "\n";
    return out + s(t.fn_close) + "\n" + s(t.close);
}

}  // namespace

TEST(toolgrammar_no_tools_no_grammar) {
    CHECK(generate_tool_grammar({}).empty());
    CHECK_EQ(tool_grammar_trigger(), s(tool_call_tags().open));
}

TEST(toolgrammar_names_the_declared_set) {
    const std::string g = generate_tool_grammar(two_tools());
    CHECK(!g.empty());
    for (const char* want : { "get_weather", "run_query", "city", "units", "sql", "limit", "meta", "root" })
        CHECK(g.find(want) != std::string::npos);
    // an undeclared name is nowhere in it
    CHECK(g.find("delete_everything") == std::string::npos);
}

// A schema the call form cannot constrain is the client's error, never a cast
// of the wrong node (the reference casts the root unchecked).
TEST(toolgrammar_refuses_a_non_object_schema) {
    ToolSpec bad;
    bad.name       = "f";
    bad.parameters = nlohmann::json::parse(R"({"type":"string"})");
    bool threw     = false;
    try {
        (void)generate_tool_grammar({ bad });
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    ToolSpec none;   // no parameters at all: an object with none, not an error
    none.name = "g";
    CHECK(!generate_tool_grammar({ none }).empty());
}

// ------------------------------------------------------- the sampler's hook

namespace {

constexpr size_t kVocab = 16;

// Allows the tokens in `ok` (all when empty); records every accept and every
// full-row mask.
struct FakeConstraint : TokenConstraint {
    std::set<int>    ok;
    std::vector<int> accepted;
    int              masks = 0;
    int              asked = 0;
    bool allows(int tok) override {
        ++asked;
        return ok.empty() || ok.count(tok) != 0;
    }
    void mask(float* logits, size_t vocab) override {
        ++masks;
        for (size_t i = 0; i < vocab; ++i)
            if (!allows(static_cast<int>(i))) logits[i] = -INFINITY;
    }
    void accept(int tok) override { accepted.push_back(tok); }
};

std::vector<float> row_with_argmax(int top, int second = -1) {
    std::vector<float> r(kVocab, 0.0f);
    for (size_t v = 0; v < kVocab; ++v) r[v] = 0.01f * static_cast<float>(v);
    r[static_cast<size_t>(top)] = 5.0f;
    if (second >= 0) r[static_cast<size_t>(second)] = 3.0f;
    return r;
}

SamplerParams greedy_grammar() {
    SamplerParams p;
    p.temperature        = 0.0f;
    p.repetition_penalty = 1.0f;
    p.tool_grammar       = true;
    return p;
}

// Review B4, driven synthetically: a constraint that loses sync with the
// generated text. Healthy it admits only the tokens in `ok` (a refusal is a
// mask-and-redraw, as the real grammar does); the first token it never admits
// through accept() breaks it, and it counts the desync on the base class the
// way the real one does (note_desync). Broken it fails open -- allows
// everything, no engine call -- so the turn can still end, and the count
// latches, so it does not run away per token. `mask_throws` simulates the
// engine itself throwing inside a full-row mask (the guard's other entry).
struct BrokenConstraint : TokenConstraint {
    std::set<int> ok{ 7, 9 };
    bool          broken      = false;
    int           masks       = 0;
    bool          mask_throws = false;
    bool allows(int tok) override {
        if (broken) return true;
        return ok.count(tok) != 0;
    }
    void mask(float* logits, size_t vocab) override {
        ++masks;
        if (mask_throws) throw std::runtime_error("synthetic: the engine threw in the mask");
        if (broken) return;   // broken: fail open, nothing to mask
        for (size_t i = 0; i < vocab; ++i)
            if (!allows(static_cast<int>(i))) logits[i] = -INFINITY;
    }
    void accept(int tok) override {
        if (broken) return;
        if (!allows(tok)) {
            broken     = true;
            note_desync(tok, "synthetic: the token left the constraint");
        }
    }
};

}  // namespace

// The red case the gate asks for, at the sampler: the row prefers a token the
// constraint refuses. Without the constraint that token is drawn; with it the
// draw is masked and redrawn among the allowed ones.
TEST(sampler_constraint_redraws_a_refused_token) {
    {
        Sampler            plain(greedy_grammar(), 1);
        std::vector<float> r = row_with_argmax(5, 7);
        CHECK_EQ(plain.sample(r.data(), kVocab), 5);   // control: the refused token wins
    }
    Sampler        sampler(greedy_grammar(), 1);
    FakeConstraint c;
    c.ok = { 7, 9 };
    sampler.set_constraint(&c);
    std::vector<float> r = row_with_argmax(5, 7);
    CHECK_EQ(sampler.sample(r.data(), kVocab), 7);
    CHECK_EQ(c.masks, 1);
    // a token that fits costs no mask
    std::vector<float> r2 = row_with_argmax(9);
    CHECK_EQ(sampler.sample(r2.data(), kVocab), 9);
    CHECK_EQ(c.masks, 1);
}

TEST(sampler_constraint_redraws_when_sampling) {
    SamplerParams p = greedy_grammar();
    p.temperature   = 0.8f;
    for (uint64_t seed = 1; seed <= 20; ++seed) {
        Sampler        sampler(p, seed);
        FakeConstraint c;
        c.ok = { 3, 11 };
        sampler.set_constraint(&c);
        std::vector<float> r = row_with_argmax(5, 7);
        const int          t = sampler.sample(r.data(), kVocab);
        CHECK(t == 3 || t == 11);
    }
}

TEST(sampler_constraint_accepts_every_observed_token) {
    Sampler        sampler(greedy_grammar(), 1);
    FakeConstraint c;
    sampler.set_constraint(&c);
    sampler.observe(4);
    sampler.observe(6);
    CHECK(c.accepted == std::vector<int>({ 4, 6 }));
}

// Review B4: a constraint that loses sync with the generated text fails open
// -- the rest of the request runs unconstrained, so the turn can still end --
// and the desync is counted on the constraint, so the backend can surface it
// in the per-request stats line instead of leaving it to a log line nobody
// reads. The synthetic constraint latches on the same path the real one
// latches (a token the constraint never admits), and this case asserts the
// three promises: the counter increments, allows() stays safe afterwards, and
// generation is not masked.
TEST(sampler_constraint_desync_counts_and_fails_open) {
    Sampler        sampler(greedy_grammar(), 1);
    BrokenConstraint c;
    sampler.set_constraint(&c);
    // Healthy first: a draw the constraint refuses is masked and redrawn,
    // and the desync counter is still zero.
    std::vector<float> r = row_with_argmax(5, 7);
    CHECK_EQ(sampler.sample(r.data(), kVocab), 7);
    CHECK_EQ(c.masks, 1);
    CHECK_EQ(c.desync(), 0);
    // A token the constraint never admits arrives through observe (the path
    // that bypasses the draw's allows() test): it loses the sync.
    sampler.observe(3);
    CHECK(c.broken);
    CHECK_EQ(c.desync(), 1);
    // From here on allows() is safe and fail-open: every token allowed, no
    // engine call, no throw.
    CHECK(c.allows(0));
    CHECK(c.allows(15));
    // Generation is not masked any more: the row's own argmax wins, even
    // though the constraint never admitted it.
    std::vector<float> r2 = row_with_argmax(5);
    CHECK_EQ(sampler.sample(r2.data(), kVocab), 5);
    CHECK_EQ(c.masks, 1);                // no new mask after the desync
    // ...and the rest of the request flows unconstrained, with the desync
    // counted once, not once per token.
    std::vector<float> r3 = row_with_argmax(12);
    CHECK_EQ(sampler.sample(r3.data(), kVocab), 12);
    CHECK_EQ(c.desync(), 1);
}

// Review B4 follow-up: the refusal can also surface as a THROW inside the
// full-row mask, on the path sample() takes right after a refused draw.
// Without a guard there it escapes sample() as a 500 mid-stream. With the
// guard the desync is counted once, and generation continues UNCONSTRAINED
// (fail open): the row was not masked (the throw came first), so the redraw
// draws the row's own argmax. The sampler detaches the broken constraint, so
// the rest of the request never asks it again.
TEST(sampler_constraint_mask_throw_is_guarded) {
    Sampler            sampler(greedy_grammar(), 1);
    BrokenConstraint   c;
    c.mask_throws = true;
    sampler.set_constraint(&c);
    std::vector<float> r = row_with_argmax(5, 7);   // 5 refused by the constraint
    bool              threw = false;
    int               got   = -1;
    try {
        got = sampler.sample(r.data(), kVocab);
    } catch (const std::exception&) {
        threw = true;   // before the guard: this throw escapes sample()
    }
    CHECK(!threw);
    CHECK_EQ(c.desync(), 1);    // counted exactly once, like every other desync
    CHECK_EQ(c.masks, 1);
    CHECK_EQ(got, 5);           // fail open: the unmasked row's own argmax wins
    // and it stays that way: a further step runs unconstrained, no new throw,
    // no new count, no second mask call (the constraint is detached).
    std::vector<float> r2 = row_with_argmax(12);
    CHECK_EQ(sampler.sample(r2.data(), kVocab), 12);
    CHECK_EQ(c.desync(), 1);
    CHECK_EQ(c.masks, 1);
}

// Review B4 follow-up: the count is the latch's, not the caller's. Before
// the fix every path that lost sync incremented the counter, and the once
// came from the callers remembering to check their own broken flag -- a
// second recording in the same request double-counted the stats line. The
// constraint below refuses every token it is handed and records every
// refusal WITHOUT checking anything itself: it simulates a second path that
// bypasses the caller-side latch, so the second note must be the base
// class's to swallow.
struct RefusingConstraint : TokenConstraint {
    bool allows(int) override { return false; }
    void mask(float*, size_t) override {}
    void accept(int tok) override { note_desync(tok, "synthetic: the engine refuses everything"); }
};

TEST(token_constraint_note_desync_latches_the_count) {
    RefusingConstraint c;
    c.accept(3);
    CHECK_EQ(c.desync(), 1);
    c.accept(4);   // a second loss of sync in the same request
    CHECK_EQ(c.desync(), 1);   // without the latch in the helper this is 2
}

// Review B4 follow-up: the stats wiring at the struct level. The count that
// reaches the server's decode line is decode_suffix_for's -- this is the
// value that lands on the line, not a grep of logs: a desynced request prints
// the count, and an undescynced one keeps the line it always had.
TEST(decode_suffix_carries_the_grammar_desync_count) {
    GenerationStats s;
    s.grammar_desyncs = 3;
    const std::string suffix = lgc::api::decode_suffix_for(s);
    CHECK(suffix.find("grammar desync 3") != std::string::npos);
    GenerationStats plain;   // zero: the term stays off the line
    CHECK(lgc::api::decode_suffix_for(plain).find("grammar desync") == std::string::npos);
}

// Dormant while a think block is open: arcint parses calls from the content
// only, so a call opener mentioned while thinking must not trigger it.
TEST(sampler_constraint_dormant_in_think_block) {
    SamplerParams p          = greedy_grammar();
    p.think_ids.open_tags    = { "<T>" };
    p.think_ids.close_tags   = { "</T>" };
    p.think_ids.eog          = { 5 };   // the turn-ender, for the mask's sake
    Sampler        sampler(p, 1);
    FakeConstraint c;
    c.ok = { 7 };
    sampler.set_constraint(&c);
    sampler.start_in_think_block(true);
    CHECK(sampler.think_block_open());
    std::vector<float> r = row_with_argmax(5);
    // not consulted -- and tracking the block for the grammar does not switch
    // the think_eog_mask on: the turn-ender 5 is still drawn inside the block
    CHECK_EQ(sampler.sample(r.data(), kVocab), 5);
    CHECK_EQ(c.asked, 0);
    sampler.observe_think_text("thinking");
    sampler.observe(5);
    CHECK(c.accepted.empty());
    sampler.observe_think_text("</T>");               // the block closes
    sampler.observe(6);
    CHECK(c.accepted == std::vector<int>({ 6 }));
    std::vector<float> r2 = row_with_argmax(5);
    CHECK_EQ(sampler.sample(r2.data(), kVocab), 7);   // and now it holds
}

// The spec/verify desync the handover warned of: the walk draws row i from the
// history it has accepted, so the constraint sees exactly the kept tokens. A
// draft the constraint refuses is redrawn, the walk breaks there, and the
// refused draft never reaches accept().
TEST(verify_walk_constraint_sees_only_kept_tokens) {
    std::vector<float> rows;
    for (int top : { 5, 6, 7 }) {
        const std::vector<float> r = row_with_argmax(top, top == 6 ? 9 : -1);
        rows.insert(rows.end(), r.begin(), r.end());
    }
    Sampler        sampler(greedy_grammar(), 1);
    FakeConstraint c;
    c.ok = { 5, 7, 9 };   // 6 is refused
    sampler.set_constraint(&c);
    std::vector<int> out;
    const VerifyWalk w = walk_verify(rows.data(), 3, kVocab, { 5, 6 }, sampler, 100, [](int) { return false; },
                                     [&](int tok) {
                                         out.push_back(tok);
                                         return Control::Continue;
                                     });
    CHECK_EQ(w.accepted, 1);              // draft 5 kept, draft 6 rejected
    CHECK(out == std::vector<int>({ 5, 9 }));
    CHECK(c.accepted == out);             // the grammar's history is the emitted stream
    CHECK(std::find(c.accepted.begin(), c.accepted.end(), 6) == c.accepted.end());
}

// ------------------------------------------- the GBNF on libllama's engine

#ifdef ARCINT_LLAMA
namespace {

struct GrammarRun {
    bool accepted = false;   // every character was legal
    bool complete = false;   // and the grammar may end here
};

GrammarRun run_grammar(const std::string& gbnf, const std::string& text) {
    GrammarRun     out;
    llama_grammar* g = llama_grammar_init_impl(nullptr, gbnf.c_str(), "root", false, nullptr, 0, nullptr, 0);
    if (g == nullptr) {
        t::fail(__FILE__, __LINE__, "libllama refused the generated grammar");
        return out;
    }
    try {
        llama_grammar_accept_str(*g, text);
        out.accepted = true;
        out.complete = std::any_of(g->stacks.begin(), g->stacks.end(), [](const auto& st) { return st.empty(); });
    } catch (const std::exception&) {
    }
    llama_grammar_free_impl(g);
    return out;
}

}  // namespace

TEST(toolgrammar_llama_accepts_the_well_formed_call) {
    const std::string g = generate_tool_grammar(two_tools());
    const GrammarRun  r = run_grammar(g, call("get_weather", { { "city", "Berlin" } }));
    CHECK(r.accepted);
    CHECK(r.complete);
    // optional parameter, and a free-text value with spaces and markup in it
    CHECK(run_grammar(g, call("get_weather", { { "city", "New <b>York</b>" }, { "units", "celsius" } })).complete);
    // required parameters in either order (the reference's permute)
    CHECK(run_grammar(g, call("run_query", { { "sql", "SELECT 1" }, { "limit", "10" } })).complete);
    CHECK(run_grammar(g, call("run_query", { { "limit", "10" }, { "sql", "SELECT 1" } })).complete);
}

TEST(toolgrammar_llama_refuses_the_malformed_calls) {
    const std::string g = generate_tool_grammar(two_tools());
    // an undeclared function
    CHECK(!run_grammar(g, call("delete_everything", {})).accepted);
    // a required parameter missing before the function closes
    CHECK(!run_grammar(g, call("run_query", { { "sql", "SELECT 1" } })).accepted);
    // an undeclared parameter
    CHECK(!run_grammar(g, call("get_weather", { { "town", "Berlin" } })).accepted);
    // an integer parameter given a word
    CHECK(!run_grammar(g, call("run_query", { { "sql", "x" }, { "limit", "ten" } })).accepted);
    // NOT refused, as in the reference: a string-typed value is raw text up to
    // the parameter closer (qwen3-coder.cpp's arg_string), so a string enum is
    // not enforced. Recorded so the scope of "schema-valid" is on the record.
    CHECK(run_grammar(g, call("get_weather", { { "city", "x" }, { "units", "kelvin" } })).complete);
    // an unterminated block: legal so far, but the grammar cannot end there
    const std::string full = call("get_weather", { { "city", "Berlin" } });
    const GrammarRun  cut  = run_grammar(g, full.substr(0, full.size() - tool_call_tags().close.size()));
    CHECK(cut.accepted);
    CHECK(!cut.complete);
}

TEST(toolgrammar_llama_parallel_calls_follow_the_request) {
    const std::string two = call("get_weather", { { "city", "A" } }) + "\n" + call("get_weather", { { "city", "B" } });
    CHECK(run_grammar(generate_tool_grammar(two_tools(), true), two).complete);
    CHECK(!run_grammar(generate_tool_grammar(two_tools(), false), two).accepted);
}
#endif
