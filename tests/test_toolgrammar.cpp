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
//     tests/test-grammar-integration.cpp drives it).
//
//     Review B5, on the record: the toolgrammar_llama_* cases are compiled
//     into an ARCINT_LLAMA build only -- they need the pinned tree's grammar
//     engine (llama-grammar.h), and the device-free configuration has no
//     source for it. The device-free `ctest -L unit` gate therefore does NOT
//     include the refusal assertions; it covers the generator's shape and the
//     sampler's constraint hook only. The refusal leg of the gate runs where
//     the libllama build is, and the campaign records it there
//     (docs/campaigns/clean-turns.md, step 2 "the gate" -- 204 cases in the
//     libllama build, the 201-case device-free number excluding them).
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

// A string value is raw text up to the parameter closer: `<` in it is data,
// including a `<` right before the closer and a line that starts like one.
TEST(toolgrammar_llama_string_values_may_hold_angle_brackets) {
    const std::string g = generate_tool_grammar(two_tools());
    for (const std::string& v : { std::string("SELECT * FROM lines WHERE amount < 0.01 AND qty <> 0"),
                                  std::string("<b>x</b>"), std::string("a <"), std::string("<"),
                                  std::string("x </b"), std::string("line one\n</b> two") })
        CHECK(run_grammar(g, call("run_query", { { "sql", v }, { "limit", "10" } })).complete);
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

// The parameter closer in both forms the model writes. Served on AREX-2 with
// --tool-grammar (measured-here, a raw trace): short values closed inline,
// `<parameter=kind>\nfinding</parameter>`, the last one as
// `unclear</parameter>\n</function>\n</tool_call>`. A grammar that only takes
// the canonical `\n</parameter>\n` kept everything after an inline closer in
// the value, never completed, and masked the end of generation.
namespace {

std::vector<ToolSpec> finding_tool() {
    ToolSpec f;
    f.name       = "record_finding";
    f.parameters = nlohmann::json::parse(R"({"type":"object","properties":{
        "kind":{"type":"string","enum":["finding","note"]},
        "lines_sql":{"type":"string"},
        "class":{"type":"string","enum":["clear","unclear"]}},
        "required":["kind","lines_sql","class"]})");
    return { f };
}

// value, then the closer on its own line (the canonical form)
std::string arg_canon(const std::string& k, const std::string& v) {
    const ToolCallTags& t = tool_call_tags();
    return s(t.param_open) + k + ">\n" + v + "\n" + s(t.param_end) + "\n";
}
// value, then the closer on the same line (the served form), then `after`
std::string arg_inline(const std::string& k, const std::string& v, const std::string& after = "\n") {
    const ToolCallTags& t = tool_call_tags();
    return s(t.param_open) + k + ">\n" + v + s(t.param_end) + after;
}
std::string wrap_call(const std::string& name, const std::string& body) {
    const ToolCallTags& t = tool_call_tags();
    return s(t.open) + "\n" + s(t.fn_open) + name + ">\n" + body + s(t.fn_close) + "\n" + s(t.close);
}

const std::string kLongSql =
    "SELECT l.key, l.amount, l.qty\nFROM lines AS l\nJOIN heads AS h ON h.id = l.head_id\n"
    "WHERE CAST(l.amount AS REAL) < 0 AND l.qty <> 0\nORDER BY l.key";

// the measured call: short enum values inline, the SQL canonical, the last inline
std::string served_call() {
    return wrap_call("record_finding", arg_inline("kind", "finding") + arg_canon("lines_sql", kLongSql) +
                                           arg_inline("class", "unclear"));
}

}  // namespace

TEST(toolgrammar_llama_inline_closer_completes_the_call) {
    const std::string g = generate_tool_grammar(finding_tool());
    const GrammarRun  r = run_grammar(g, served_call());
    CHECK(r.accepted);
    CHECK(r.complete);  // the end of generation is allowed after the call
    // the inline closer directly followed by the next tag, no newline
    CHECK(run_grammar(g, wrap_call("record_finding", arg_inline("kind", "finding", "") +
                                                         arg_canon("lines_sql", "SELECT 1") +
                                                         arg_inline("class", "unclear", "")))
              .complete);
    // the parser reads the same call (it always ended a value at the first closer)
    const ToolCallParse p = parse_qwen_tool_calls(served_call());
    CHECK_EQ(p.calls.size(), size_t(1));
    if (p.calls.size() == 1) {
        const auto a = nlohmann::json::parse(p.calls[0].arguments);
        CHECK_EQ(a.value("kind", ""), std::string("finding"));
        CHECK_EQ(a.value("class", ""), std::string("unclear"));
        CHECK_EQ(a.value("lines_sql", ""), kLongSql);
    }
}

// canonical and inline closers mixed in one call, JSON-typed values included
TEST(toolgrammar_llama_mixed_closers_in_one_call) {
    const std::string g = generate_tool_grammar(two_tools());
    CHECK(run_grammar(g, wrap_call("run_query", arg_canon("sql", "SELECT 1") + arg_inline("limit", "10"))).complete);
    CHECK(run_grammar(g, wrap_call("run_query", arg_inline("sql", "SELECT 1") + arg_canon("limit", "10"))).complete);
    CHECK(run_grammar(g, wrap_call("run_query", arg_inline("sql", "SELECT 1") + arg_inline("limit", "10") +
                                                    arg_canon("meta", R"({"a":1})")))
              .complete);
    CHECK(run_grammar(g, wrap_call("run_query", arg_canon("sql", "SELECT 1") + arg_canon("limit", "10") +
                                                    arg_inline("meta", R"({"a":1})", "")))
              .complete);
    // a JSON-typed value is still typed under the inline closer
    CHECK(!run_grammar(g, wrap_call("run_query", arg_inline("sql", "x") + arg_inline("limit", "ten"))).accepted);
}

// `<` in a value stays data under either closer. The one text that ends a
// value is the parameter closer itself, `</parameter>`, in either form: the
// grammar now ends the value at its first occurrence, as the parser does
// (core/toolcall.cpp reads a value up to the first closer), so a value cannot
// carry that literal text; what follows it must be the next parameter or the
// function closer.
TEST(toolgrammar_llama_inline_closer_keeps_angle_brackets_in_values) {
    const std::string g = generate_tool_grammar(two_tools());
    for (const std::string& v : { std::string("SELECT key FROM t WHERE CAST(amount AS REAL) < 0 AND qty <> 0"),
                                  std::string("x </b> y"), std::string("a </param y"), std::string("a </parameter y"),
                                  std::string("a </parameter"), std::string("<"), std::string("line one\n</b> two") }) {
        CHECK(run_grammar(g, wrap_call("run_query", arg_inline("sql", v) + arg_inline("limit", "10"))).complete);
        CHECK(run_grammar(g, wrap_call("run_query", arg_canon("sql", v) + arg_canon("limit", "10"))).complete);
    }
    // the literal closer inside a value ends the value there: text after it is refused
    const std::string end = s(tool_call_tags().param_end);
    CHECK(!run_grammar(g, wrap_call("run_query", arg_canon("sql", "a" + end + "b") + arg_canon("limit", "10"))).accepted);
    CHECK(!run_grammar(g, wrap_call("run_query", arg_canon("sql", "a\n" + end + "b") + arg_canon("limit", "10"))).accepted);
}

// the canonical form is accepted exactly as before
TEST(toolgrammar_llama_canonical_closer_unchanged) {
    const std::string g = generate_tool_grammar(finding_tool());
    const std::string c = wrap_call("record_finding", arg_canon("kind", "finding") + arg_canon("lines_sql", kLongSql) +
                                                          arg_canon("class", "unclear"));
    CHECK(c == call("record_finding", { { "kind", "finding" }, { "lines_sql", kLongSql }, { "class", "unclear" } }));
    const GrammarRun r = run_grammar(g, c);
    CHECK(r.accepted);
    CHECK(r.complete);
}

// parallel_tool_calls false: after one inline-closed call only the end of
// generation (or trailing whitespace) is legal; true: the end or another call
TEST(toolgrammar_llama_inline_closer_parallel_follows_the_request) {
    const std::string one = served_call();
    const std::string gs  = generate_tool_grammar(finding_tool(), false);
    CHECK(run_grammar(gs, one).complete);
    CHECK(!run_grammar(gs, one + "\nResult:").accepted);
    CHECK(!run_grammar(gs, one + "\n" + one).accepted);
    const std::string gp = generate_tool_grammar(finding_tool(), true);
    CHECK(run_grammar(gp, one).complete);
    CHECK(run_grammar(gp, one + "\n" + one).complete);
    CHECK(!run_grammar(gp, one + "\nResult:").accepted);
}
#endif
