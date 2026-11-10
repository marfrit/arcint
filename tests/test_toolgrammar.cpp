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

namespace {
// Two tools and a tool without parameters, no enum and no const anywhere.
std::vector<ToolSpec> plain_tools() {
    ToolSpec weather;
    weather.name       = "get_weather";
    weather.parameters = nlohmann::json::parse(R"({"type":"object","properties":{
        "city":{"type":"string"},
        "units":{"type":"string"},
        "note":{"type":["string","null"]},
        "days":{"type":"integer"}},
        "required":["city","days"]})");
    ToolSpec query;
    query.name       = "run_query";
    query.parameters = nlohmann::json::parse(R"({"type":"object","properties":{
        "sql":{"type":"string"},
        "limit":{"type":"integer"},
        "meta":{"type":"object"},
        "tags":{"type":"array","items":{"type":"string"}}},
        "required":["sql","limit"]})");
    ToolSpec ping;
    ping.name       = "ping";
    ping.parameters = nlohmann::json::parse(R"({"type":"object"})");
    return { weather, query, ping };
}
}  // namespace

// A request that lists nothing (no enum, no const) gets the grammar text it got
// before listed values existed. Every parser is an arena node and node ids are
// in the rule names (ac-354, ...), so a node built for nothing shows here as a
// moved name. The golden is the output of generate_tool_grammar(plain_tools())
// on the parent of the listed-values change (b669f57c), taken by building that
// tree on the recipe host and printing it; it is not written by hand. Red on
// the listed-values change as first made (an eps() node per property), red on
// any later change that moves a rule name or the text of an unlisted request.
TEST(toolgrammar_unlisted_request_grammar_is_the_parents) {
    const std::string golden = R"GBNF(ac-354 ::= [<] ac-354-01 | [^<] ac-354
ac-354-01 ::= [<] ac-354-01 | [/] ac-354-02 | [^/<] ac-354
ac-354-02 ::= [<] ac-354-01 | [p] ac-354-03 | [^<p] ac-354
ac-354-03 ::= [<] ac-354-01 | [a] ac-354-04 | [^<a] ac-354
ac-354-04 ::= [<] ac-354-01 | [r] ac-354-05 | [^<r] ac-354
ac-354-05 ::= [<] ac-354-01 | [a] ac-354-06 | [^<a] ac-354
ac-354-06 ::= [<] ac-354-01 | [m] ac-354-07 | [^<m] ac-354
ac-354-07 ::= [<] ac-354-01 | [e] ac-354-08 | [^<e] ac-354
ac-354-08 ::= [<] ac-354-01 | [t] ac-354-09 | [^<t] ac-354
ac-354-09 ::= [<] ac-354-01 | [e] ac-354-10 | [^<e] ac-354
ac-354-10 ::= [<] ac-354-01 | [r] ac-354-11 | [^<r] ac-354
ac-354-11 ::= [>] | [<] ac-354-01 | [^<>] ac-354
array ::= "[" space ( value ("," space value)* )? space "]"
boolean ::= ("true" | "false")
char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
decimal-part ::= [0-9]{1,16}
integer ::= ("-"? integral-part)
integral-part ::= [0] | [1-9] [0-9]{0,15}
json-array ::= "[" space ("]" | json-value (space "," space json-value)* space "]")
json-bool ::= "true" | "false"
json-null ::= "null"
json-number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? (("e" | "E") [+-]? [0-9]+)?
json-object ::= "{" space ("}" | json-string space ":" space json-value (space "," space json-string space ":" space json-value)* space "}")
json-string ::= "\"" ( [^"\\] | "\\" ( ["\\/ bfnrt] | "u" [0-9a-fA-F]{4} ) )* "\""
json-value ::= json-object | json-array | json-string | json-number | json-bool | json-null
null ::= "null"
number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
root ::= tool-call-root
space ::= | " " | "\n"{1,2} [ \t]{0,20}
string ::= "\"" char* "\""
tool-0 ::= ("<function=" "get_weather" ">\n") (tool-0-args-3 (tool-0-arg-note | tool-0-arg-units)*) "</function>\n"
tool-0-arg-city ::= ("<parameter=" "city" ">\n") xml-arg-string
tool-0-arg-days ::= ("<parameter=" "days" ">\n") integer ("\n"? "</parameter>") "\n"?
tool-0-arg-note ::= ("<parameter=" "note" ">\n") xml-arg-string
tool-0-arg-units ::= ("<parameter=" "units" ">\n") xml-arg-string
tool-0-args-1 ::= tool-0-arg-city
tool-0-args-2 ::= tool-0-arg-days
tool-0-args-3 ::= tool-0-arg-city tool-0-args-2 | tool-0-arg-days tool-0-args-1
tool-1 ::= ("<function=" "run_query" ">\n") (tool-1-args-3 (tool-1-arg-meta | tool-1-arg-tags)*) "</function>\n"
tool-1-arg-limit ::= ("<parameter=" "limit" ">\n") integer ("\n"? "</parameter>") "\n"?
tool-1-arg-meta ::= ("<parameter=" "meta" ">\n") tool-1-arg-meta-schema ("\n"? "</parameter>") "\n"?
tool-1-arg-meta-schema ::= object
tool-1-arg-sql ::= ("<parameter=" "sql" ">\n") xml-arg-string
tool-1-arg-tags ::= ("<parameter=" "tags" ">\n") tool-1-arg-tags-schema ("\n"? "</parameter>") "\n"?
tool-1-arg-tags-schema ::= "[" space (string ("," space string)*)? space "]"
tool-1-args-1 ::= tool-1-arg-limit
tool-1-args-2 ::= tool-1-arg-sql
tool-1-args-3 ::= tool-1-arg-limit tool-1-args-2 | tool-1-arg-sql tool-1-args-1
tool-2 ::= ("<function=" "ping" ">\n") "</function>\n"
tool-call ::= "<tool_call>\n" (tool-0 | tool-1 | tool-2) "</tool_call>" space
tool-call-root ::= (tool-call tool-call*)?
value ::= object | array | string | number | boolean | null
xml-arg-string ::= ac-354 "\n"?
)GBNF";
    CHECK_EQ(golden.size(), size_t(3297));
    CHECK(generate_tool_grammar(plain_tools()) == golden);
}

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
    // a string enum given a value outside it: refused since the operator
    // decision of 2026-11-09 (a deviation from the reference, whose arg_string
    // leaves every string value free; core/toolgrammar.h). Until then this
    // line asserted the reference's behaviour, that "kelvin" completes.
    CHECK(!run_grammar(g, call("get_weather", { { "city", "x" }, { "units", "kelvin" } })).accepted);
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

// A string parameter declared as an enum (or a const) takes only its listed
// strings (operator decision 2026-11-09, a deviation from the reference; the
// reason is in core/toolgrammar.h). Served on AREX-2 with --tool-grammar
// (measured-here): `"kind":"<parameter=class>\nunclear"`, i.e. the model
// opened `kind`, wrote no value, opened the next parameter inside it, and the
// free-text value took the opener as data.
namespace {

std::vector<ToolSpec> enum_tools() {
    ToolSpec f;
    f.name       = "record_finding";
    f.parameters = nlohmann::json::parse(R"({"type":"object","properties":{
        "kind":{"type":"string","enum":["finding","unclear","note"]},
        "lines_sql":{"type":"string"},
        "class":{"type":"string","enum":["clear","unclear"]}},
        "required":["kind","lines_sql","class"]})");
    ToolSpec m;
    m.name       = "pick";
    m.parameters = nlohmann::json::parse(R"({"type":"object","properties":{
        "mode":{"type":"string","enum":["find","finding","finder"]},
        "sql":{"type":"string"},
        "limit":{"type":"integer"},
        "flag":{"type":"boolean"},
        "meta":{"type":"object"},
        "level":{"type":["string","null"],"enum":["low","high",null]},
        "either":{"anyOf":[{"type":"string","enum":["a","b"]},{"type":"integer"}]},
        "fixed":{"const":"exact"},
        "loose":{"anyOf":[{"type":"string","enum":["a"]},{"type":"string"}]},
        "odd":{"type":"string","enum":["a<b","two\nlines"]}},
        "required":["mode"]})");
    return { f, m };
}

std::string pick(const std::string& body) { return wrap_call("pick", body); }

}  // namespace

// 1. the served form: kind opened, its value skipped, the next parameter opened
TEST(toolgrammar_llama_enum_refuses_the_skipped_value) {
    const std::string g      = generate_tool_grammar(enum_tools());
    const ToolCallTags& t    = tool_call_tags();
    const std::string prefix = s(t.param_open) + "kind>\n" + s(t.param_open) + "class>\nunclear" + s(t.param_end) + "\n";
    CHECK(!run_grammar(g, s(t.open) + "\n" + s(t.fn_open) + "record_finding>\n" + prefix).accepted);
    // the whole call as a free-text grammar completed it (kind took the opener as data)
    CHECK(!run_grammar(g, wrap_call("record_finding", prefix + arg_canon("lines_sql", "SELECT 1") +
                                                          arg_inline("class", "unclear")))
               .accepted);
    // even the bare opener text of another parameter cannot start an enum value
    CHECK(!run_grammar(g, s(t.open) + "\n" + s(t.fn_open) + "record_finding>\n" + s(t.param_open) + "kind>\n<").accepted);
}

// 2. an enum value closed inline or canonically completes the call
TEST(toolgrammar_llama_enum_value_takes_both_closers) {
    const std::string g = generate_tool_grammar(enum_tools());
    for (const std::string& after : { std::string("\n"), std::string("") }) {
        const GrammarRun r = run_grammar(g, wrap_call("record_finding", arg_inline("kind", "finding", after) +
                                                                            arg_canon("lines_sql", kLongSql) +
                                                                            arg_inline("class", "unclear", after)));
        CHECK(r.accepted);
        CHECK(r.complete);  // the end of generation is allowed after the call
    }
    const GrammarRun c = run_grammar(g, call("record_finding", { { "kind", "finding" }, { "lines_sql", kLongSql },
                                                                 { "class", "unclear" } }));
    CHECK(c.accepted);
    CHECK(c.complete);
}

// 3. values that are prefixes of each other: each exact, the closer decides
TEST(toolgrammar_llama_enum_prefix_values_are_exact) {
    const std::string g = generate_tool_grammar(enum_tools());
    for (const std::string& v : { std::string("find"), std::string("finding"), std::string("finder") }) {
        CHECK(run_grammar(g, pick(arg_inline("mode", v))).complete);
        CHECK(run_grammar(g, pick(arg_inline("mode", v, ""))).complete);
        CHECK(run_grammar(g, pick(arg_canon("mode", v))).complete);
    }
    for (const std::string& v : { std::string("findi"), std::string("findingx"), std::string("fin"),
                                  std::string("finde"), std::string("finders"), std::string("") }) {
        CHECK(!run_grammar(g, pick(arg_inline("mode", v))).accepted);
        CHECK(!run_grammar(g, pick(arg_canon("mode", v))).accepted);
    }
}

// 4. a value outside the enum is refused, whatever the closer; no case folding,
// no padding, no second blank line (the parser strips one newline a side)
TEST(toolgrammar_llama_enum_refuses_other_values) {
    const std::string g = generate_tool_grammar(enum_tools());
    for (const std::string& v : { std::string("maybe"), std::string("Finding"), std::string(" finding"),
                                  std::string("finding ") }) {
        CHECK(!run_grammar(g, wrap_call("record_finding", arg_inline("kind", v) + arg_canon("lines_sql", "SELECT 1") +
                                                              arg_canon("class", "clear")))
                   .accepted);
        CHECK(!run_grammar(g, wrap_call("record_finding", arg_canon("kind", v) + arg_canon("lines_sql", "SELECT 1") +
                                                              arg_canon("class", "clear")))
                   .accepted);
    }
    // a trailing newline in the value: canonically that is a second blank line
    // (inline it is the canonical form of "finding" itself, and accepted)
    CHECK(!run_grammar(g, wrap_call("record_finding", arg_canon("kind", "finding\n") +
                                                          arg_canon("lines_sql", "SELECT 1") + arg_canon("class", "clear")))
               .accepted);
    // the same through a "type" array with null, an anyOf, and a const
    CHECK(run_grammar(g, pick(arg_canon("mode", "find") + arg_canon("level", "low"))).complete);
    CHECK(run_grammar(g, pick(arg_canon("mode", "find") + arg_inline("level", "null"))).complete);
    CHECK(!run_grammar(g, pick(arg_canon("mode", "find") + arg_canon("level", "medium"))).accepted);
    CHECK(run_grammar(g, pick(arg_canon("mode", "find") + arg_inline("either", "b"))).complete);
    CHECK(run_grammar(g, pick(arg_canon("mode", "find") + arg_canon("either", "7"))).complete);
    CHECK(!run_grammar(g, pick(arg_canon("mode", "find") + arg_canon("either", "c"))).accepted);
    CHECK(run_grammar(g, pick(arg_canon("mode", "find") + arg_inline("fixed", "exact"))).complete);
    CHECK(!run_grammar(g, pick(arg_canon("mode", "find") + arg_canon("fixed", "other"))).accepted);
    // a value is raw text: `<` and a newline in a listed value are the value itself
    CHECK(run_grammar(g, pick(arg_canon("mode", "find") + arg_inline("odd", "a<b"))).complete);
    CHECK(run_grammar(g, pick(arg_canon("mode", "find") + arg_canon("odd", "two\nlines"))).complete);
    CHECK(!run_grammar(g, pick(arg_canon("mode", "find") + arg_canon("odd", "a<c"))).accepted);
    // and the parser reads a listed value back exactly
    const ToolCallParse p = parse_qwen_tool_calls(pick(arg_canon("mode", "finder") + arg_inline("odd", "two\nlines")));
    CHECK_EQ(p.calls.size(), size_t(1));
    if (p.calls.size() == 1) {
        const auto a = nlohmann::json::parse(p.calls[0].arguments);
        CHECK_EQ(a.value("mode", ""), std::string("finder"));
        CHECK_EQ(a.value("odd", ""), std::string("two\nlines"));
    }
}

// 5. every other parameter keeps today's rules
TEST(toolgrammar_llama_enum_leaves_other_parameters_alone) {
    const std::string g = generate_tool_grammar(enum_tools());
    for (const std::string& v : { std::string("SELECT * FROM lines WHERE amount < 0.01 AND qty <> 0"), std::string("<"),
                                  std::string("maybe"), std::string("line one\n</b> two") }) {
        CHECK(run_grammar(g, pick(arg_canon("mode", "find") + arg_canon("sql", v))).complete);
        CHECK(run_grammar(g, pick(arg_canon("mode", "find") + arg_inline("sql", v))).complete);
    }
    CHECK(run_grammar(g, pick(arg_canon("mode", "find") + arg_inline("limit", "10") + arg_canon("flag", "true") +
                              arg_canon("meta", R"({"a":1})")))
              .complete);
    CHECK(!run_grammar(g, pick(arg_canon("mode", "find") + arg_canon("limit", "ten"))).accepted);
    CHECK(!run_grammar(g, pick(arg_canon("mode", "find") + arg_canon("flag", "yes"))).accepted);
    CHECK(!run_grammar(g, pick(arg_canon("mode", "find") + arg_canon("meta", "not json"))).accepted);
    // a string branch without a list leaves the value free
    CHECK(run_grammar(g, pick(arg_canon("mode", "find") + arg_canon("loose", "anything <at> all"))).complete);
}

// 6. parallel_tool_calls as before, with enum values in the calls
TEST(toolgrammar_llama_enum_parallel_follows_the_request) {
    const std::string one = wrap_call("record_finding", arg_inline("kind", "note") + arg_canon("lines_sql", "SELECT 1") +
                                                            arg_inline("class", "clear"));
    const std::string gs = generate_tool_grammar(enum_tools(), false);
    CHECK(run_grammar(gs, one).complete);
    CHECK(!run_grammar(gs, one + "\n" + one).accepted);
    const std::string gp = generate_tool_grammar(enum_tools(), true);
    CHECK(run_grammar(gp, one).complete);
    CHECK(run_grammar(gp, one + "\n" + one).complete);
    CHECK(!run_grammar(gp, one + "\nResult:").accepted);
}

// ---------------------------------------------------------------------------
// Review follow-up cells for the listed values. Each names the mutation that
// turns it red.
namespace {

// one tool "t" whose parameters are optional string enums, in the order given
// (the schema's keys are ordered by nlohmann::json, so the order is the sorted one)
std::vector<ToolSpec> enum_params_tool(const std::vector<std::pair<std::string, nlohmann::json>>& props) {
    nlohmann::json schema = { { "type", "object" }, { "properties", nlohmann::json::object() } };
    for (const auto& [name, values] : props) schema["properties"][name] = { { "type", "string" }, { "enum", values } };
    ToolSpec t;
    t.name       = "t";
    t.parameters = schema;
    return { t };
}

bool takes(const std::string& g, const std::string& k, const std::string& v) {
    return run_grammar(g, wrap_call("t", arg_canon(k, v))).complete;
}
bool refuses(const std::string& g, const std::string& k, const std::string& v) {
    return !run_grammar(g, wrap_call("t", arg_canon(k, v))).accepted;
}

}  // namespace

// A listed value holding a closer the parser reads first (</parameter>,
// </function>, </tool_call>) is not enforceable: the property stays free, its
// neighbours stay listed. Red if the guard checks only </parameter>: the
// </function> and </tool_call> properties then take their list as a list and
// refuse the free text.
TEST(toolgrammar_llama_enum_closer_in_a_value_keeps_the_property_free) {
    const std::string g = generate_tool_grammar(enum_params_tool({
        { "p", nlohmann::json::array({ "a</parameter>b", "x" }) },
        { "f", nlohmann::json::array({ "a</function>b", "x" }) },
        { "c", nlohmann::json::array({ "a</tool_call>b", "x" }) },
        { "z", nlohmann::json::array({ "x", "y" }) } }));
    for (const char* k : { "p", "f", "c" }) {
        CHECK(takes(g, k, "anything <at> all"));
        CHECK(takes(g, k, "x"));
    }
    CHECK(takes(g, "z", "y"));
    CHECK(refuses(g, "z", "w"));
}

// A NUL in a listed value cannot be written into the grammar text (it is
// handed on as a C string): the property stays free. Red if NUL is not guarded:
// the grammar text then holds a NUL and libllama refuses or truncates it.
TEST(toolgrammar_llama_enum_nul_in_a_value_keeps_the_property_free) {
    const std::string g = generate_tool_grammar(enum_params_tool({
        { "n", nlohmann::json::array({ std::string("a\0b", 3), "x" }) },
        { "z", nlohmann::json::array({ "x", "y" }) } }));
    CHECK(g.find('\0') == std::string::npos);
    CHECK(takes(g, "n", "anything <at> all"));
    CHECK(takes(g, "n", "x"));
    CHECK(takes(g, "z", "y"));
    CHECK(refuses(g, "z", "w"));
}

// Two parameters whose rule names meet must each keep their own values: `kind`
// and `kind-listed` (the listed rule of `kind` was named like the argument rule
// of `kind-listed`), `a.b` and `a_b` (the builder maps both to `a-b`), `m` and
// `m-schema` (the JSON-value rule of `m`). Red when add_rule's last-wins lets
// one resolve to the other's rule.
TEST(toolgrammar_llama_enum_rule_names_cannot_collide) {
    const std::string g = generate_tool_grammar(enum_params_tool({
        { "kind", nlohmann::json::array({ "a", "b" }) },
        { "kind-listed", nlohmann::json::array({ "c", "d" }) },
        { "a.b", nlohmann::json::array({ "e", "f" }) },
        { "a_b", nlohmann::json::array({ "g", "h" }) } }));
    for (const char* v : { "a", "b" }) CHECK(takes(g, "kind", v));
    for (const char* v : { "c", "d" }) CHECK(takes(g, "kind-listed", v));
    for (const char* v : { "e", "f" }) CHECK(takes(g, "a.b", v));
    for (const char* v : { "g", "h" }) CHECK(takes(g, "a_b", v));
    CHECK(refuses(g, "kind", "c"));
    CHECK(refuses(g, "kind-listed", "a"));
    CHECK(refuses(g, "a.b", "g"));
    CHECK(refuses(g, "a_b", "e"));

    // a mixed-type property has the "-schema" rule; a parameter named like it
    ToolSpec t;
    t.name       = "t";
    t.parameters = nlohmann::json::parse(R"({"type":"object","properties":{
        "m":{"type":"integer"},
        "m-schema":{"type":"string","enum":["p","q"]}}})");
    const std::string g2 = generate_tool_grammar({ t });
    CHECK(takes(g2, "m", "12"));
    CHECK(refuses(g2, "m", "p"));
    CHECK(takes(g2, "m-schema", "q"));
    CHECK(refuses(g2, "m-schema", "12"));
}

// `nullable: true` is not a type the closure reads, and the parent grammar
// took `null` there as free text: not listed. Red when the enum is enforced
// (the free-text cells then refuse).
TEST(toolgrammar_llama_enum_nullable_stays_free) {
    ToolSpec t;
    t.name       = "t";
    t.parameters = nlohmann::json::parse(R"({"type":"object","properties":{
        "n":{"type":"string","enum":["a"],"nullable":true},
        "o":{"anyOf":[{"type":"string","enum":["a"],"nullable":true}]},
        "m":{"type":"string","enum":["a"],"nullable":false},
        "l":{"type":"string","enum":["a"]}}})");
    const std::string g = generate_tool_grammar({ t });
    for (const char* k : { "n", "o" }) {
        CHECK(takes(g, k, "null"));
        CHECK(takes(g, k, "a"));
        CHECK(takes(g, k, "something else"));
    }
    for (const char* k : { "m", "l" }) {
        CHECK(takes(g, k, "a"));
        CHECK(refuses(g, k, "null"));
    }
}

// Listed values at the edges of the closer and of GBNF literals. Red if a value
// ending in a newline also took the inline closer (the parser strips one
// newline before it, so "a\n</parameter>" reads back as "a"), if the empty
// string were dropped from the list, or if `"` or `\` went into the grammar
// text unescaped.
TEST(toolgrammar_llama_enum_edge_values) {
    const std::string g = generate_tool_grammar(enum_params_tool({
        { "nl", nlohmann::json::array({ "a\n", "b" }) },
        { "em", nlohmann::json::array({ "", "x" }) },
        { "q", nlohmann::json::array({ "say \"hi\"", "x" }) },
        { "bs", nlohmann::json::array({ "a\\b", "x" }) } }));
    // newline last: canonical closer only
    CHECK(takes(g, "nl", "a\n"));
    CHECK(!run_grammar(g, wrap_call("t", arg_inline("nl", "a\n"))).accepted);
    CHECK(run_grammar(g, wrap_call("t", arg_inline("nl", "b"))).complete);
    CHECK(refuses(g, "nl", "a"));
    const ToolCallParse p = parse_qwen_tool_calls(wrap_call("t", arg_canon("nl", "a\n")));
    CHECK_EQ(p.calls.size(), size_t(1));
    if (p.calls.size() == 1) CHECK_EQ(nlohmann::json::parse(p.calls[0].arguments).value("nl", "?"), std::string("a\n"));
    // the empty string, both closers
    CHECK(takes(g, "em", ""));
    CHECK(run_grammar(g, wrap_call("t", arg_inline("em", ""))).complete);
    CHECK(takes(g, "em", "x"));
    CHECK(refuses(g, "em", "y"));
    // a quote, a backslash
    CHECK(takes(g, "q", "say \"hi\""));
    CHECK(refuses(g, "q", "say hi"));
    CHECK(refuses(g, "q", "say \\\"hi\\\""));
    CHECK(takes(g, "bs", "a\\b"));
    CHECK(refuses(g, "bs", "ab"));
    CHECK(refuses(g, "bs", "a/b"));
}
#endif
