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
#include "core/sampler.h"
#include "core/toolcall.h"
#include "core/toolgrammar.h"
#include "exec/verify_walk.h"
#include "harness.h"

#include <algorithm>
#include <cmath>
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
