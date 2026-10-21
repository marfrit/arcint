// clean-turns step 2: the forced-malformed gate on the real vocabulary, with
// no card. Loads a GGUF vocab-only (no tensors, no device), builds the
// request's tool-call grammar exactly as the backend does
// (exec/llama_toolgrammar.h), and drives arcint's own Sampler with logits
// written by an ADVERSARY that wants a malformed call:
//
//   at every step the adversary prefers the next piece of a malformed text M
//   (logit 10), then the next piece of a well-formed text V (logit 5), and
//   wants to end the turn once M is complete (8); everything else is 0.
//
// So with the grammar off the output is M, byte for byte: the red case. With
// it on, the sampler must refuse M's pieces where M leaves the grammar and
// fall back to V, and the result must parse as a schema-valid call. Each case
// runs in the plain loop's shape and in the speculative shape (the adversary
// drafts M's next pieces, the target rows are written as if every draft were
// kept, the verify walk keeps what the sampler draws): a grammar that
// accepted a rejected draft would refuse the next token, give up, and fail
// the case.
//
// Exit 0 only when every OFF arm is malformed and every ON arm is valid.
// ARCINT_PROBE_RED_DESYNC=1 makes the speculative shape also feed the grammar
// the first rejected draft: the spec ON arms must then fail (the check's own
// red case).
// Built in a libllama build with ARCINT_TESTS (not a ctest case: it needs the
// GGUF). Usage: arcint-tool-grammar-probe <file.gguf> [seeds]
#include <llama.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "core/sampler.h"
#include "core/toolcall.h"
#include "exec/llama_toolgrammar.h"
#include "exec/verify_walk.h"

using namespace lgc;

namespace {

std::string s(std::string_view v) { return std::string(v); }

std::string call(const std::string& name, const std::vector<std::pair<std::string, std::string>>& args,
                 bool closed = true) {
    const ToolCallTags& t   = tool_call_tags();
    std::string         out = s(t.open) + "\n" + s(t.fn_open) + name + ">\n";
    for (const auto& [k, v] : args) out += s(t.param_open) + k + ">\n" + v + "\n" + s(t.param_end) + "\n";
    if (!closed) return out;
    return out + s(t.fn_close) + "\n" + s(t.close);
}

std::vector<ToolSpec> tools() {
    ToolSpec rf;
    rf.name       = "read_file";
    rf.parameters = nlohmann::json::parse(R"({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]})");
    ToolSpec rq;
    rq.name       = "run_query";
    rq.parameters = nlohmann::json::parse(
        R"({"type":"object","properties":{"sql":{"type":"string"},"limit":{"type":"integer"}},"required":["sql","limit"]})");
    return { rf, rq };
}

struct Case {
    const char* name;
    std::string bad;    // what the adversary wants
    std::string good;   // its second choice
};

struct World {
    const llama_vocab*       v = nullptr;
    int                      n = 0;
    std::vector<std::string> piece;
    std::vector<bool>        eog;
};

// The adversary's row after `text`; `bad` complete means it wants to stop.
// Longer pieces are preferred hard (2 per byte), so the adversary spells its
// text the way a model does -- the opener as its one special token -- and the
// margins (bad 100, good 50, at most ~40 from length) keep its order of
// preference at T=0.7 too. Only tokens whose piece occurs in either text can
// score; they are collected once per case.
void adversary_row(const World& w, const Case& c, const std::vector<int>& cand, const std::string& text, float* row) {
    std::fill(row, row + w.n, 0.0f);
    const bool at_bad = text == c.bad, at_good = text == c.good;
    for (int t = 0; t < w.n; ++t) {
        if (!w.eog[static_cast<size_t>(t)]) continue;
        row[t] = at_bad ? 150.0f : at_good ? 120.0f : 0.0f;
    }
    for (int t : cand) {
        const std::string& p   = w.piece[static_cast<size_t>(t)];
        const float        len = 2.0f * static_cast<float>(p.size());
        auto continues = [&](const std::string& target) {
            return target.size() >= text.size() + p.size() && target.compare(0, text.size(), text) == 0 &&
                   target.compare(text.size(), p.size(), p) == 0;
        };
        if (continues(c.bad)) row[t] = 100.0f + len;
        else if (continues(c.good)) row[t] = 50.0f + len;
    }
}

// The adversary's own drafts: the argmax of its rows, as if each were kept.
std::vector<int> adversary_drafts(const World& w, const Case& c, const std::vector<int>& cand, std::string text, int k,
                                  std::vector<float>& rows) {
    std::vector<int> d;
    rows.assign(static_cast<size_t>(k + 1) * static_cast<size_t>(w.n), 0.0f);
    for (int i = 0; i <= k; ++i) {
        float* row = rows.data() + static_cast<size_t>(i) * static_cast<size_t>(w.n);
        adversary_row(w, c, cand, text, row);
        if (i == k) break;
        const int best = Sampler::argmax(row, static_cast<size_t>(w.n));
        if (w.eog[static_cast<size_t>(best)]) {
            rows.resize(static_cast<size_t>(i + 1) * static_cast<size_t>(w.n));
            break;
        }
        d.push_back(best);
        text += w.piece[static_cast<size_t>(best)];
    }
    return d;
}

bool red_desync() {
    static const bool on = [] {
        const char* e = std::getenv("ARCINT_PROBE_RED_DESYNC");
        return e != nullptr && std::string(e) == "1";
    }();
    return on;
}

struct Arm {
    std::string text;
    bool        broken = false;
    int         masked = 0;
};

Arm run(const World& w, const Case& c, const std::vector<int>& cand, bool grammar_on, bool spec, float temp,
        uint64_t seed) {
    SamplerParams p;
    p.temperature        = temp;
    p.top_k              = 0;
    p.top_p              = 1.0f;
    p.repetition_penalty = 1.0f;
    p.tool_grammar       = grammar_on;
    Sampler                           sampler(p, seed);
    std::unique_ptr<LlamaToolGrammar> g;
    if (grammar_on) {
        g = std::make_unique<LlamaToolGrammar>(w.v, tools(), true);
        sampler.set_constraint(g.get());
    }
    Arm                a;
    const int          cap = 400;
    std::vector<float> rows;
    int                emitted = 0;
    while (emitted < cap) {
        if (!spec) {
            rows.assign(static_cast<size_t>(w.n), 0.0f);
            adversary_row(w, c, cand, a.text, rows.data());
            const int tok = sampler.sample(rows.data(), static_cast<size_t>(w.n));
            if (w.eog[static_cast<size_t>(tok)]) break;
            a.text += w.piece[static_cast<size_t>(tok)];
            sampler.observe_think_text(w.piece[static_cast<size_t>(tok)]);
            sampler.observe(tok);
            ++emitted;
            continue;
        }
        const std::vector<int> draft = adversary_drafts(w, c, cand, a.text, 4, rows);
        const VerifyWalk vw = walk_verify(rows.data(), rows.size() / static_cast<size_t>(w.n), static_cast<size_t>(w.n),
                                          draft, sampler, cap - emitted,
                                          [&](int tok) { return static_cast<bool>(w.eog[static_cast<size_t>(tok)]); },
                                          [&](int tok) {
                                              a.text += w.piece[static_cast<size_t>(tok)];
                                              sampler.observe_think_text(w.piece[static_cast<size_t>(tok)]);
                                              return Control::Continue;
                                          });
        emitted += vw.emitted;
        // the red case for the desync check: a loop that fed the grammar the
        // first rejected draft as well (ARCINT_PROBE_RED_DESYNC=1) must fail
        if (g && red_desync() && static_cast<size_t>(vw.accepted) < draft.size() && !vw.done)
            g->accept(draft[static_cast<size_t>(vw.accepted)]);
        if (vw.done) break;
    }
    if (g) {
        a.broken = g->broken();
        a.masked = g->rejections();
    }
    return a;
}

// Schema-valid: exactly one call, closed, a declared name, every required
// parameter present, every declared integer an integer, nothing undeclared.
bool valid(const std::string& text, std::string& why) {
    ToolSchemas schemas;
    schemas["read_file"] = { { "path", "string" } };
    schemas["run_query"] = { { "sql", "string" }, { "limit", "integer" } };
    const ToolCallParse parsed = parse_qwen_tool_calls(text, &schemas);
    if (parsed.truncated) { why = "unterminated call block"; return false; }
    if (parsed.calls.size() != 1) { why = std::to_string(parsed.calls.size()) + " calls"; return false; }
    const ToolCall& tc = parsed.calls[0];
    for (const ToolSpec& t : tools()) {
        if (t.name != tc.name) continue;
        nlohmann::json args;
        try {
            args = nlohmann::json::parse(tc.arguments);
        } catch (...) {
            why = "arguments are not JSON";
            return false;
        }
        for (const auto& r : t.parameters["required"])
            if (!args.contains(r.get<std::string>())) { why = "required '" + r.get<std::string>() + "' missing"; return false; }
        for (auto it = args.begin(); it != args.end(); ++it) {
            if (!t.parameters["properties"].contains(it.key())) { why = "undeclared '" + it.key() + "'"; return false; }
            if (t.parameters["properties"][it.key()]["type"] == "integer" && !it.value().is_number_integer()) {
                why = "'" + it.key() + "' is not an integer";
                return false;
            }
        }
        why = tc.name + " " + tc.arguments;
        return true;
    }
    why = "undeclared function '" + tc.name + "'";
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <file.gguf> [seeds]\n", argv[0]);
        return 2;
    }
    const int seeds = argc > 2 ? std::atoi(argv[2]) : 3;
    llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only         = true;
    llama_model* m        = llama_model_load_from_file(argv[1], mp);
    if (m == nullptr) {
        std::fprintf(stderr, "load failed\n");
        return 1;
    }
    World w;
    w.v = llama_model_get_vocab(m);
    w.n = llama_vocab_n_tokens(w.v);
    w.piece.resize(static_cast<size_t>(w.n));
    w.eog.resize(static_cast<size_t>(w.n));
    for (int t = 0; t < w.n; ++t) {
        char      buf[256];
        const int k = llama_token_to_piece(w.v, t, buf, sizeof buf, 0, true);
        if (k > 0) w.piece[static_cast<size_t>(t)].assign(buf, static_cast<size_t>(k));
        w.eog[static_cast<size_t>(t)] = llama_vocab_is_eog(w.v, t);
    }
    {
        LlamaToolGrammar probe(w.v, tools(), true);
        std::printf("vocabulary %d tokens; grammar %zu bytes; the opener is %s\n", w.n, probe.gbnf().size(),
                    probe.trigger_is_token() ? "one token (trigger token)" : "several tokens (trigger pattern)");
    }

    const std::vector<Case> cases = {
        { "undeclared-function", call("delete_everything", { { "path", "/" } }), call("read_file", { { "path", "/" } }) },
        { "missing-required", call("run_query", { { "sql", "SELECT 1" } }),
          call("run_query", { { "sql", "SELECT 1" }, { "limit", "10" } }) },
        { "wrong-type", call("run_query", { { "sql", "SELECT 1" }, { "limit", "ten" } }),
          call("run_query", { { "sql", "SELECT 1" }, { "limit", "10" } }) },
        { "unterminated", call("read_file", { { "path", "a.txt" } }, false), call("read_file", { { "path", "a.txt" } }) },
        { "undeclared-param", call("read_file", { { "file", "a.txt" } }), call("read_file", { { "path", "a.txt" } }) },
    };

    int bad = 0;
    for (const Case& c : cases) {
        std::vector<int> cand;
        for (int t = 0; t < w.n; ++t) {
            const std::string& p = w.piece[static_cast<size_t>(t)];
            if (!p.empty() && !w.eog[static_cast<size_t>(t)] &&
                (c.bad.find(p) != std::string::npos || c.good.find(p) != std::string::npos))
                cand.push_back(t);
        }
        for (int spec = 0; spec < 2; ++spec) {
            for (int arm = 0; arm < 1 + seeds; ++arm) {
                const float    temp = arm == 0 ? 0.0f : 0.7f;
                const uint64_t seed = static_cast<uint64_t>(arm);
                for (int on = 0; on < 2; ++on) {
                    const Arm   a = run(w, c, cand, on != 0, spec != 0, temp, seed);
                    std::string why;
                    const bool  ok   = valid(a.text, why);
                    const bool  pass = on ? (ok && !a.broken) : (!ok && a.text == c.bad);
                    if (!pass) ++bad;
                    std::printf("%-20s %-5s T=%.1f seed=%d grammar=%-3s -> %-9s %s%s (masked %d)\n", c.name,
                                spec ? "spec" : "plain", temp, arm, on ? "on" : "off", ok ? "VALID" : "MALFORMED",
                                why.c_str(), a.broken ? " [grammar gave up]" : "", a.masked);
                    if (!pass) std::printf("    FAIL: expected %s\n", on ? "a valid call" : "the adversary's malformed text");
                }
            }
        }
    }
    llama_model_free(m);
    std::printf(bad ? "RESULT: %d arms failed\n" : "RESULT: every OFF arm malformed, every ON arm valid\n", bad);
    return bad ? 1 : 0;
}
