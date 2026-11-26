// Inkling 2 Mini's tool-call grammar (core/toolgrammar.h
// generate_inkling_tool_grammar, after mainline llama.cpp's
// common/parsers/inkling.cpp at 690b87d8) and the parser's reject reasons
// (core/toolcall.h ToolCallParse::reject_reason). Device-free.
//
// The event behind it (`measured-here`, served, Inkling 2 Mini, twice on
// 2026-11-26): a bash call, correctly framed, whose "command" held
// tab-indented C copied raw into a heredoc. The JSON inside the block held a
// raw TAB, json::parse refused it, and the call came back as raw content with
// the log line "unterminated <|message_model|> block".
//
// Layers:
//   - every build: the parser's five reject reasons, each its own line with
//     the failing byte; the tab-heredoc block rejected without a grammar (the
//     red case as served); the think-block tracker keeping the grammar's
//     input free of thinking tokens in Inkling's token-id layout, with the
//     think mask on and off;
//   - libllama builds (the recipe's): the GBNF on llama.cpp's grammar engine
//     on text (accepts the escaped tab, refuses the raw one; the reference's
//     string rule admits it, which is the red arm and the deviation's reason),
//     the trigger pattern on llama.cpp's own matcher, and arcint's Sampler
//     driving the production LlamaToolGrammar on a real vocabulary (the
//     pinned tree's models/ggml-vocab-gpt-2.gguf, a vocab-only load with no
//     device): without the grammar the sampler emits the raw tab and the call
//     is lost, with it the tab is written as \t and the call parses; a
//     thinking block that spells the call syntax (a raw tab inside) leaves the
//     grammar dormant and the turn ends by itself, with the think mask on and
//     off, while the same run with the tracker unarmed triggers inside the
//     block (the interaction's red arm).
#include "api/handlers.h"
#include "core/chat.h"
#include "core/markers.h"
#include "core/reasoning.h"
#include "core/sampler.h"
#include "core/toolcall.h"
#include "core/toolgrammar.h"
#include "exec/backend.h"
#include "harness.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef ARCINT_LLAMA
#include <llama.h>

#include "exec/llama_toolgrammar.h"
#include "json-schema-to-grammar.h"
#include "llama-grammar.h"
#include "peg-parser.h"
#endif

using namespace lgc;

namespace {

std::string s(std::string_view v) { return std::string(v); }

const std::string kHeader = s(markers::kInkMessageModel);
const std::string kMarker = s(markers::kInkContentToolJson);
const std::string kEnd    = s(markers::kInkEndMessage);
const std::string kThink  = s(markers::kInkContentThink);
const std::string kText   = s(markers::kInkContentText);

// The served failure's shape: a heredoc of tab-indented C with a fprintf
// debug print, then a compile and a run.
const std::string kCommand = "cat > /tmp/dbg.c <<'EOF'\n"
                             "#include <stdio.h>\n"
                             "int main(void) {\n"
                             "\tint n = 3;\n"
                             "\tfprintf(stderr, \"n=%d\\n\", n);\n"
                             "\treturn 0;\n"
                             "}\n"
                             "EOF\n"
                             "gcc /tmp/dbg.c -o /tmp/dbg && /tmp/dbg";

// The args object as the template writes it (tojson, compact): every
// control byte escaped.
std::string args_escaped() { return nlohmann::json{ { "command", kCommand } }.dump(); }

// As the model wrote it when the call was lost: the newlines escaped, the
// tabs raw (the original bytes of the served block: 0 raw newlines, a raw 0x09).
std::string args_raw_tab() {
    std::string a = args_escaped();
    for (size_t at = a.find("\\t"); at != std::string::npos; at = a.find("\\t", at + 1)) a.replace(at, 2, "\t");
    return a;
}

std::string block(const std::string& name, const std::string& body) { return kHeader + name + kMarker + body + kEnd; }
std::string call_block(const std::string& name, const std::string& args) {
    return block(name, "{\"name\":" + nlohmann::json(name).dump() + ",\"args\":" + args + "}");
}

std::vector<ToolSpec> ink_tools() {
    ToolSpec bash;
    bash.name       = "bash";
    bash.parameters = nlohmann::json::parse(R"({"type":"object","properties":{"command":{"type":"string"}},"required":["command"]})");
    ToolSpec read;
    read.name       = "read_file";
    read.parameters = nlohmann::json::parse(R"({"type":"object","properties":{"path":{"type":"string"}}})");
    return { bash, read };
}

}  // namespace

// ------------------------------------------------ the reject reasons (b)

// Red as served: the block is framed right, the JSON holds a raw TAB, and the
// call is lost. The reason names the parse error, and the offset points at
// the TAB itself.
TEST(inkling_tab_heredoc_call_is_rejected_without_a_grammar) {
    const std::string   t = call_block("bash", args_raw_tab());
    const ToolCallParse p = parse_tool_calls(t, ChatDialect::Inkling);
    CHECK(p.truncated);
    CHECK(p.calls.empty());
    CHECK_EQ(p.content, t);   // returned as raw content, as before
    CHECK(p.reject_reason.rfind("JSON parse error: ", 0) == 0);
    CHECK(p.reject_offset != std::string::npos);
    CHECK(p.reject_offset < t.size() && t[p.reject_offset] == '\t');
    CHECK_EQ(p.reject_offset, t.find('\t'));
    // one log line: no raw control byte, and the model's text is not echoed
    for (const char c : p.reject_reason) CHECK(static_cast<unsigned char>(c) >= 0x20);
    CHECK(p.reject_reason.find("last read") == std::string::npos);
    CHECK(p.reject_reason.find("fprintf") == std::string::npos);
    // the escaped form is a call, with the tabs back in the decoded command
    const ToolCallParse ok = parse_tool_calls(call_block("bash", args_escaped()), ChatDialect::Inkling);
    CHECK(!ok.truncated);
    CHECK_EQ(ok.calls.size(), 1u);
    CHECK_EQ(nlohmann::json::parse(ok.calls[0].arguments).at("command").get<std::string>(), kCommand);
}

// Each of the five rejects names its own reason and the failing byte, counted
// from the block's header; the log line carries both. The Qwen form keeps its
// line exactly.
TEST(inkling_each_reject_logs_its_own_reason) {
    struct Case {
        std::string text;
        std::string reason_prefix;
        size_t      offset;
    };
    const std::string head = kHeader + "bash" + kMarker;
    const std::string pre  = "{\"name\":\"bash\",\"args\":";
    const std::string unbalanced = head + pre + "{\"a\":1}";
    const std::string open_str   = head + pre + "{\"a\":\"x";
    const std::string bad_json   = head + pre + "{\"a\":}}" + kEnd;
    const std::vector<Case> cases = {
        // 1: no `{` after the marker
        { head + "[1]" + kEnd, "no '{' after the call marker", head.size() },
        // 2: the object never closes, by braces and by an open string
        { unbalanced, "the object never closes: unbalanced braces", unbalanced.size() },
        { open_str, "the object never closes: a string is still open", open_str.size() },
        // 3: a JSON parse error, at the `}` where a value was due
        { bad_json, "JSON parse error: ", (head + pre + "{\"a\":").size() },
        // 4: name / args mismatch: another name in the header, args not an object
        { kHeader + "other" + kMarker + pre + "{}}" + kEnd, "name / args mismatch", (kHeader + "other" + kMarker).size() },
        { head + pre + "[1]}" + kEnd, "name / args mismatch", head.size() },
        // 5: no closer right after the object
        { head + pre + "{}}" + " " + kEnd, "no " + kEnd + " right after the object's '}'", (head + pre + "{}}").size() },
    };
    std::set<std::string> distinct;
    for (size_t i = 0; i < cases.size(); ++i) {
        const Case&         c = cases[i];
        const ToolCallParse p = parse_tool_calls(c.text, ChatDialect::Inkling);
        CHECK(p.truncated);
        CHECK(p.calls.empty());
        CHECK_EQ(p.content, c.text);
        if (p.reject_reason.rfind(c.reason_prefix, 0) != 0)
            t::fail(__FILE__, __LINE__, "case " + std::to_string(i) + ": reason '" + p.reject_reason + "'");
        CHECK_EQ(p.reject_offset, c.offset);
        distinct.insert(c.reason_prefix.substr(0, 12));
        const std::string line = tool_call_reject_message(p, ChatDialect::Inkling);
        CHECK(line.rfind(kHeader + " block rejected: " + p.reject_reason, 0) == 0);
        CHECK(line.find(" (at byte " + std::to_string(p.reject_offset) + " of the block)") != std::string::npos);
        CHECK(line.find("unterminated") == std::string::npos);
    }
    CHECK_EQ(distinct.size(), 5u);   // the five reasons, each its own words
    // the header-name mismatch: no model text reaches the line -- the header's
    // "name" is whatever the model wrote before the marker (here a line break
    // and a long run of text), so the reason does not quote it
    {
        const std::string junk = "other\nrm -rf / " + std::string(200, 'x');
        const std::string   t = kHeader + junk + kMarker + pre + "{}}" + kEnd;
        const ToolCallParse p = parse_tool_calls(t, ChatDialect::Inkling);
        CHECK(p.reject_reason.rfind("name / args mismatch", 0) == 0);
        CHECK(p.reject_reason.find("other") == std::string::npos);
        CHECK(p.reject_reason.find("rm -rf") == std::string::npos);
        CHECK(p.reject_reason.size() < 120);
        for (const char c : tool_call_reject_message(p, ChatDialect::Inkling))
            CHECK(static_cast<unsigned char>(c) >= 0x20);
        CHECK_EQ(p.reject_offset, (kHeader + junk + kMarker).size());
    }
    // a reject keeps the text before the block: "no '{'" kept only the block
    {
        const std::string   t = "Running it now. " + kHeader + "bash" + kMarker + "[1]" + kEnd;
        const ToolCallParse p = parse_tool_calls(t, ChatDialect::Inkling);
        CHECK(p.truncated);
        CHECK_EQ(p.content, t);
    }
    // the Qwen form: its one reject keeps the old line, word for word
    const ToolCallParse q = parse_tool_calls("<tool_call>\n{\"name\":\"x\"", ChatDialect::Qwen);
    CHECK(q.truncated);
    CHECK(q.reject_reason.empty());
    CHECK_EQ(tool_call_reject_message(q, ChatDialect::Qwen), std::string("unterminated <tool_call> block"));
}

// ------------------------------------------------ the generator's shape

TEST(inkling_grammar_names_the_declared_tools_and_the_reference_trigger) {
    const std::string g = generate_inkling_tool_grammar(ink_tools());
    CHECK(g.find("root ::= ink-call") != std::string::npos);
    CHECK(g.find("\"bash\" ink-name-ws \"<|content_invoke_tool_json|>{\\\"name\\\":\\\"bash\\\",\\\"args\\\":\"") !=
          std::string::npos);
    CHECK(g.find("\"read_file\" ink-name-ws \"<|content_invoke_tool_json|>") != std::string::npos);
    CHECK(g.find("\\x00-\\x1F") != std::string::npos);   // the strict string rule
    CHECK(generate_inkling_tool_grammar({}).empty());
    // mainline's trigger (inkling.cpp, grammar_triggers: <\|message_model\|>([^<])),
    // with the whitespace the parser trims skipped before the capture
    CHECK_EQ(inkling_tool_grammar_trigger_pattern(), std::string("<\\|message_model\\|>\\s*([^<\\s])"));
    // a name the form cannot carry is the client's error
    for (const std::string& bad : { std::string(""), std::string("<x"), std::string(" x"), std::string("x "),
                                    std::string("a") + kMarker }) {
        ToolSpec t;
        t.name       = bad;
        t.parameters = nlohmann::json::object();
        bool threw   = false;
        try {
            generate_inkling_tool_grammar({ t });
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw);
    }
}

// ------------------------------------------------ the Inkling 1 refusal (5)

// The refusal reads the prompt's last token ids, not its text: a request
// given as ids has no text, and layout after the header must not hide it.
TEST(inkling_prompt_ending_with_the_header_is_found_by_ids) {
    const std::vector<std::string> piece = { kHeader, "\n", kEnd, "<", "|message", "_model", "|>", "hello", " " };
    const auto r = [&](const std::vector<int>& ids) {
        return prompt_ends_with_inkling_header(ids, [&](int id) { return piece[static_cast<size_t>(id)]; });
    };
    CHECK(r({ 7, 0 }));                // the header as one special token
    CHECK(r({ 7, 0, 1 }));             // layout after it
    CHECK(r({ 7, 0, 1, 8, 1, 1 }));
    CHECK(r({ 7, 3, 4, 5, 6 }));       // spelled as text tokens
    CHECK(r({ 3, 4, 5, 6, 1 }));
    CHECK(!r({ 7, 2 }));               // Inkling 2 Mini's prompts end with a closer
    CHECK(!r({ 0, 7 }));
    CHECK(!r({}));
    CHECK(!r({ 1, 1 }));
}

// ------------------------------------- the known limit's audit (operator)

// The watch follows the trigger's rule on what the grammar is fed and says
// whether the trigger fired after a thinking block that opened once content
// had begun (the dormancy covers the first thinking block only).
TEST(inkling_trigger_watch_flags_a_trigger_after_a_later_thinking_block) {
    struct Case {
        std::string fed;
        bool        triggered, late_think, late_trigger;
    };
    const std::vector<Case> cases = {
        // the first thinking block (only its header and closer are fed), then a call
        { kHeader + kEnd + kHeader + "bash" + kMarker, true, false, false },
        // a text block, then a second thinking block that spells a call
        { kHeader + kEnd + kHeader + kText + "x" + kEnd + kHeader + kThink + "I could " + kHeader + "bash", true, true,
          true },
        // no first thinking block: a text block, then a thinking block that spells a call
        { kHeader + kText + "x" + kEnd + kHeader + kThink + "run " + kHeader + "bash", true, true, true },
        // a later thinking block with no call in it
        { kHeader + kEnd + kHeader + kThink + "more thought", false, true, false },
        // a header, layout, a content marker: not a call
        { kHeader + "\n" + kText + "x", false, false, false },
        // a header, layout, a name: a call
        { kHeader + "  bash", true, false, false },
        // a call in the first text block
        { kHeader + kText + "x" + kEnd + kHeader + "bash", true, false, false },
    };
    for (size_t i = 0; i < cases.size(); ++i) {
        const Case& c = cases[i];
        for (const bool bytewise : { false, true }) {
            InklingTriggerWatch w;
            if (bytewise)
                for (const char ch : c.fed) w.feed(std::string_view(&ch, 1));
            else
                w.feed(c.fed);
            if (w.triggered() != c.triggered || w.late_think() != c.late_think || w.late_trigger() != c.late_trigger)
                t::fail(__FILE__, __LINE__, "watch case " + std::to_string(i) + (bytewise ? " (bytewise)" : ""));
        }
    }
}

// The count reaches the request's decode line, where the audit greps it; a
// turn without it keeps the line it had.
TEST(decode_suffix_carries_the_grammar_late_think_count) {
    GenerationStats s;
    s.grammar_late_think = 1;
    CHECK(lgc::api::decode_suffix_for(s).find("grammar after a later think block 1") != std::string::npos);
    CHECK(lgc::api::decode_suffix_for(GenerationStats{}).find("grammar after a later think block") == std::string::npos);
}

// ------------------------------------------- the think block, token-id layout

namespace {

// Records what the sampler hands the constraint; admits everything.
struct Recorder : TokenConstraint {
    std::vector<int> accepted;
    int              asked = 0;
    bool allows(int) override {
        ++asked;
        return true;
    }
    void mask(float*, size_t) override {}
    void accept(int tok) override { accepted.push_back(tok); }
};

// Inkling's framing as single special tokens (the served vocabulary's ids are
// not needed: the roles are), plus a few text pieces.
enum : int { kIdHeader, kIdThink, kIdEnd, kIdText, kIdBash, kIdMarker, kIdJson, kIdPlan, kIdEog, kIdDone, kIdVocab };
const char* const kIdPiece[kIdVocab] = { "<|message_model|>", "<|content_thinking|>", "<|end_message|>",
                                         "<|content_text|>",  "bash",
                                         "<|content_invoke_tool_json|>",
                                         "{\"name\":\"bash\",\"args\":{\"command\":\"\tls\"}}", "plan ",
                                         "<|content_model_end_sampling|>", "No call." };

// The backend's think ids for this layout (backend_llama.cpp build_think_ids):
// the opener is two tokens, so it is matched as text; the closer is one id.
ThinkBlockIds inkling_ids_by_token(bool with_close_id) {
    ThinkBlockIds ids;
    for (const auto& [open, close] : think_tag_pairs(ChatDialect::Inkling)) {
        ids.open_close_ids.emplace_back(-1, with_close_id ? kIdEnd : -1);
        if (with_close_id) ids.close_ids.push_back(kIdEnd);
        ids.open_tags.push_back(open);
        ids.close_tags.push_back(close);
    }
    ids.eog = { kIdEog };
    return ids;
}

std::vector<int> drive_ids(const std::vector<int>& script, bool mask, bool armed, Recorder& rec, bool& ended) {
    SamplerParams p;
    p.temperature        = 0.0f;
    p.repetition_penalty = 1.0f;
    p.tool_grammar       = true;
    p.think_eog_mask     = mask;
    Sampler sampler(p, 1);
    if (armed) {
        sampler.set_think_mask_ids(inkling_ids_by_token(true));
        sampler.start_in_think_block(false, false);
    }
    sampler.set_constraint(&rec);
    std::vector<int> out;
    ended = false;
    for (int want : script) {
        std::vector<float> row(kIdVocab, 0.0f);
        row[static_cast<size_t>(want)] = 5.0f;
        const int tok = sampler.sample(row.data(), row.size());
        if (tok == kIdEog) {
            ended = true;
            break;
        }
        out.push_back(tok);
        sampler.observe(tok, kIdPiece[tok]);
    }
    sampler.set_constraint(nullptr);
    return out;
}

const std::vector<int> kThinkMentionScript = { kIdHeader, kIdThink, kIdPlan, kIdHeader, kIdBash, kIdMarker, kIdJson,
                                               kIdPlan,   kIdEnd,   kIdHeader, kIdText, kIdDone, kIdEnd, kIdEog };

[[maybe_unused]] std::string pieces_of(const std::vector<int>& ids) {
    std::string out;
    for (int id : ids) out += kIdPiece[id];
    return out;
}

}  // namespace

// The operator's constraint (2026-11-26): the call grammar must never trigger
// inside the thinking block. arcint feeds the constraint no token while the
// block is open, so a header, a tool name and a call object the model writes
// while thinking never reach the trigger -- with the think mask on and off.
// What the constraint sees is the block's opener header, its closer, and the
// text block that follows.
TEST(inkling_think_block_tokens_never_reach_the_grammar) {
    for (const bool mask : { false, true }) {
        Recorder         rec;
        bool             ended = false;
        const auto       out   = drive_ids(kThinkMentionScript, mask, /*armed*/ true, rec, ended);
        const std::vector<int> want(kThinkMentionScript.begin(), kThinkMentionScript.end() - 1);
        CHECK(out == want);
        CHECK(ended);   // the turn ends by itself: no endless turn
        CHECK(rec.accepted == std::vector<int>({ kIdHeader, kIdEnd, kIdHeader, kIdText, kIdDone, kIdEnd }));
        // no constraint test while the block was open: two tokens before it,
        // the closer's draw is inside it, then five after (the end included)
        CHECK_EQ(rec.asked, 2 + 5);
    }
    // the red arm: the same turn with the tracker unarmed hands the constraint
    // every token, the thinking ones included
    Recorder rec;
    bool     ended = false;
    drive_ids(kThinkMentionScript, false, /*armed*/ false, rec, ended);
    CHECK(std::find(rec.accepted.begin(), rec.accepted.end(), kIdBash) != rec.accepted.end());
}

#ifdef ARCINT_LLAMA
// ... and on llama.cpp's own matcher (the code the lazy sampler runs on its
// trigger buffer): the text the armed tracker lets through cannot trigger,
// the text the unarmed one lets through does, and a real call after the
// block does.
TEST(inkling_think_block_text_cannot_trigger_the_pattern) {
    llama_grammar_trigger_pattern pat;
    pat.pattern = inkling_tool_grammar_trigger_pattern();
    pat.regex   = std::regex(pat.pattern);
    for (const bool mask : { false, true }) {
        Recorder rec;
        bool     ended = false;
        drive_ids(kThinkMentionScript, mask, true, rec, ended);
        CHECK(pat.find(pieces_of(rec.accepted)) == std::string::npos);
    }
    Recorder rec;
    bool     ended = false;
    drive_ids(kThinkMentionScript, false, false, rec, ended);
    const std::string unarmed = pieces_of(rec.accepted);
    const size_t      at      = pat.find(unarmed);
    CHECK(at != std::string::npos);
    CHECK_EQ(unarmed.substr(at, 4), std::string("bash"));   // the grammar would start at the name
    // a text block and a thinking block are not calls; a call is
    CHECK(pat.find(kHeader + kText + "x" + kEnd + kHeader + kThink) == std::string::npos);
    CHECK_EQ(pat.find(kHeader + kText + "x" + kEnd + kHeader + "bash" + kMarker), (kHeader + kText + "x" + kEnd + kHeader).size());
}

// ---------------------------------------------- the GBNF on text (libllama)

namespace {

struct Run {
    bool accepted = false;
    bool complete = false;
};

Run run_grammar(const std::string& gbnf, const std::string& text) {
    Run            out;
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

// What the lazy grammar is fed once triggered: the block from the name on
// (the trigger's capture group starts there), the header itself not included.
std::string from_name(const std::string& block_text) { return block_text.substr(kHeader.size()); }

}  // namespace

TEST(inkling_gbnf_accepts_the_escaped_tab_and_refuses_the_raw_one) {
    const std::string g = generate_inkling_tool_grammar(ink_tools());
    const Run good = run_grammar(g, from_name(call_block("bash", args_escaped())));
    CHECK(good.accepted);
    CHECK(good.complete);
    const Run raw = run_grammar(g, from_name(call_block("bash", args_raw_tab())));
    CHECK(!raw.accepted);
    // every control byte, inside a string; and each escape is legal
    for (int c = 0x01; c < 0x20; ++c) {
        const std::string args = std::string("{\"command\":\"a") + static_cast<char>(c) + "b\"}";
        CHECK(!run_grammar(g, from_name(call_block("bash", args))).accepted);
    }
    for (const char* e : { "\\t", "\\n", "\\r", "\\b", "\\f", "\\\"", "\\\\", "\\/", "\\u0009" })
        CHECK(run_grammar(g, from_name(call_block("bash", std::string("{\"command\":\"a") + e + "b\"}"))).complete);
    // outside a string, a tab is JSON whitespace: the reference's space rule
    CHECK(run_grammar(g, from_name(call_block("bash", "{\n\t\"command\": \"ls\"\n}"))).complete);
}

// The red arm, and the reason for the deviation: the reference's string rule
// (its PEG builder's p.json(), which mainline's inkling.cpp uses for args)
// admits the raw TAB, so under mainline's grammar the served failure is still
// possible. The second half is a tripwire: the vendored builder's GBNF for a
// JSON string must still be the rule the red arm copies; if a pin update
// changes it, this case says so.
TEST(inkling_gbnf_reference_string_rule_admits_the_raw_tab) {
    const std::string lax = generate_inkling_tool_grammar(ink_tools(), true, /*reference_strings*/ true);
    CHECK(run_grammar(lax, from_name(call_block("bash", args_raw_tab()))).complete);
    const common_peg_arena arena = build_peg_parser([](common_peg_parser_builder& p) { return p.json(); });
    const std::string      ref   = build_grammar([&](const common_grammar_builder& b) { arena.build_grammar(b, false); });
    CHECK(ref.find(R"(( [^"\\] | "\\" ( ["\\/ bfnrt] | "u" [0-9a-fA-F]{4} ) )*)") != std::string::npos);
}

TEST(inkling_gbnf_refuses_the_malformed_calls) {
    const std::string g = generate_inkling_tool_grammar(ink_tools());
    // an undeclared tool
    CHECK(!run_grammar(g, from_name(call_block("rm_rf", "{}"))).accepted);
    // the object's name differs from the header's
    CHECK(!run_grammar(g, "bash" + kMarker + "{\"name\":\"read_file\",\"args\":{}}" + kEnd).accepted);
    // args not an object
    CHECK(!run_grammar(g, from_name(call_block("bash", "[1]"))).accepted);
    CHECK(!run_grammar(g, from_name(call_block("bash", "\"ls\""))).accepted);
    // the key spelled "arguments" (the template writes "args")
    CHECK(!run_grammar(g, "bash" + kMarker + "{\"name\":\"bash\",\"arguments\":{}}" + kEnd).accepted);
    // no closer yet: legal so far, not complete
    const std::string full = from_name(call_block("bash", "{\"command\":\"ls\"}"));
    const Run         cut  = run_grammar(g, full.substr(0, full.size() - kEnd.size()));
    CHECK(cut.accepted);
    CHECK(!cut.complete);
    // nested values, numbers, literals
    CHECK(run_grammar(g, from_name(call_block("read_file", R"({"path":"a","n":[1,-2.5e3,true,false,null,{"x":{}}]})")))
              .complete);
}

// The watch and llama.cpp's matcher agree on when the trigger fires.
TEST(inkling_trigger_watch_agrees_with_the_pattern) {
    llama_grammar_trigger_pattern pat;
    pat.pattern = inkling_tool_grammar_trigger_pattern();
    pat.regex   = std::regex(pat.pattern);
    for (const std::string& fed : { kHeader + "bash", kHeader + " \n\tbash", kHeader + "\n" + kText + "x",
                                    kHeader + kThink + "x", kHeader + kEnd + kHeader + "read_file", kHeader + "   ",
                                    std::string("plain text") }) {
        InklingTriggerWatch w;
        w.feed(fed);
        CHECK_EQ(w.triggered(), pat.find(fed) != std::string::npos);
    }
    // the capture starts at the name, past the layout
    const std::string fed = kHeader + " \n bash";
    CHECK_EQ(pat.find(fed), fed.find('b'));
}

TEST(inkling_gbnf_admits_the_whitespace_the_parser_trims) {
    const std::string g = generate_inkling_tool_grammar(ink_tools());
    const std::string body = "{\"name\":\"bash\",\"args\":{\"command\":\"ls\"}}" + kEnd;
    CHECK(run_grammar(g, "bash" + kMarker + body).complete);
    CHECK(run_grammar(g, "bash \t\n" + kMarker + body).complete);   // after the name
    // a second call's header may carry layout too
    CHECK(run_grammar(g, "bash" + kMarker + body + kHeader + "\n read_file" + kMarker +
                             "{\"name\":\"read_file\",\"args\":{}}" + kEnd)
              .complete);
    // and the parser reads these as calls
    const ToolCallParse p = parse_tool_calls(kHeader + "  bash \n" + kMarker + body, ChatDialect::Inkling);
    CHECK_EQ(p.calls.size(), 1u);
}

// A \uXXXX escape may not be a lone surrogate: nlohmann refuses one, so a
// call holding it would be lost exactly as the raw TAB was.
TEST(inkling_gbnf_refuses_lone_surrogates_as_the_parser_does) {
    const std::string g = generate_inkling_tool_grammar(ink_tools());
    const auto args = [](const std::string& esc) { return "{\"command\":\"a" + esc + "b\"}"; };
    for (const std::string& ok : { "\\u0041", "\\uD83D\\uDE00", "\\ud83d\\ude00", "\\uFFFF", "\\uE000", "\\uD7FF" }) {
        CHECK(run_grammar(g, from_name(call_block("bash", args(ok)))).complete);
        CHECK_EQ(parse_tool_calls(call_block("bash", args(ok)), ChatDialect::Inkling).calls.size(), 1u);
    }
    for (const std::string& bad : { "\\uD800", "\\uDBFF", "\\uDC00", "\\uDFFF", "\\uD800\\u0041", "\\uDE00\\uD83D" }) {
        CHECK(!run_grammar(g, from_name(call_block("bash", args(bad)))).accepted);
        CHECK(parse_tool_calls(call_block("bash", args(bad)), ChatDialect::Inkling).truncated);
    }
}

TEST(inkling_gbnf_parallel_calls_follow_the_request) {
    const std::string two = from_name(call_block("bash", "{\"command\":\"ls\"}")) + call_block("read_file", "{}");
    CHECK(run_grammar(generate_inkling_tool_grammar(ink_tools(), true), two).complete);
    CHECK(!run_grammar(generate_inkling_tool_grammar(ink_tools(), false), two).accepted);
    // a text block after a call is not a call: refused either way
    CHECK(!run_grammar(generate_inkling_tool_grammar(ink_tools(), true),
                       from_name(call_block("bash", "{}")) + kHeader + kText)
               .accepted);
}

// ------------------------------- the sampler on a real vocabulary (libllama)

namespace {

struct World {
    const llama_vocab*       v = nullptr;
    int                      n = 0;
    std::vector<std::string> piece;
    std::vector<bool>        eog;
    std::vector<int>         eog_ids;
};

const World& world() {
    static const World w = [] {
        World out;
        ggml_log_callback prev_cb = nullptr;
        void*             prev_ud = nullptr;
        llama_log_get(&prev_cb, &prev_ud);
        llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only         = true;
        static ggml_backend_dev_t no_devices[1] = { nullptr };   // no device enumerated
        mp.devices                              = no_devices;
        llama_model* m = llama_model_load_from_file(ARCINT_LLAMA_VOCAB_GGUF, mp);
        llama_log_set(prev_cb, prev_ud);
        if (m == nullptr) return out;   // kept for the process: the cases share it
        out.v = llama_model_get_vocab(m);
        out.n = llama_vocab_n_tokens(out.v);
        out.piece.resize(static_cast<size_t>(out.n));
        out.eog.resize(static_cast<size_t>(out.n));
        for (int t = 0; t < out.n; ++t) {
            char      buf[256];
            const int k = llama_token_to_piece(out.v, t, buf, sizeof buf, 0, true);
            if (k > 0) out.piece[static_cast<size_t>(t)].assign(buf, static_cast<size_t>(k));
            out.eog[static_cast<size_t>(t)] = llama_vocab_is_eog(out.v, t);
            if (out.eog[static_cast<size_t>(t)]) out.eog_ids.push_back(t);
        }
        return out;
    }();
    return w;
}

// tools/tool_grammar_probe.cpp's adversary: at every step it prefers the next
// piece of `bad` (100), then of `good` (50), longer pieces first, and wants
// to end the turn once either is complete.
void adversary_row(const World& w, const std::vector<int>& cand, const std::string& bad, const std::string& good,
                   const std::string& text, float* row) {
    std::fill(row, row + w.n, 0.0f);
    const bool at_bad = text == bad, at_good = text == good;
    for (int e : w.eog_ids) row[e] = at_bad ? 150.0f : at_good ? 120.0f : 0.0f;
    for (int t : cand) {
        const std::string& p = w.piece[static_cast<size_t>(t)];
        auto continues = [&](const std::string& target) {
            return target.size() >= text.size() + p.size() && target.compare(0, text.size(), text) == 0 &&
                   target.compare(text.size(), p.size(), p) == 0;
        };
        const float len = 2.0f * static_cast<float>(p.size());
        if (continues(bad)) row[t] = 100.0f + len;
        else if (continues(good)) row[t] = 50.0f + len;
    }
}

struct Arm {
    std::string text;
    bool        ended  = false;   // by the end-of-generation token, not the cap
    int         masked = 0;
    bool        broken = false;
    bool        late   = false;   // the grammar's late-think audit
};

// The backend's plain decode loop in miniature: sample, stop on EOG, emit,
// observe(id, piece). `grammar`: the production LlamaToolGrammar for
// Inkling. `armed`: the think-block tracker set up as prepare_think_mask
// does for this vocabulary (Inkling's tag pair, which it spells as text).
Arm run_arm(const std::string& bad, const std::string& good, bool grammar, bool armed, bool mask) {
    const World&     w = world();
    std::vector<int> cand;
    for (int t = 0; t < w.n; ++t) {
        const std::string& p = w.piece[static_cast<size_t>(t)];
        if (!p.empty() && !w.eog[static_cast<size_t>(t)] &&
            (bad.find(p) != std::string::npos || good.find(p) != std::string::npos))
            cand.push_back(t);
    }
    SamplerParams p;
    p.temperature        = 0.0f;
    p.repetition_penalty = 1.0f;
    p.tool_grammar       = grammar;
    p.think_eog_mask     = mask;
    std::unique_ptr<LlamaToolGrammar> g;
    if (grammar) g = std::make_unique<LlamaToolGrammar>(w.v, ink_tools(), true, ChatDialect::Inkling);
    Sampler sampler(p, 1);
    if (armed) {
        ThinkBlockIds ids;
        for (const auto& [open, close] : think_tag_pairs(ChatDialect::Inkling)) {
            ids.open_close_ids.emplace_back(-1, -1);   // text in this vocabulary
            ids.open_tags.push_back(open);
            ids.close_tags.push_back(close);
        }
        ids.eog = w.eog_ids;
        sampler.set_think_mask_ids(std::move(ids));
        sampler.start_in_think_block(false, false);
    }
    if (g) sampler.set_constraint(g.get());
    Arm                a;
    std::vector<float> row(static_cast<size_t>(w.n));
    const size_t       cap = std::max(bad.size(), good.size()) + 64;
    for (size_t i = 0; i < cap; ++i) {
        adversary_row(w, cand, bad, good, a.text, row.data());
        const int tok = sampler.sample(row.data(), row.size());
        if (w.eog[static_cast<size_t>(tok)]) {
            a.ended = true;
            break;
        }
        a.text += w.piece[static_cast<size_t>(tok)];
        sampler.observe(tok, w.piece[static_cast<size_t>(tok)]);
        // off both scripts: the arm has failed its case already, and the
        // rest would be all-zero rows under the grammar's full-row masks
        if (bad.compare(0, a.text.size(), a.text) != 0 && good.compare(0, a.text.size(), a.text) != 0) break;
    }
    sampler.set_constraint(nullptr);
    if (g) {
        a.masked = g->rejections();
        a.broken = g->broken();
        a.late   = g->late_trigger();
    }
    return a;
}

std::string think_block(const std::string& body) { return s(markers::kInkThinkOpen) + body + kEnd; }

// A thinking block that spells the call syntax, a raw TAB inside the object:
// if the grammar woke up in here, it would have to refuse that TAB.
const std::string kThinkMention = think_block("I could run " + kHeader + "bash" + kMarker +
                                              "{\"name\":\"bash\",\"args\":{\"command\":\"\tls\"}} but a plain "
                                              "answer will do.");

}  // namespace

TEST(inkling_vocab_loads_for_the_grammar_cases) {
    const World& w = world();
    CHECK(w.v != nullptr);   // the pinned tree's models/ggml-vocab-gpt-2.gguf: a missing file fails here
    CHECK(w.n > 1000);
    CHECK(!w.eog_ids.empty());
}

// The served failure, end to end on a real vocabulary: without the grammar
// the sampler writes the raw TAB and the call is lost (red); with it the
// draws that carry a raw TAB are masked, the model's second choice (\t) is
// written, and the block parses as the call, its command holding the tabs.
TEST(inkling_sampler_cannot_emit_a_raw_tab_in_a_call_string) {
    if (world().v == nullptr) return;   // inkling_vocab_loads_for_the_grammar_cases fails
    const std::string bad  = call_block("bash", args_raw_tab());
    const std::string good = call_block("bash", args_escaped());
    const Arm off = run_arm(bad, good, /*grammar*/ false, /*armed*/ false, false);
    CHECK_EQ(off.text, bad);
    CHECK(parse_tool_calls(off.text, ChatDialect::Inkling).calls.empty());
    for (const bool armed : { false, true }) {
        const Arm on = run_arm(bad, good, true, armed, false);
        CHECK_EQ(on.text, good);
        CHECK(on.ended);
        CHECK(!on.broken);
        CHECK(on.masked > 0);
        CHECK(on.text.find('\t') == std::string::npos);
        const ToolCallParse p = parse_tool_calls(on.text, ChatDialect::Inkling);
        CHECK(!p.truncated);
        CHECK_EQ(p.calls.size(), 1u);
        if (p.calls.size() == 1)
            CHECK_EQ(nlohmann::json::parse(p.calls[0].arguments).at("command").get<std::string>(), kCommand);
    }
}

// The operator's constraint on a real vocabulary: a thinking block that
// spells the call syntax (header, tool name, call marker, an object with a
// raw TAB) does not wake the grammar, the answer after it is free text, and
// the turn ends by itself -- with the think mask on and off. The red arm:
// the same turn with the tracker unarmed wakes the grammar inside the block
// (it masks the TAB there and the text leaves the script).
TEST(inkling_sampler_grammar_stays_dormant_while_thinking) {
    if (world().v == nullptr) return;
    const std::string answer = kThinkMention + kHeader + kText + "No call is needed." + kEnd;
    for (const bool mask : { false, true }) {
        const Arm a = run_arm(answer, answer, true, /*armed*/ true, mask);
        CHECK_EQ(a.text, answer);
        CHECK(a.ended);
        CHECK_EQ(a.masked, 0);
        CHECK(!a.broken);
        CHECK(!a.late);
        const ReasoningSplit sp = split_reasoning(a.text, ThinkStart::Undecided, ChatDialect::Inkling);
        CHECK(sp.reasoning.find(kHeader + "bash" + kMarker) != std::string::npos);
        const ToolCallParse p = parse_tool_calls(sp.content, ChatDialect::Inkling);
        CHECK(!p.truncated);
        CHECK(p.calls.empty());
        CHECK_EQ(p.content, std::string("No call is needed."));
    }
    const Arm red = run_arm(answer, answer, true, /*armed*/ false, false);
    CHECK(red.text != answer);
    CHECK(red.masked > 0);
}

// ... and a real call after such a thinking block is still constrained: the
// raw TAB in the thinking block passes (reasoning is not a call), the one in
// the call's string does not.
TEST(inkling_sampler_call_after_a_thinking_mention_is_constrained) {
    if (world().v == nullptr) return;
    const std::string bad  = kThinkMention + call_block("bash", args_raw_tab());
    const std::string good = kThinkMention + call_block("bash", args_escaped());
    for (const bool mask : { false, true }) {
        const Arm a = run_arm(bad, good, true, true, mask);
        CHECK_EQ(a.text, good);
        CHECK(a.ended);
        CHECK(!a.broken);
        const ReasoningSplit sp = split_reasoning(a.text, ThinkStart::Undecided, ChatDialect::Inkling);
        CHECK(sp.reasoning.find('\t') != std::string::npos);
        const ToolCallParse p = parse_tool_calls(sp.content, ChatDialect::Inkling);
        CHECK_EQ(p.calls.size(), 1u);
        if (p.calls.size() == 1)
            CHECK_EQ(nlohmann::json::parse(p.calls[0].arguments).at("command").get<std::string>(), kCommand);
    }
}
// Finding 3: whitespace after the header (the parser trims it) no longer
// lets the call run unconstrained: the trigger skips it, the grammar starts at
// the name, and the raw TAB is still refused.
TEST(inkling_sampler_space_after_the_header_is_still_constrained) {
    if (world().v == nullptr) return;
    const auto blk = [](const std::string& args) {
        return kHeader + " bash" + kMarker + "{\"name\":\"bash\",\"args\":" + args + "}" + kEnd;
    };
    const std::string bad = blk(args_raw_tab()), good = blk(args_escaped());
    CHECK_EQ(run_arm(bad, good, false, true, false).text, bad);
    const Arm on = run_arm(bad, good, true, true, false);
    CHECK_EQ(on.text, good);
    CHECK(on.ended);
    CHECK(!on.broken);
    CHECK(on.masked > 0);
    const ToolCallParse p = parse_tool_calls(on.text, ChatDialect::Inkling);
    CHECK_EQ(p.calls.size(), 1u);
}

// The known limit, driven: a first thinking block, a text block, then a
// second thinking block that spells a call. The tracker covers the first
// block only, so the grammar wakes inside the second -- and the audit says
// so (the decode line's count, decode_suffix_carries_the_grammar_late_think_count).
// The limit is the operator's to change; this case pins today's behaviour
// and the audit together.
TEST(inkling_sampler_late_thinking_block_is_audited) {
    if (world().v == nullptr) return;
    const std::string answer = think_block("plan") + kHeader + kText + "Let me check." + kEnd + kThinkMention +
                               kHeader + kText + "Done." + kEnd;
    const Arm a = run_arm(answer, answer, true, /*armed*/ true, false);
    CHECK(a.late);
    CHECK(a.text != answer);   // today: the second block's call syntax woke the grammar
    // the first block alone never trips it (inkling_sampler_grammar_stays_dormant_while_thinking)
}
#endif
