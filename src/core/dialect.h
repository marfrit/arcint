#pragma once

namespace lgc {

// Which chat dialect a loaded model speaks: the think-block tags its reasoning
// is split on and the form its tool calls are parsed from. A model never
// changes dialect, so the backend sets it once at load (ModelStatus::dialect)
// and every request carries it (Prepared::dialect).
//
//   Qwen      the Qwen think pair; the Qwen tool-call forms (the default, and
//             every model admitted before Gemma 4)
//   Extended  the Qwen pair plus the angle- and square-bracket pairs of the
//             Mistral family; tool calls are Qwen's (not parsed on that family)
//   Gemma4    Gemma 4's thought channel and its own tool-call form
//   Inkling   Inkling 2 Mini's message-block framing: a reasoning block, one
//             or more answer blocks and tool-call blocks, all closed by the
//             same marker; tool calls are its own JSON form
//   Neutral   the "runs" tier's default: think blocks are split only by the
//             vocabulary's single-token tag ids (no text scan), no tool-call
//             parsing. The Qwen dialect replaces it only when the template
//             carries the Qwen call form (core/chat.cpp).
//
// The constructor from bool (true = Extended, false = Qwen) is explicit: a
// bool is not a dialect by accident.
struct ChatDialect {
    enum Kind : unsigned char { Qwen, Extended, Gemma4, Inkling, Neutral };
    Kind kind = Qwen;

    constexpr ChatDialect() = default;
    constexpr ChatDialect(Kind k) : kind(k) {}
    constexpr explicit ChatDialect(bool extended) : kind(extended ? Extended : Qwen) {}

    constexpr bool extended() const { return kind == Extended; }
    constexpr bool gemma4() const { return kind == Gemma4; }
    constexpr bool inkling() const { return kind == Inkling; }
    constexpr bool neutral() const { return kind == Neutral; }
    // Assistant tool calls are replayed into the template with their arguments as
    // an object, whatever the template's capability probe says. Gemma 4's template
    // renders an object in its own key:value form and a JSON string by stripping
    // the outer braces, so the probe (which accepts a string) would have the model
    // see its earlier calls in a syntax it never writes.
    // What the model's template does with a request that says nothing about
    // thinking: the Qwen templates (Granite's too) think; Gemma 4's reads
    // `enable_thinking | default(false)`. Inkling's template takes a reasoning
    // effort, not an enable flag, and leaves the answer's channel to the effort. An unset request is resolved to this
    // before the render, the sampler row and the answer-row switch read it, so
    // the three cannot disagree.
    // Inkling's template reads only reasoning_effort and thinks at 0.9 when a request names none:
    // an unset request is a thinking turn there (1.8.0; false here made arcint set enable_thinking=false,
    // which the effort mapping then rendered as none).
    constexpr bool thinks_by_default() const { return kind != Gemma4; }
    constexpr bool replays_arguments_as_object() const { return kind == Gemma4 || kind == Inkling; }
    // A tool result is rendered with its function's name from our side: Inkling's template
    // looks a nameless result's tool_call_id up over ALL messages and appends every match
    // to the header, so a history whose ids repeat (a proxy that rewrites them per turn)
    // renders "readbash" and changes earlier prompt text between turns. With a `name` on
    // the result the template takes that name and does no lookup (core/chat_json.h).
    constexpr bool names_tool_results() const { return kind == Inkling; }
    friend constexpr bool operator==(ChatDialect a, ChatDialect b) { return a.kind == b.kind; }
    friend constexpr bool operator!=(ChatDialect a, ChatDialect b) { return a.kind != b.kind; }
};

}  // namespace lgc
