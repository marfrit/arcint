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
//
// Converts implicitly from bool for the callers that predate the third
// dialect: true meant "extended tags".
struct ChatDialect {
    enum Kind : unsigned char { Qwen, Extended, Gemma4 };
    Kind kind = Qwen;

    constexpr ChatDialect() = default;
    constexpr ChatDialect(Kind k) : kind(k) {}
    constexpr ChatDialect(bool extended) : kind(extended ? Extended : Qwen) {}

    constexpr bool extended() const { return kind == Extended; }
    constexpr bool gemma4() const { return kind == Gemma4; }
    // Assistant tool calls are replayed into the template with their arguments as
    // an object, whatever the template's capability probe says. Gemma 4's template
    // renders an object in its own key:value form and a JSON string by stripping
    // the outer braces, so the probe (which accepts a string) would have the model
    // see its earlier calls in a syntax it never writes.
    // What the model's template does with a request that says nothing about
    // thinking: the Qwen templates (Granite's too) think; Gemma 4's reads
    // `enable_thinking | default(false)`. An unset request is resolved to this
    // before the render, the sampler row and the answer-row switch read it, so
    // the three cannot disagree.
    constexpr bool thinks_by_default() const { return kind != Gemma4; }
    constexpr bool replays_arguments_as_object() const { return kind == Gemma4; }
    friend constexpr bool operator==(ChatDialect a, ChatDialect b) { return a.kind == b.kind; }
    friend constexpr bool operator!=(ChatDialect a, ChatDialect b) { return a.kind != b.kind; }
};

}  // namespace lgc
