# Local patches to vendored minja

Kept small and listed here so a future update can re-apply or drop them.

## `undefined` test (2026-08-28)

`minja.hpp`, in the `is` operator: upstream implements `defined` but not
`undefined`. Qwen3.8-27B's `chat_template.jinja` opens with

    {%- if enable_thinking is undefined or enable_thinking is true %}

so without the complement, rendering that artifact's own template throws
`Unknown type for 'is' operator: undefined` and the model cannot be served at
all. DESIGN.md §3.7 makes the artifact's template the single source of truth,
so working around it by polyfilling or substituting a template is exactly the
drift that rule exists to prevent — the engine has to render what shipped.

Upstream: worth a PR against google/minja.

## `tojson(ensure_ascii=...)` (2026-11-06)

`minja.hpp`, the global `tojson`: upstream's signature is `(value, indent)`.
Granite 4.2 8B's `chat_template` declares its tools with
`v | tojson(ensure_ascii=False)`, so without the keyword every render with
`tools` throws `Unknown argument ensure_ascii for function tojson` and the
model cannot take a tool request at all. The keyword is accepted and ignored:
`Value::dump` already writes non-ASCII text unescaped, which is what
`ensure_ascii=False` asks for. (`ensure_ascii=True`, Python's default, would
escape it; minja has never done that and no admitted template asks for it.)

Upstream: worth a PR against google/minja.

## Trailing comma in a dictionary literal (2026-11-22)

`minja.hpp`, `Parser::parseDictionary`: a `,` directly before the closing
`}` threw `Expected value expression`. Python/Jinja accepts the trailing
comma, and Inkling 2 Mini's `chat_template.jinja` writes every tool spec that
way, so without this the template could not be parsed and the model could not
be served. The parser now returns the dictionary when the comma is followed by
`}`, exactly as Jinja does.

Upstream: worth a PR against google/minja.

## `float` filter (2026-11-22)

`minja.hpp`, `Context::builtins`: Jinja's `float` filter was absent, so a
numeric `reasoning_effort` in Inkling 2 Mini's template (`set num = eff | float`)
threw `Value is not callable: null` on every render. Accepts numbers unchanged
and parses strings with Jinja's fallback of `0.0` on failure.

Upstream: worth a PR against google/minja.

## `tojson` options and `dictsort(case_sensitive=)` (2026-11-22)

Inkling 2 Mini passes `tojson(sort_keys=true, separators=(",", ":"))` and
`dictsort(case_sensitive=true)`. `tojson` now accepts both keywords: compact
separators are `dump`'s behaviour at indent -1, and `sort_keys` sorts object
keys recursively, like Python. `dictsort` no longer rejects extra keywords and
accepts `case_sensitive` (keys sort as strings either way).

## Macro and call-block scope (2026-11-27)

`minja.hpp`, `MacroNode::do_render` and `CallNode::do_render`: the macro's
callable captured the defining context (`macro_context`, a
`std::shared_ptr<Context>`) by value and was then stored into that same
context; a call block's `caller` did the same with its context. That is a
reference cycle (Context -> Value -> std::function -> lambda -> Context), so
every render that executed a `{% macro %}` definition or a `{% call %}` block
leaked its scope chain: ~15-25 KB a render (LeakSanitizer), linear in the
render count. Every admitted production template defines a macro.

The callables now capture a `std::weak_ptr<Context>` and lock it at call
time. Jinja macros close over their defining scope lexically, and that scope
must stay reachable for the whole render (a macro defined in a loop body and
kept in a `namespace` is called after the loop's context is gone), so
`render_scope` keeps each defining context alive until the outermost
`TemplateNode::render` on the thread returns. The bound: one render keeps
every defining context it creates until it returns (a loop defining a macro
10^5 times holds 10^5 contexts until then; the old engine held them forever).
The keeper is a thread-local static that no context or value refers to, so it
closes no cycle. A macro called after
that render, with no other owner of its context (reachable only through the
C++ API), throws `Macro <name> called after the render that defined it ended`.
Within a render the scopes resolve as before; the production templates render
byte-identically. Tests: `tests/test_minja_scope.cpp`.

Upstream: google/minja `include/minja/minja.hpp` at `main` (last changed by
3e4c61c6, 2025-09-09) has the same strong capture in both nodes; no upstream
fix or issue was found. llama.cpp at the recipe's pin (3e4dfa8) no longer
carries minja: its own `common/jinja` engine runs a macro body in a scope
copied from the *caller's* context at call time (dynamic scoping), so it
holds no defining context and has no cycle, but resolves names differently
from Jinja and from minja; not followed here, since the templates must render
as before. Worth a PR against google/minja.

## `tojson` key sorter (2026-11-27)

`minja.hpp`, the global `tojson` (this file's 2026-11-22 entry): the
recursive key sorter was a `std::function` held by a `shared_ptr` that the
function captured by value, a self-cycle created on every `tojson` call,
sorting or not (~100 B a call). It is now a local `std::function` captured by
reference, which lives for the call. Local code; nothing upstream.
