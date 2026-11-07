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
