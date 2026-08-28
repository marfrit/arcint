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
