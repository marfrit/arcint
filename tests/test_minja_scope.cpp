#include "harness.h"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

// minja declares `using json = nlohmann::ordered_json` at global scope; its
// cases live in their own translation unit (see test_chat_render.cpp).
#include <minja/chat-template.hpp>

#if defined(__SANITIZE_ADDRESS__)
#define ARCINT_TEST_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define ARCINT_TEST_ASAN 1
#endif
#endif
#if defined(ARCINT_TEST_ASAN)
// The sanitizer runtime's own count (sanitizer/allocator_interface.h, which not
// every toolchain installs; the runtime always exports the symbol).
extern "C" size_t __sanitizer_get_current_allocated_bytes(void);
#else
#include <malloc.h>
#endif

// The vendored minja's macro scope (third_party/minja/PATCHES.md, "Macro and
// call-block scope"). A {% macro %} definition stored a callable that owned its
// defining context into that same context, and a {% call %} block did the same
// with `caller`: a reference cycle, so every render of a template that defines a
// macro leaked its whole scope chain (every production template defines one).
// Mutation: put back the strong capture in MacroNode::do_render (capture
// macro_context, drop the lock) and the *_released / *_flat cases fail; put it
// back in CallNode::do_render and the call-block arms fail; make tojson's
// sort_recursive a shared_ptr captured in itself again and the tojson case fails.

namespace {

// Bytes the process holds right now, from the allocator itself: ASan's in a
// sanitized build, glibc's arenas plus mmapped chunks otherwise.
size_t live_heap_bytes() {
#if defined(ARCINT_TEST_ASAN)
    return __sanitizer_get_current_allocated_bytes();
#else
    const struct mallinfo2 mi = mallinfo2();
    return mi.uordblks + mi.hblkhd;
#endif
}

constexpr const char* kOneMacro =
    "{%- macro greet(name, punct='!') -%}hello {{ name }}{{ punct }}{%- endmacro -%}"
    "{%- for m in messages -%}{{ greet(m.content) }}{%- endfor -%}";

constexpr const char* kCallBlock =
    "{%- macro wrap(tag) -%}<{{ tag }}>{{ caller() }}</{{ tag }}>{%- endmacro -%}"
    "{%- for m in messages -%}{%- call wrap('b') -%}{{ m.content }}{%- endcall -%}{%- endfor -%}";

// The local `tojson` patch (PATCHES.md, 2026-11-22): its key sorter held itself
// through a shared_ptr, a cycle made on every call, sorting or not (~100 B a call).
constexpr const char* kToJson =
    "{%- for m in messages -%}{{ m | tojson }}{{ m | tojson(sort_keys=true) }}"
    "{{ m | tojson(separators=(',', ':')) }}{{ m | tojson(sort_keys=true, separators=(',', ':')) }}{%- endfor -%}";

std::string apply(const minja::chat_template& tmpl) {
    minja::chat_template_inputs in;
    in.messages              = nlohmann::ordered_json::array({{{"role", "user"}, {"content", "hi"}}});
    in.add_generation_prompt = true;
    minja::chat_template_options opts;
    opts.apply_polyfills = false;
    return tmpl.apply(in, opts);
}

// Growth of the live heap across n renders of one template object, after one
// warm-up render (one-time statics, the first render's caches).
size_t heap_growth(const minja::chat_template& tmpl, int n) {
    (void) apply(tmpl);
    const size_t before = live_heap_bytes();
    for (int i = 0; i < n; ++i) (void) apply(tmpl);
    const size_t after = live_heap_bytes();
    return after > before ? after - before : 0;
}

void check_flat(const char* source) {
    minja::chat_template tmpl(source, "", "");
    const size_t g1  = heap_growth(tmpl, 1);
    const size_t g50 = heap_growth(tmpl, 50);
    // Flat in N: 49 more renders hold no more memory than one did. Before the
    // fix a one-macro render kept ~15 KB, so 49 more renders keep ~0.7 MB.
    if (!(g50 <= g1 + 2048)) {
        t::fail(__FILE__, __LINE__,
                "live heap grows with renders: +" + std::to_string(g1) + " B after 1, +" +
                    std::to_string(g50) + " B after 50 (" + source + ")");
    }
}

// Renders `source` against a root context the test owns, then drops it: the
// root context must die with its last owner (no callable inside owns it).
bool root_context_released(const char* source) {
    auto root = minja::Parser::parse(source, {/* trim_blocks */ true, /* lstrip_blocks */ true, false});
    auto ctx  = minja::Context::make(minja::Value::object());
    ctx->set("items", minja::Value(nlohmann::ordered_json::array({1, 2})));
    (void) root->render(ctx);
    std::weak_ptr<minja::Context> watch = ctx;
    ctx.reset();
    return watch.expired();
}

std::string render(const char* source, const std::shared_ptr<minja::Context>& ctx = nullptr) {
    auto root = minja::Parser::parse(source, {true, true, false});
    return root->render(ctx ? ctx : minja::Context::make(minja::Value::object()));
}

}  // namespace

TEST(minja_macro_defining_context_is_released_after_render) {
    CHECK(root_context_released("{% macro m(x) %}[{{ x }}]{% endmacro %}{{ m(1) }}"));
    CHECK(root_context_released("{% macro m() %}x{% endmacro %}"));   // defined, never called
    CHECK(root_context_released(
        "{% macro w() %}<{{ caller() }}>{% endmacro %}{% call w() %}b{% endcall %}"));
    // a macro defined inside a loop body (its context is the loop's)
    CHECK(root_context_released("{% for i in items %}{% macro m() %}{{ i }}{% endmacro %}{{ m() }}{% endfor %}"));
}

TEST(minja_macro_render_live_heap_is_flat_in_render_count) {
    check_flat(kOneMacro);
    check_flat(kCallBlock);
}

TEST(minja_tojson_live_heap_is_flat_in_render_count) {
    check_flat(kToJson);
    minja::chat_template tmpl(kToJson, "", "");
    CHECK_EQ(apply(tmpl), std::string("{\"role\": \"user\", \"content\": \"hi\"}"
                                      "{\"content\": \"hi\", \"role\": \"user\"}"
                                      "{\"role\":\"user\",\"content\":\"hi\"}"
                                      "{\"content\":\"hi\",\"role\":\"user\"}"));
}

// Semantics the fix must keep (each output is what the pre-fix engine renders).
TEST(minja_macro_semantics_unchanged) {
    // a plain call, positional + keyword + default
    CHECK_EQ(render("{% macro g(a, b='B') %}{{ a }}-{{ b }}{% endmacro %}{{ g(1) }} {{ g(2, b='x') }}"),
             std::string("1-B 2-x"));
    // a default evaluated in the call's context, a global read from the defining scope
    CHECK_EQ(render("{% set v = 3 %}{% macro g(a=v) %}{{ a }}{{ v }}{% endmacro %}{{ g() }}"), std::string("33"));
    // recursion
    CHECK_EQ(render("{% macro f(n) %}{{ n }}{% if n > 0 %}{{ f(n - 1) }}{% endif %}{% endmacro %}{{ f(3) }}"),
             std::string("3210"));
    // call block / caller(), nested
    CHECK_EQ(render("{% macro w(t) %}<{{ t }}>{{ caller() }}</{{ t }}>{% endmacro %}"
                    "{% call w('a') %}{% call w('b') %}x{% endcall %}{% endcall %}"),
             std::string("<a><b>x</b></a>"));
    // a macro passed as a value and called through it
    CHECK_EQ(render("{% macro m(x) %}<{{ x }}>{% endmacro %}{% set h = m %}{{ h(7) }}"), std::string("<7>"));
    // a macro defined in a loop scope, kept in a namespace, called after the loop
    // ended (its defining context outlives the loop for the whole render)
    CHECK_EQ(render("{% set ns = namespace(f=none) %}{% for i in [1, 2] %}"
                    "{% macro m() %}i={{ i }}{% endmacro %}{% set ns.f = m %}{% endfor %}{{ ns.f() }}"),
             std::string("i=2"));
    // a caller() kept by the macro and called after its call block ended
    CHECK_EQ(render("{% set ns = namespace(c=none) %}{% macro w() %}{% set ns.c = caller %}{% endmacro %}"
                    "{% call w() %}late{% endcall %}{{ ns.c() }}"),
             std::string("late"));
}

// After the render that defined it, a macro whose defining context has no other
// owner is gone: calling it is a clear minja error, not a crash. While the
// caller still holds that context it works as before. (The parsed template is
// kept alive here: a macro's callable refers to its node, as it always has.)
TEST(minja_macro_called_after_its_render_errors_cleanly) {
    minja::ArgumentsValue no_args;
    auto in_loop = minja::Parser::parse(
        "{% for i in [1] %}{% macro m() %}y{% endmacro %}{% set box.f = m %}{% endfor %}", {true, true, false});
    auto box = minja::Value::object();
    {
        auto ctx = minja::Context::make(minja::Value::object());
        ctx->set("box", box);
        (void) in_loop->render(ctx);
    }
    bool threw = false;
    try {
        (void) box.get("f").call(minja::Context::make(minja::Value::object()), no_args);
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("called after the render that defined it ended") != std::string::npos;
    }
    CHECK(threw);

    auto at_root = minja::Parser::parse("{% macro m() %}z{% endmacro %}{% set box.f = m %}", {true, true, false});
    auto ctx  = minja::Context::make(minja::Value::object());
    auto kept = minja::Value::object();
    ctx->set("box", kept);
    (void) at_root->render(ctx);   // defined in ctx, which lives on
    CHECK_EQ(kept.get("f").call(minja::Context::make(minja::Value::object()), no_args).get<std::string>(),
             std::string("z"));
}
