#include <minja/chat-template.hpp>
#include "harness.h"

// A runs model with no chat template: the backend passes an empty source rather than a null
// pointer. The minja object's capability read (used by /props and the pre-flight) must not
// crash on that source.
TEST(minja_an_empty_template_source_builds_and_is_safe) {
    minja::chat_template t("", "", "");
    const minja::chat_template_caps c = t.original_caps();   // the capability read must not crash
    (void) c;
}
