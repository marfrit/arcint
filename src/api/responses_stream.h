#pragma once

#include "api/handlers.h"
#include "api/responses.h"

// The Open Responses SSE writer (docs/campaigns/feyntalk.md W2): one event per
// call, `event:` equal to the body's `type`, sequence_number from 1 without
// gaps, the item/content-part lifecycle for reasoning, message and
// function_call, and the terminal event before `[DONE]`.
namespace lgc::api {

// Streams one response to `write`. Returns when the stream ends, the client is
// gone, or the process is stopping; a gone client gets no further bytes (the
// probe already stops the generation at its next boundary).
void stream_response(const Context& ctx, const PreparedResponse& prep, int slot,
                     const SseWriter& write);

}  // namespace lgc::api
