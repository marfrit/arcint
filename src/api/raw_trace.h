// The raw-generation trace: one JSON line a chat or Responses turn, off by
// default. A diagnostic for tool calls that come back wrong: it records what
// the model wrote before anything was split or parsed, next to what the
// parser made of it, so a malformed call body is told apart from a parser
// defect on served traffic (--tool-grammar on against off, MTP or not).
//
// ARCINT_RAW_TRACE=<file> writes the lines; unset (or empty) is OFF and costs
// one getenv a turn: nothing is built, copied or opened. The file opens once a
// process, on the first traced turn, and every line is appended and flushed;
// clearing or changing the variable later stops the writes or leaves them on
// the open file, as ARCINT_MTP_TRACE does (src/exec/mtp_trace.h).
//
// One line a turn, written after the turn ended (format `raw1`):
//
//   {"trace":"raw1","n":<turns traced by this process, 1, 2, ...>,
//    "request_id":<X-Request-Id>,"route":"chat"|"responses","stream":<bool>,
//    "token_mode":<bool>,"finish":<finish reason>,
//    "think_start":"undecided"|"open"|"closed",
//    "raw":<the generated text: the pieces the reasoning split received, in
//           order, concatenated exactly>,
//    "reasoning":<the split's reasoning>,"tool_parse":<bool>,
//    "parsed_input":<the text handed to the tool-call parser (the split's
//                    content), null when no parse ran>,
//    "tool_calls":[{"name":..,"arguments":<the arguments string the parser
//                   produced>}],
//    "content":<the content that remained after the split and the parse>,
//    "error":<null, or the message of an exception that ended the turn>}
//
// An aborted turn (client gone, stop signal, cancel) still gets its line, with
// the text it had and no parse; so does a turn that threw, with finish
// "error" and the exception's message. Bytes that are not UTF-8 (a turn cut
// inside a code point) are written as U+FFFD: the line must stay JSON.
//
// Several processes may share one file: each line goes out as one write() on
// an O_APPEND descriptor, so lines do not interleave inside each other (a
// regular file on Linux). `n` counts per process; `request_id` tells turns
// apart across processes.
#pragma once

#include <string>
#include <vector>

#include "core/reasoning.h"
#include "core/toolcall.h"

namespace lgc::api {

struct RawTraceRecord {
    std::string                  route;          // "chat" or "responses"
    std::string                  request_id;
    bool                         stream     = false;
    bool                         token_mode = false;
    std::string                  finish;
    ThinkStart                   think_start;
    const std::string*           raw        = nullptr;   // the turn's generated text
    std::string                  reasoning;
    bool                         tool_parse = false;
    const std::vector<ToolCall>* calls      = nullptr;
    std::string                  content;
    const std::string*           parsed_input = nullptr;   // the parser's input; null: no parse
    std::string                  error;                    // empty: the turn did not throw
};

// ARCINT_RAW_TRACE=<file> set and not empty. Read per call.
bool raw_trace_on();

// The line without its newline, `n` the turn counter.
std::string raw_trace_line(const RawTraceRecord& r, unsigned long long n);

// Appends the turn's line to ARCINT_RAW_TRACE's file (opened once a process,
// O_APPEND, created with mode 0600: it holds generated text) with one write().
// Off: returns at once. The line is built before the writer's lock is taken.
// A write that fails or writes nothing ends the trace for the process,
// reported once on stderr like a failed open. It does not throw: a line that
// cannot be built is dropped, so a call site outside a try block cannot turn
// a finished turn into an error.
void raw_trace_write(const RawTraceRecord& r) noexcept;

// The record of a turn of `prep` on `route`; the caller fills the split and
// the parse when the turn got that far.
struct PreparedChat;
RawTraceRecord raw_trace_record(const PreparedChat& prep, const char* route, bool stream, const char* finish,
                                const std::string& raw);

// A turn that threw: its partial text and the exception's message, finish
// "error". Off: returns at once. Called from error paths, so it never throws:
// a failure of the trace itself (an allocation while dumping a large text) is
// swallowed and the error response stays the one the route builds.
void raw_trace_error(const PreparedChat& prep, const char* route, bool stream, const std::string& raw,
                     const char* message) noexcept;

}  // namespace lgc::api
