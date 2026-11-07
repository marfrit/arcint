// An MTP head file (--llama-mtp-gguf) made without `output.weight` (llama.cpp
// patch 0105; tools/coder_mtp_gguf.py strip-output) has no LM head: the MTP
// context can only return the nextn rows, which the draft vocabulary
// (--llama-mtp-vocab) scores against rows read from the served file
// (DraftHead). Without the vocabulary the context would ask for logits and
// find nothing to compute them with, so the engine refuses at start. Pure, so
// it is testable without a model (tests/test_mtp_head_file.cpp).
#pragma once

#include <string>
#include <vector>

namespace lgc {

struct MtpHeadFile {
    std::string arch;               // general.architecture
    bool has_output      = false;   // output.weight
    bool has_shared_head = false;   // blk.<n>.nextn.shared_head_head.weight
};

inline bool mtp_name_ends_with(const std::string& s, const char* suffix) {
    const std::string x(suffix);
    return s.size() >= x.size() && s.compare(s.size() - x.size(), x.size(), x) == 0;
}

// What the file's tensor names say about its LM head.
inline MtpHeadFile mtp_head_file_from_names(const std::vector<std::string>& names) {
    MtpHeadFile f;
    for (const std::string& n : names) {
        if (n == "output.weight") f.has_output = true;
        if (mtp_name_ends_with(n, ".nextn.shared_head_head.weight")) f.has_shared_head = true;
    }
    return f;
}

// "" when the head file may be served with or without a draft vocabulary; else why not.
// Patch 0105 changes qwen35moe's loader only: other architectures keep what they did
// (a head file without output.weight loads there as before, by their own fallback).
inline std::string mtp_head_file_refusal(const MtpHeadFile& f, bool have_vocab) {
    if (f.arch != "qwen35moe" || f.has_output || f.has_shared_head || have_vocab) return {};
    return "the file has no output.weight (and no nextn.shared_head_head.weight): without --llama-mtp-vocab "
           "the MTP context needs logits and has no LM head to compute them; set --llama-mtp-vocab, or use "
           "a head file with its output.weight";
}

}  // namespace lgc
