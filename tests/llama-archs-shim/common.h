// A stand-in for llama.cpp's common/common.h, for building the pinned tree's
// tests/test-llama-archs.cpp inside arcint's unit tests (see log.h beside it).
// common_batch here covers what that test uses: token entries with one position.
#pragma once

#include "llama.h"
#include "llama-cpp.h"

#include <stdexcept>
#include <string>
#include <vector>

inline void common_init() {}

// the tensors of the MoE experts, as common/common.h names them
const char * const LLM_FFN_EXPS_REGEX = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";

struct common_batch {
    llama_batch_ext_ptr batch;
    int32_t n = 0;

    common_batch() = default;
    explicit common_batch(struct llama_context * ctx) : batch(llama_batch_ext_init(ctx)) {}

    llama_batch_ext * get() { return batch.get(); }

    int32_t size() const { return n; }

    void clear() {
        llama_batch_ext_clear(batch.get());
        n = 0;
    }

    // returns the batch index
    int32_t add(llama_token id, llama_pos pos, llama_seq_id seq_id, bool output) {
        const int32_t idx = llama_batch_ext_add_token(batch.get(), seq_id, id);
        if (idx < 0) {
            throw std::runtime_error("common_batch (shim): the batch is full");
        }
        const llama_pos p[4] = { pos, 0, 0, 0 }; // as common_batch::add: only pos[0] is used for text tokens
        llama_batch_ext_set_pos(batch.get(), idx, p);
        llama_batch_ext_set_output_logits(batch.get(), idx, output);
        n++;
        return idx;
    }
};
