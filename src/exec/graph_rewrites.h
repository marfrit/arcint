#pragma once

// Graph rewrites the backend applies to a loaded model before compiling it,
// exposed for their unit tests (the served check is the reservation probe's
// verification against a real forward, DESIGN §7.0.2e).

#include <cstdint>
#include <memory>

#include <openvino/core/model.hpp>

namespace lgc {

// Walks back from the first Result through Convert/Reshape to the MatMul or
// FullyConnectedKQuant that is the LM head's projection. Returns the node,
// or nullptr when the head is not unmistakably one of those two within 8 hops.
std::shared_ptr<ov::Node> find_projection_head(const std::shared_ptr<ov::Model>& model);

// Slices the LM head's input to its last `keep_rows` rows along `token_axis`
// (-1: rank - 2, the dense export's token axis; 0: the paged export's), so a
// prefill computes and copies `keep_rows` rows of logits instead of one per
// prompt token. The head is the MatMul the walk from the first Result reaches
// through Convert/Reshape only -- or the FullyConnectedKQuant a GGUF-opened
// model has in its place (exec/kquant_op.h). Returns false, and leaves the
// model untouched, when the head is not unmistakably one of those two.
bool slice_logits_to_last_token(const std::shared_ptr<ov::Model>& model, int64_t keep_rows, int64_t token_axis);

// Publishes the base model's final hidden state -- the LM head's activation
// input -- as a second output named "hidden_states", so the MTP head can be
// primed on a prompt instead of seeing only the rows the logits slice keeps
// (backend_ov.cpp: this has to run before that slice). The head is the same
// terminator slice_logits_to_last_token walks to: the MatMul the walk from
// the first Result reaches through Convert/Reshape only, or the
// FullyConnectedKQuant a GGUF-opened model has in its place when
// output.weight stays in the file's rows (exec/kquant_op.h). Returns false,
// and leaves the model untouched, when the head is not unmistakably one of
// those two within 8 hops -- --mtp on then has no hidden state to prime the
// head with (log tag "mtp").
bool expose_hidden_state(const std::shared_ptr<ov::Model>& model);

}  // namespace lgc
