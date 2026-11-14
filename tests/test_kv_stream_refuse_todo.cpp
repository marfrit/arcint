// TODO(I5): the KV stream's start refusal -- the cell this file becomes.
//
// A paged FLASH_ATTN_EXT carries a page table and a sync tag. Patch 0119's CPU
// backend takes it, and the OpenCL backend declines it (its kernels learn the
// page table in the design's I3). ggml's scheduler responds to a declined op by
// placing the node on the CPU reference, which is correct and is exactly what
// I1's byte-equality cell runs on -- but in a served session it would be a
// silent collapse to one host thread gathering every page.
//
// I5 stops that at the start: the backend probes one FLASH_ATTN_EXT with a
// 1-page table through ggml_backend_supports_op and refuses
// --llama-kv-resident with a message when the paged decode shape is not
// supported, before a model loads. This file is the disabled marker for that
// cell (docs/campaigns/kv-streaming-design.md section 4.3):
//  - green: a build whose OpenCL supports_op returns false for pages refuses
//    the flag at start, the message names the missing kernel/patch, and no
//    context is created;
//  - red: the same build with the refusal check removed accepts the flag and
//    then serves through the CPU reference -- the arm that must fail.
// Nothing can run here yet: the refusal, the probe and the message do not exist
// (I5 builds them). Until then ctest reports this case as DISABLED, and the
// TODO line below is the only output a manual run prints.

#include <cstdio>

int main() {
    std::printf("TODO(I5): the start refusal for a paged FLASH_ATTN_EXT on a backend without paged FA\n");
    return 0;
}
