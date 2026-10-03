# The libllama engine (0.5.3.1459)

arcint serves without OpenVINO: `--engine llama --gguf FILE` runs the GGUF
through libllama with ggml's OpenCL backend on the Arc card
(`src/exec/backend_llama.cpp`). arcint keeps the HTTP server, the chat
template rendering (the GGUF's own template), the sampler, stop handling and
the lanes; llama.cpp keeps the weights, tokenizer, attention KV and the gated
delta-net state, one sequence per lane.

Build: `-DARCINT_LLAMA=ON -DARCINT_LLAMA_DIR=<llama.cpp tree>`, the tree at
commit `bed0a85` (ggml-org/llama.cpp, 2026-10-02). It builds as a static
subproject with `GGML_OPENCL=ON`, Adreno kernels off, kernels embedded.
`--device GPU.N` is the N-th GPU in OpenCL's enumeration, or a substring of
the device name (`B60`, `A770`).

Served models (`measured-here`, 2026-10-03, one lane, temperature 0, n_ctx
32,768): Qwen3.8-27B Q4_K_M on the B60 and the Qwen3.6 coder (qwen35moe,
184 experts) Q4_K_M on the A770 both answer the capital-of-France check and
score 10/10 on the acceptance task. Decode 9.6 and 7.4 t/s during the task;
ggml's OpenCL kernels are tuned for Adreno, and the OpenVINO path decodes
the dense 27B at ~24 t/s on the same card, so the kernels are the next work.
