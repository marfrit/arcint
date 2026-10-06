# llama-eval-dump — whole tensors out of a llama.cpp forward

A sibling of llama.cpp's `examples/eval-callback` that writes named graph
tensors as raw f32 instead of printing three-element corners. It is the
reference side of `tools/llama_tap_compare.py`: the same GGUF, the same
token ids, every intermediate of llama.cpp's own graph (`hc_mixed-0`,
`attn_output-0`, `l_last-3`, …) as a file the emitter's cut dumps and the
real-geometry reference (`tools/ref_forward_real.py`) can be compared with,
element for element. It is how the three Flash-Next fill defects were found
and the depth-4 re-export accepted (DESIGN §7.0.2bz; the campaign record is
on the development branch, `qfndev`).

Build inside a llama.cpp checkout (the pinned one the KLD capture came
from), as one more example:

    cp -r contrib/llama-eval-dump <llama.cpp>/examples/eval-dump
    echo 'add_subdirectory(eval-dump)' >> <llama.cpp>/examples/CMakeLists.txt
    cmake -B build && cmake --build build --target llama-eval-dump

Run with the names to keep and a directory to write into:

    LLAMA_DUMP_DIR=out LLAMA_DUMP_NAMES=model.input_embed,l_last-0,l_last-3 \
      llama-eval-dump -m model.gguf -p "The capital of France is" -t 8 -c 512

`out/index.txt` lists `<name>#<k> ne0 ne1 ne2 ne3 type` per dumped tensor
(`k` = the k-th occurrence of that name in the graph; a layer's two
hyper-connection mixes share a name), `out/<name>#<k>.f32` holds the data
with ne0 fastest — numpy shape `(ne3, ne2, ne1, ne0)`. Non-contiguous
tensors (views) are skipped and said so in the index; dump the op that
produced them instead.
