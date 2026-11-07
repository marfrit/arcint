# arcint: the patched llama.cpp, flattened

This branch of [arcint](https://github.com/marfrit/arcint) holds llama.cpp at the
commit arcint pins, with arcint's patch series applied, so the code can be read
and searched as files. This is v1.6.1: 26 patches (the series was folded after 1.6.0: `contrib/llama.cpp/patches/FOLDED.md` on `main` maps the old numbers).

- The first commit is the unmodified pin; each later commit is one release. The
  diff between the first commit and any other is what arcint changes in llama.cpp.
- Most of the work is the OpenCL backend for Intel Arc:
  `ggml/src/ggml-opencl/ggml-opencl.cpp` and `ggml/src/ggml-opencl/kernels/`.
- Nothing builds from this branch. arcint builds from the pin plus
  `contrib/llama.cpp/patches` on `main`, and `contrib/llama.cpp/README.md` there
  has each patch's measurements.
- `.github/` is left out of this tree, so that llama.cpp's workflows do not run here.
- llama.cpp is MIT licensed; see `LICENSE`. The patches carry arcint's license.
