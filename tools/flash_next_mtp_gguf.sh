#!/bin/bash
# The MTP-only GGUF that --llama-mtp-gguf takes for Qwen3.8-Flash-Next (docs/llama-engine.md, "Flash-Next").
# Flash-Next's GGUFs carry no MTP layer; the checkpoint has it as 31 mtp.* tensors. This builds a stub HF
# directory (config and tokenizer of the pinned revision, the mtp.* tensors, the embedding and the LM head,
# about 7.3 GB fetched by range requests), converts it with the pinned llama.cpp's converter (--mtp, BF16),
# and quantizes it as served: Q8_0 dense, Q4_K gate/up experts, IQ4_NL down experts, the head and the
# embedding Q6_K (mtp-flash-next-q8e4n.gguf, 2.4 GiB).
#
# usage: tools/flash_next_mtp_gguf.sh <llama.cpp tree (the pin with contrib/llama.cpp/patches)> <out dir>
#   PYTHON (default python3) needs safetensors, numpy, torch-free gguf-py deps of the converter;
#   QUANTIZE (default <tree>/build/bin/llama-quantize).
set -euo pipefail
LLAMA=${1:?llama.cpp tree}; OUT=${2:?output directory}
PY=${PYTHON:-python3}
QUANTIZE=${QUANTIZE:-$LLAMA/build/bin/llama-quantize}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=Qwen/Qwen3.8-Flash-Next
REV=de4b8e4d43b917e7706784d8bb445c9af86a3540
HF=$OUT/hf
mkdir -p "$HF"
cd "$HF"
for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt \
         chat_template.jinja preprocessor_config.json video_preprocessor_config.json; do
    [ -s "$f" ] || curl -sSfL --retry 5 -o "$f" "https://huggingface.co/$REPO/resolve/$REV/$f"
done
[ -s mtp_head.safetensors ] || "$PY" "$HERE/fetch_safetensors_tensors.py" --repo "$REPO" --revision "$REV" \
    --prefix 'mtp.' --out "$HF/stage-mtp" --assemble mtp_head.safetensors
[ -s mtp_head.safetensors ] || mv "$HF/stage-mtp/mtp_head.safetensors" .
[ -s emb_head.safetensors ] || "$PY" "$HERE/fetch_safetensors_tensors.py" --repo "$REPO" --revision "$REV" \
    --regex '^(lm_head\.weight|model\.language_model\.embed_tokens\.weight)$' --out "$HF/stage-emb" \
    --assemble emb_head.safetensors
[ -s emb_head.safetensors ] || mv "$HF/stage-emb/emb_head.safetensors" .
"$PY" - <<'PY'
import json, struct
idx = {"metadata": {}, "weight_map": {}}
for f in ("mtp_head.safetensors", "emb_head.safetensors"):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    for k in h:
        if k != "__metadata__":
            idx["weight_map"][k] = f
json.dump(idx, open("model.safetensors.index.json", "w"), indent=1)
print(len(idx["weight_map"]), "tensors indexed")
PY
cd "$LLAMA"
"$PY" convert_hf_to_gguf.py "$HF" --mtp --outtype bf16 --outfile "$OUT/mtp-flash-next-bf16.gguf"
"$QUANTIZE" --output-tensor-type q6_K --token-embedding-type q6_K --tensor-type ffn_gate_exps=q4_K --tensor-type ffn_up_exps=q4_K --tensor-type ffn_down_exps=iq4_nl \
    "$OUT/mtp-flash-next-bf16.gguf" "$OUT/mtp-flash-next-q8e4n.gguf" q8_0
ls -la "$OUT"
