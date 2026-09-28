"""tools/export_serving_artifact.py, device-free: the config.json it writes
carries every key `src/core/artifact.cpp` reads for a Flash-Next artifact, the
vocab comparison refuses a tokenizer that disagrees with the GGUF's tokens, and
the embedding model it saves beside the language model is DYNAMIC in T (the
served `embed_paged` feeds [1, n] for every chunk and every decode step).

Red first, each: the loader-key cell was written against a config with the
n-gram five missing (red), the mismatch cell against a comparison that returned
an empty list for a swapped vocab (red), the embed cell against
`pwe.build_embed_piece` (static in T -- a second T refuses at set_input_tensor).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import export_serving_artifact as esa  # noqa: E402

# artifact.cpp `load_artifact` + `admit_ngram_table_from_disk` +
# backend_ov.cpp `bind_ngram_ports`: the keys read by name.
LOADER_KEYS = (
    "model_type", "architectures", "num_hidden_layers", "hidden_size",
    "max_position_embeddings", "num_experts", "full_attention_interval",
    "layer_types", "ngram_size", "ngram_vocab_size_base", "heads_per_ngram",
    "ple_embed_dim", "ple_layer_ids", "vocab_size", "eos_token_id",
)


def test_the_config_carries_every_key_the_loader_reads_at_depth_4():
    cfg = esa.serving_config(4, ple_eos_token_id=248044)
    for k in LOADER_KEYS:
        assert k in cfg, k
    assert cfg["num_hidden_layers"] == 4
    assert len(cfg["layer_types"]) == 4
    # build_serving_shape_ir: layer i % 4 == 3 is the full-attention layer
    assert cfg["layer_types"] == ["linear_attention"] * 3 + ["qwen_sparse_attention"]
    assert cfg["ple_layer_ids"] == [2]                # 1-based; decoder layer 1
    assert cfg["eos_token_id"] == 248044
    assert cfg["ngram_size"] == 3 and cfg["heads_per_ngram"] == 8
    assert cfg["ngram_vocab_size_base"] == 20_000_000
    assert cfg["ple_embed_dim"] == 2560 and cfg["vocab_size"] == 248320
    assert cfg["model_type"] == "qwen4_exp"
    # the GDN's output gate: sigmoid for this checkpoint (llama.cpp hard-codes
    # it for the architecture; the pin defaults to hidden_act = silu when the
    # key is absent, which is how the first artifacts were exported)
    assert cfg["output_gate_type"] == "sigmoid"
    assert cfg["gdn_key_head_map"] == "tiled"
    json.dumps(cfg)                                    # serialisable as written


def test_the_config_at_full_depth_keeps_the_real_layer_count():
    cfg = esa.serving_config(48, ple_eos_token_id=248044)
    assert cfg["num_hidden_layers"] == 48
    assert cfg["layer_types"].count("qwen_sparse_attention") == 12
    assert cfg["layer_types"].count("linear_attention") == 36


def test_a_depth_without_the_ple_layer_is_refused():
    with pytest.raises(ValueError):
        esa.serving_config(1, ple_eos_token_id=248044)   # layer 1 is not built
    with pytest.raises(ValueError):
        esa.serving_config(49, ple_eos_token_id=248044)


def test_the_vocab_comparison_refuses_a_swapped_token(tmp_path):
    toks = ["a", "b", "c", "<eos>", "pad0", "pad1"]
    hf = {"model": {"vocab": {"a": 0, "b": 1, "c": 2}},
          "added_tokens": [{"id": 3, "content": "<eos>"}]}
    p = tmp_path / "tokenizer.json"
    p.write_text(json.dumps(hf))
    bad, n = esa.vocab_mismatches(p, toks)
    assert bad == [] and n == 4                        # the padded tail is not a mismatch
    hf["model"]["vocab"] = {"a": 0, "c": 1, "b": 2}
    p.write_text(json.dumps(hf))
    bad, n = esa.vocab_mismatches(p, toks)
    assert sorted(i for i, _, _ in bad) == [1, 2]


def test_the_embedding_model_is_dynamic_in_t():
    ov = pytest.importorskip("openvino")
    table = np.arange(7 * 3, dtype=np.float32).reshape(7, 3)
    model = esa.build_embed_model(table)
    assert [str(p.get_any_name()) for p in model.inputs] == ["input_ids"]
    assert model.inputs[0].get_partial_shape().is_dynamic
    comp = ov.Core().compile_model(model, "CPU")
    req = comp.create_infer_request()
    for ids in ([4, 1, 6], [2], [0, 0, 5, 3, 1]):
        req.set_input_tensor(ov.Tensor(np.array([ids], dtype=np.int64)))
        req.infer()
        out = req.get_output_tensor(0).data
        assert out.shape == (1, len(ids), 3)
        np.testing.assert_array_equal(out[0], table[ids])


def test_segment_ranges_cover_the_depth_and_refuse_a_segment_without_attention():
    """SEGMENTED (0.5.1): 4 x 12 over 48, a short tail when the depth is not a
    multiple, None = one segment; a segment length that is not a multiple of
    4 or a tail without an attention layer is refused by name."""
    assert esa.segment_ranges(48, 12) == [(0, 12), (12, 24), (24, 36), (36, 48)]
    assert esa.segment_ranges(48, 8) == [(0, 8), (8, 16), (16, 24), (24, 32), (32, 40), (40, 48)]
    assert esa.segment_ranges(12, 8) == [(0, 8), (8, 12)]
    assert esa.segment_ranges(12, None) == [(0, 12)]
    with pytest.raises(ValueError, match="multiple of 4"):
        esa.segment_ranges(48, 6)
    with pytest.raises(ValueError, match="no full-attention"):
        esa.segment_ranges(14, 12)          # tail (12, 14) has no index 3 mod 4


def test_a_qwen4_exp_export_stages_the_ngram_table_unless_pinned():
    """Since 2026-09-27 the export declares the n-gram table as a per-forward
    staging window by default (DESIGN 7.0.2cz: without per-expert dispatch the
    staged twin gives the pinned twin's greedy digests and decodes 1.6-2.7x
    faster, with 2.9 MiB of staging per lane instead of a 26.82 GiB pin).
    `--ngram-pinned` keeps the whole-table ports for A/B twins.

    RED before the default changed: the parser's default was None (pinned).
    """
    base = ["--layers", "4", "--shards", "x", "--tokenizer-from", "y", "--out", "z"]
    parse = esa.build_parser().parse_args
    assert esa.ngram_staging_rows(parse(base)) == esa.NGRAM_STAGING_ROWS_DEFAULT == 33600
    assert esa.ngram_staging_rows(parse(base + ["--ngram-pinned"])) is None
    assert esa.ngram_staging_rows(parse(base + ["--ngram-staging-rows", "8192"])) == 8192
    # "not given" stays distinguishable, so qwen35moe can refuse an explicit value
    assert parse(base).ngram_staging_rows is None
    # the bound covers the served prefill chunk (2,048 tokens) x 16 n-gram heads
    assert esa.NGRAM_STAGING_ROWS_DEFAULT >= 2048 * 16


def test_qsa_is_off_by_default_and_refused_for_qwen35moe():
    """Campaign qsa, step 3: `--qsa` defaults off so every existing artifact
    and its arch hash are untouched, and the flag carries the Flash-Next
    indexer into the export. qwen35moe has no indexer, so an explicit --qsa
    with that family is refused before any shard is read.

    RED before the flag existed: the parser had no `qsa` attribute.
    """
    base = ["--layers", "4", "--shards", "x", "--tokenizer-from", "y", "--out", "z"]
    parse = esa.build_parser().parse_args
    assert parse(base).qsa is False
    assert parse(base + ["--qsa"]).qsa is True
    with pytest.raises(SystemExit):
        esa.main(base + ["--qsa", "--family", "qwen35moe"])
