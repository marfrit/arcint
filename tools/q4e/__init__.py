# q4e: arcint-original graph emission for the Qwen Flash Next (qwen4_exp)
# architecture, built against the pinned transformers reference
# (see tools/export_qwen4_exp.py REFERENCE_COMMIT).
#
# Each subgraph module emits one architectural block as opset-13 ops and is
# validated against the torch reference (CPU + GPU, max-abs + KLD) by a parity
# test under tests/python/. Validation now spans the five block modules below --
#   gdn      -> tests/python/test_gdn_block.py
#   hc       -> tests/python/test_hc_block.py  (+ test_hc_combine_block.py)
#   moe      -> tests/python/test_moe_block.py
#   ple      -> tests/python/test_ple_block.py
#   backbone -> tests/python/test_backbone.py
# -- each carrying a transcription-vs-pin 0.0 leg plus the OV-parity legs.

from . import gdn
from . import hc
from . import moe
from . import ple
from . import backbone

__all__ = ["gdn", "hc", "moe", "ple", "backbone"]
