# q4e: arcint-original graph emission for the Qwen Flash Next (qwen4_exp)
# architecture, built against the pinned transformers reference
# (see tools/export_qwen4_exp.py REFERENCE_COMMIT).
#
# Each subgraph module emits one architectural block as opset-13 ops and is
# validated against the torch reference by the test in
# tests/python/test_gdn_block.py (CPU + GPU, max-abs + KLD).

from . import gdn
from . import hc
from . import moe

__all__ = ["gdn", "hc", "moe"]
