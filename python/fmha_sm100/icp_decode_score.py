"""Compatibility import for the canonical MSA ICP decode scorer."""
import sys
from .icp.scorer.decode import icp_decode_score as _implementation
sys.modules[__name__] = _implementation
