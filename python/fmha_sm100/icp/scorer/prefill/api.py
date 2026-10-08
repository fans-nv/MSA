"""Compatibility path to the shared MSA api module."""
import sys
from fmha_sm100 import api as _implementation
sys.modules[__name__] = _implementation
