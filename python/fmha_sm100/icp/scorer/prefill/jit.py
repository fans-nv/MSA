"""Compatibility path to the shared MSA jit module."""
import sys
from fmha_sm100 import jit as _implementation
sys.modules[__name__] = _implementation
