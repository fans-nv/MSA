"""CuTe DSL decode scorer for refined-icp-v1 (TP=ICP2, page 128, fragment 64).

The implementation is :mod:`.icp_decode_score`; importing this package does not
import CUTLASS DSL or torch.
"""

_SUBMODULES = frozenset({"icp_decode_score"})
_LAZY = frozenset({
    "ICP_DECODE_SCORE_ABI_VERSION",
    "ICP_DECODE_SCORE_LAUNCH_ABI_VERSION",
    "DEFAULT_NUM_STAGES",
    "auto_split_k",
    "get_icp_decode_scorer",
    "icp_decode_score",
    "supports_icp_decode_score",
})

__all__ = sorted(_LAZY)


def __getattr__(name):
    from importlib import import_module  # noqa: PLC0415

    module = import_module(f"{__name__}.icp_decode_score")
    if name in _SUBMODULES:
        return module
    if name in _LAZY:
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *__all__})
