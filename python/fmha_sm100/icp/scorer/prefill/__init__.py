"""OnlyScore / OnlyScoreIcp FMHA prefill scorer (MSA-derived), JIT per variant.

``api`` is the entry point (``_fmha_sm100_plan``, ``_fmha_sm100``,
``sparse_topk_select``, ``IcpDevicePlanStore``, ``prewarm_icp_scorer``);
``jit`` owns the variant table, nvcc flags and the cache. Importing this
package imports neither: ``api`` imports torch and numpy.
"""

_SUBMODULES = frozenset({"api", "jit"})

__all__ = sorted(_SUBMODULES)


def __getattr__(name):
    from importlib import import_module  # noqa: PLC0415

    if name in _SUBMODULES:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *__all__})
