"""MSA indexer context parallelism: scorers, selection and candidate transport.

The serving framework owns the KV writer, metadata, cache lifetime and phase
policy. Tensor APIs are lazy; native carrier/scorer ABIs remain independently
versioned. ``ICP_INTEGRATION_ABI`` identifies this additive Python surface.
"""

from __future__ import annotations

from . import abi
from ._build import arch, gencode_flags, loaded_artifacts, ptxas_info

__version__ = "0.1.1"

# Python integration surface, independent of the native score/plan/carrier ABIs.
# Consumers still check those ABIs against the extensions they actually load.
ICP_INTEGRATION_ABI = 1
VLLM_INTEGRATION_ABI = ICP_INTEGRATION_ABI
# Public vLLM head-slot layout, calibrated prefill and isolated Q8KV4 prewarm.
# Native score/plan/carrier layouts are unchanged.
PUBLIC_VLLM_ABI = 1

#: Public name -> the submodule that defines it. Resolved on first access, so
#: importing this package costs no torch import: `_build` answers arch and
#: gencode questions without one, `abi` is pure Python, and a host with no torch
#: can still run the build gate, the ABI self-check and the pure-CPU tests.
_LAZY = {
    "CandidateExchange": "candidate_exchange",
    "IcpExchange": "exchange",
    "IcpTiledExchange": "tiled_exchange",
    "IcpFusedExchange": "fused_exchange",
    "MaskPreset": "exchange",
    "exchange_and_merge": "exchange",
    "close_all": "exchange",
    "opt_bits": "exchange",
    "merge_candidates": "merge",
    "forced_rows": "merge",
    "IcpMergeError": "merge",
    "MergeArm": "merge",
    "merge_backend": "merge",
    "last_merge_arm": "merge",
    "IcpReferenceArmWarning": "merge",
    "PRODUCTION_ARMS": "merge",
    "REFERENCE_ARMS": "merge",
    "MERGE_ARM_NAMES": "merge",
    "MERGE_ARM_ENV": "merge",
    "merge_candidates_torch": "merge_reference",
    "merge_candidates_triton": "merge_reference",
    "ICP_STATUS_OK": "merge",
    "ICP_STATUS_NAN": "merge",
    "ICP_STATUS_ROW_META": "merge",
    "ICP_NO_FORCED_BLOCK": "merge",
    "canonical_key_reference": "merge",
    "canonical_key_cuda": "merge",
    "KEY_BIAS": "merge",
    "TRANSPORTS": "carrier",
    "pack_send_carrier": "carrier",
    "exchange_carrier": "carrier",
    "invalid_carrier": "carrier",
    "send_carrier_shape": "carrier",
    "recv_carrier_shape": "carrier",
    "peer_split_int32_elements": "carrier",
    "collective_exchange_and_merge": "carrier",
    "CarrierWorkspace": "carrier",
    "allocate_carrier_workspace": "carrier",
    "live_blocks": "candidates",
    "SelectArm": "candidates",
    "SHIPPED_ARM": "candidates",
    "MEASUREMENT_ARMS": "candidates",
    "selector_arm": "candidates",
    "ladder_items": "candidates",
    "prefill_launch": "candidates",
    "prefill_scan_coverage": "candidates",
    "PrefillLaunch": "candidates",
    "live_local_blocks": "candidates",
    "SelectorPlan": "candidates",
    "PrefillPlan": "candidates",
    "DecodePlan": "candidates",
    "CandidateGeometry": "candidates",
    "CANDIDATE_MAPPING_ABI_VERSION": "candidates",
    "SUPPORTED_GLOBAL_BLOCK_STRIDES": "candidates",
    "CandidateWorkspace": "candidates",
    "allocate_workspace": "candidates",
    "select_prefill_candidates": "candidates",
    "select_decode_candidates": "candidates",
    "select_with_control": "candidates",
    "planned_kernel": "candidates",
    "planned_prefill_kernel": "candidates",
    "record_launches": "candidates",
    "last_launch": "candidates",
}

_SUBMODULES = frozenset({"abi", "candidates", "carrier", "exchange", "merge",
                         "merge_reference", "tiled_exchange",
                         "fused_exchange", "candidate_exchange", "exchange_plan",
                         "local_indexer"})

__all__ = [
    *sorted(_LAZY),
    "abi",
    "arch",
    "gencode_flags",
    "loaded_artifacts",
    "ptxas_info",
    "__version__",
    "VLLM_INTEGRATION_ABI",
    "ICP_INTEGRATION_ABI",
    "PUBLIC_VLLM_ABI",
]


def __getattr__(name: str):
    from importlib import import_module  # noqa: PLC0415

    if name in _SUBMODULES:
        return import_module(f".{name}", __name__)
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*__all__, *_SUBMODULES, *globals()})
