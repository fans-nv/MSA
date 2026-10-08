"""The `refined-icp-v1` ABI, vendored as an **executable** artifact.

:mod:`fmha_sm100.icp.abi.refined_icp_v1` is a byte-for-byte copy of the frozen ABI
module (sha256 ``37af7487b14bc5a92bfa643d482bc7bae4aeab5917c4677ff466bfc223e751fd``).
It is pure Python -- no torch, no CUDA, no device -- and carries 10,571
self-checks that run with ``python -m fmha_sm100.icp.abi``.

It is vendored rather than restated in prose for one reason: a consumer that
re-derives a constant drifts from it, and the drift is silent. Import the
constants::

    from fmha_sm100.icp import abi
    abi.R(4)                   # 32 rows of every block per rank at W=4
    abi.owner_rank(u, W)       # which rank holds token offset u
    abi.forced_block_id(p)     # p // 128
    abi.valid_selection_count(p)

Three version strings, and they move independently
--------------------------------------------------

``ABI_VERSION`` (``refined-icp-v1.abi.1``)
    The contract: geometry, placement, the carrier's layout, the key, the output
    shape. Changing a value here is a contract amendment, not a refactor.
``K2_ABI_VERSION`` (``refined-icp-v1.k2.2``)
    The K2 **host** entry point's arity and argument meaning. Moved 5 -> 8 when
    the C3 forced-row planes and the C5 status word were added.
``K2_CARRIER_VERSION`` (``refined-icp-v1.C4``)
    The candidate carrier's dtype and axis meaning. Moved when the fp32
    ``[C, T, H_group, 16, 2]`` gathered tensor was replaced by the int32
    ``[W, Qchunk, H_local, 16, 2]`` head-directed carrier.

The last two are **read back off the built extension**, not asserted from here,
so they describe the object that was actually loaded. A caller checking only the
host ABI version would happily hand a ``k2.2`` build the old gathered tensor;
:func:`capabilities` returns all three so that cannot happen quietly.
"""

from __future__ import annotations

from .refined_icp_v1 import *  # noqa: F401,F403
from .refined_icp_v1 import (
    ABI_VERSION,
    CONTRACT_FAMILY,
    FROZEN_DATE,
    SOURCE_BASELINE_COMMIT,
    abi_digest,
    self_check,
    to_json,
)

#: The digest of the frozen ABI, recomputed in-process by :func:`abi_digest`.
#: Asserting the recomputed value against this constant is what makes the
#: vendoring checkable rather than declared.
ABI_DIGEST = "69dfb5aedb5a2a2c10d1e92304f32038072fd42064b31759e36dab0e81a3733d"

#: The host entry point's arity contract, mirrored from ``k2_merge.cu``'s
#: ``m.attr("k2_abi_version")``.
K2_ABI_VERSION = "refined-icp-v1.k2.2"

#: The candidate carrier's dtype/axis contract, mirrored from
#: ``m.attr("k2_carrier")``.
K2_CARRIER_VERSION = "refined-icp-v1.C4"


def capabilities() -> dict[str, str]:
    """The three version strings of the extension **that is actually loaded**.

    Builds K2 if it is not built yet, so this is a device-touching call. The
    values come from the module's ``m.attr`` entries, not from the constants
    above, which is the point: a stale build reports its own version and the
    mismatch is visible instead of assumed.

    A consumer's import-time assertion is one line::

        from fmha_sm100.icp import abi
        abi.assert_compatible()
    """
    from .. import _build  # noqa: PLC0415

    module = _build._k2()
    return {
        "abi": ABI_VERSION,
        "abi_digest": abi_digest(),
        "k2_abi": getattr(module, "k2_abi_version", "<absent>"),
        "k2_carrier": getattr(module, "k2_carrier", "<absent>"),
    }


def assert_compatible() -> dict[str, str]:
    """Refuse to proceed against a build that is not ``refined-icp-v1``.

    Returns the capability dict on success so a caller can log exactly what it
    validated against. Raises :class:`RuntimeError`, naming the mismatch, on
    failure -- including the two failure modes that would otherwise be silent:
    a ``k2.1`` build (no status word, so a NaN candidate wins the merge) and a
    build with no ``k2_carrier`` attribute (the pre-C4 fp32 gathered tensor,
    whose axis 2 is ``H_group`` rather than ``H_local``).
    """
    caps = capabilities()
    problems = []
    if caps["abi_digest"] != ABI_DIGEST:
        problems.append(
            f"the vendored ABI digest is {caps['abi_digest']}, expected "
            f"{ABI_DIGEST}: fmha_sm100.icp/abi/refined_icp_v1.py has been edited "
            "without bumping ABI_VERSION"
        )
    if caps["k2_abi"] != K2_ABI_VERSION:
        problems.append(
            f"the loaded K2 extension reports host ABI {caps['k2_abi']!r}, "
            f"expected {K2_ABI_VERSION!r}. A pre-k2.2 build has no C5 status "
            "word, which means a NaN candidate wins the merge silently rather "
            "than failing the invocation"
        )
    if caps["k2_carrier"] != K2_CARRIER_VERSION:
        problems.append(
            f"the loaded K2 extension reports carrier {caps['k2_carrier']!r}, "
            f"expected {K2_CARRIER_VERSION!r}. A pre-C4 build consumes the "
            "fp32 gathered tensor, whose axis 2 is H_group rather than "
            "H_local, so a C4 carrier would be read with the wrong head stride"
        )
    if problems:
        raise RuntimeError(
            "fmha_sm100.icp ABI mismatch:\n  - " + "\n  - ".join(problems)
        )
    return caps


__all__ = [
    "ABI_VERSION",
    "ABI_DIGEST",
    "CONTRACT_FAMILY",
    "FROZEN_DATE",
    "K2_ABI_VERSION",
    "K2_CARRIER_VERSION",
    "SOURCE_BASELINE_COMMIT",
    "abi_digest",
    "assert_compatible",
    "capabilities",
    "self_check",
    "to_json",
]
