"""The vendored ``refined-icp-v1`` ABI, and the build's agreement with it.

This module imports **no torch**, on purpose: the ABI is the one artifact a
consumer can check on any host, and the digest check is what makes "vendored"
mean something rather than "declared". ``fmha_sm100.icp/abi/refined_icp_v1.py`` is
a byte-for-byte copy of the frozen module; if someone edits it without bumping
``ABI_VERSION``, :func:`abi_digest` moves and every test below fails.

Two tiers:

* **unmarked** -- the digest, the three version strings and the frozen geometry.
  Pure Python, runs on a workstation with no torch and no CUDA.
* ``@pytest.mark.gpu`` -- :func:`fmha_sm100.icp.abi.assert_compatible`, which reads
  the version strings **off the built extension** and therefore needs a JIT
  build. That is the whole point of it: a caller checking only the constants
  below would happily hand a ``k2.1`` build a C4 carrier.
"""

from __future__ import annotations

import pytest

from fmha_sm100.icp import abi

# --------------------------------------------------------------------------
# the vendored artifact
# --------------------------------------------------------------------------


def test_the_recomputed_digest_equals_the_frozen_one():
    # The constant is a literal and `abi_digest()` hashes the module's own
    # source, so the two cannot drift in lockstep: editing the vendored ABI
    # moves one and not the other.
    assert abi.abi_digest() == abi.ABI_DIGEST


def test_the_abi_self_check_runs_and_passes():
    summary = abi.self_check()
    assert summary.abi_version == abi.ABI_VERSION
    assert summary.contract_family == abi.CONTRACT_FAMILY
    # A self-check that checked nothing would still "pass"; assert it is
    # actually a battery.
    assert summary.checks > 1000, summary


def test_the_three_version_strings_are_the_frozen_ones():
    # They move independently -- the contract, the K2 host arity and the
    # carrier's dtype/axis meaning are three different landing units -- so all
    # three are pinned as literals.
    assert abi.ABI_VERSION == "refined-icp-v1.abi.1"
    assert abi.K2_ABI_VERSION == "refined-icp-v1.k2.2"
    assert abi.K2_CARRIER_VERSION == "refined-icp-v1.C4"
    assert abi.CONTRACT_FAMILY == "refined-icp-v1"


# --------------------------------------------------------------------------
# the placement predicates the tests themselves rely on
# --------------------------------------------------------------------------


@pytest.mark.parametrize("W", [2, 4])
def test_fragment_placement_gives_every_rank_a_row_of_every_block(W):
    # refined-icp-v1 C1. This is the fact that makes duplicate global block ids
    # NORMAL: R = 128/W rows of EVERY logical block on EVERY rank, so every rank
    # publishes a partial maximum for the same id. The pre-refinement contract
    # gave each block exactly one owner, which is what
    # `test_merge_key.py::test_duplicate_ids_*` used to be able to assume away.
    assert abi.R(W) == 128 // W
    assert abi.h_local(W) * W == abi.GLOBAL_INDEX_Q_HEADS
    # Every token offset in a block is owned by exactly one rank, and every rank
    # owns at least one: placement is a partition of the 128 rows, not a
    # partition of the blocks.
    owners = {abi.owner_rank(u, W) for u in range(128)}
    assert owners == set(range(W))


def test_the_forced_block_and_its_ordinary_count_are_functions_of_p():
    # C3. `n_ordinary = min(15, f)` is what `fmha_sm100.icp.forced_rows` derives
    # and what the kernel re-derives and cross-checks; `valid_selection_count`
    # counts the forced block too, so the two differ by exactly one.
    for p in (0, 1, 127, 128, 129, 255, 256, 128 * 15, 128 * 16, 128 * 40 + 7):
        f = abi.forced_block_id(p)
        assert f == p // 128
        assert abi.valid_selection_count(p) == min(16, f + 1)
        assert abi.valid_selection_count(p) - 1 == min(15, f)
    assert abi.valid_selection_count(4096, active=False) == 0


@pytest.mark.parametrize("W", [2, 4])
def test_the_carrier_shapes_are_the_same_shape_with_different_axis_meaning(W):
    # A0: send and receive carriers have IDENTICAL shapes; only axis 0's
    # meaning changes (destination -> source). That identity is what makes the
    # `all_to_all_single` split symmetric.
    for qchunk in (1, 5):
        assert (
            abi.send_carrier_shape(W, qchunk)
            == abi.recv_carrier_shape(W, qchunk)
            == (W, qchunk, abi.h_local(W), 16, 2)
        )
        assert (
            abi.peer_split_int32_elements(W, qchunk) == 2 * qchunk * abi.h_local(W) * 16
        )
    assert abi.CARRIER_DTYPE == "int32"
    assert abi.INVALID_BLOCK_ID == -1


def test_owned_global_heads_are_contiguous_and_rank_major():
    # C7/C9. Invisible at H_local == 1, which is why W=2 (H_local=2) is the
    # case that can fail.
    W = 2
    seen = []
    for rank in range(W):
        window = list(abi.owned_global_heads(rank, W))
        assert window == [rank * abi.h_local(W) + i for i in range(abi.h_local(W))]
        seen += window
    assert seen == list(range(abi.GLOBAL_INDEX_Q_HEADS))


# --------------------------------------------------------------------------
# the build. GPU.
# --------------------------------------------------------------------------


@pytest.mark.gpu
def test_assert_compatible_passes_against_the_built_extension():
    """The import-time check a consumer is told to make, run for real.

    ``capabilities()`` reads ``k2_abi_version`` and ``k2_carrier`` off the
    module that was actually loaded, so this fails on a stale build rather than
    on a stale constant -- the two failure modes it names (a ``k2.1`` build with
    no status word, a pre-C4 build whose carrier axis 2 is ``H_group``) are both
    silent otherwise.
    """
    caps = abi.assert_compatible()
    assert caps["abi"] == abi.ABI_VERSION
    assert caps["abi_digest"] == abi.ABI_DIGEST
    assert caps["k2_abi"] == abi.K2_ABI_VERSION
    assert caps["k2_carrier"] == abi.K2_CARRIER_VERSION
    # "<absent>" is what `capabilities()` reports for a build that predates the
    # attribute; assert it is not what we got, or the check above would pass on
    # a build that simply never advertised itself.
    assert "<absent>" not in caps.values()


# --------------------------------------------------------------------------
# "a source edit does not prove a rebuild"
# --------------------------------------------------------------------------


def test_a_header_edit_moves_the_freshness_stamp(tmp_path):
    """The property that keeps a header edit from being served out of a stale object.

    ``EXTENSIONS`` lists only the ``.cu`` translation units. Every ``.cuh`` --
    including ``merge_topk.cuh``, which is the SINGLE implementation of the C5
    key and of the duplicate max-reduce -- reaches the compiler through
    ``#include`` and is listed nowhere, and torch's JIT versioner hashes the
    listed sources and the build arguments and nothing else. So a header-only
    edit does not move torch's version, and within one process the stale object
    is served. ``_build.header_digest()`` closes that by folding the header set
    into a build argument, which torch does hash.

    This gates the mechanism, not the artifact, and that is deliberate. The
    obvious test -- edit a header, rebuild, assert the ``.so`` hash moved --
    cannot work: measured on GB300 / torch 2.14, **rebuilding from
    byte-identical sources yields a different ``.so`` hash**, so the object is
    not reproducible build to build and its hash answers neither "did it
    rebuild" nor "is it the same build". An earlier revision of this file tried
    exactly that and reported a freshness hole that did not exist.

    Both halves are asserted: a header edit MUST move the stamp, and an edit to
    a non-header file in the same directory MUST NOT -- otherwise the stamp
    would be a directory-wide sweep that rebuilds on any unrelated change, which
    is the other way to get this wrong.
    """
    from fmha_sm100.icp import _build

    (tmp_path / "a.cuh").write_text("// one\n")
    (tmp_path / "b.cuh").write_text("// two\n")
    (tmp_path / "k.cu").write_text("// a translation unit, listed in EXTENSIONS\n")
    before = _build.header_digest(tmp_path)

    (tmp_path / "b.cuh").write_text("// two, edited\n")
    assert _build.header_digest(tmp_path) != before, (
        "a header edit did not move the freshness stamp; torch's versioner "
        "does not see headers, so nothing else would notice it either"
    )

    (tmp_path / "b.cuh").write_text("// two\n")
    assert _build.header_digest(tmp_path) == before, (
        "the stamp is not a function of the bytes"
    )

    (tmp_path / "k.cu").write_text("// edited translation unit\n")
    assert _build.header_digest(tmp_path) == before, (
        "the stamp moved on a .cu edit. It must not: .cu files are listed in "
        "EXTENSIONS and torch already hashes them, and a stamp that sweeps the "
        "whole directory would force a rebuild on every unrelated change"
    )
