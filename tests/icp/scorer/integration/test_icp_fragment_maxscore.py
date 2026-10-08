# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Fragment-vs-whole-block differential for ``SparseAttnMode::OnlyScoreIcp``.

THE MEASUREMENT
---------------
Under indexer context parallelism rank ``r`` of ``W`` stores only ``R = 128 // W``
index-key rows of each 128-token logical block, so it can only compute a PARTIAL
maximum block score; the true block score is the max across the ``W`` ranks'
partials.  This test drives that path on ONE GPU with no distributed anything:

  * ONE immutable fp8 index-key fixture over ``nblk`` logical 128-token blocks,
    laid out as ``[nblk, 1, 128, D]``.  With a single index KV head that buffer is
    ALSO, byte-for-byte and without a copy, the ``[nblk*W, 1, 128//W, D]`` page
    pool the ICP arm needs: ``view()`` splits the 128-row axis into ``(W, R)`` and
    page ``p = b*W + f`` is exactly rows ``[f*R, (f+1)*R)`` of block ``b``, which is
    exactly the global KV span ``[p*R, p*R+R)`` the kernel addresses as
    ``pos = local_page*W + rank`` (mainloop `sparse_update_cS`, ICP arm).
    Both arms therefore score THE SAME BYTES.
  * reference arm: ``icp_c=1``, page 128 -> true per-block maxima;
  * test arm: ``icp_c=W``, one call per rank, each over its own R-row fragment;
    the ``W`` partial planes are max-reduced.

The bar is EXACT equality, and that is legitimate here SPECIFICALLY because the
operator is max: max is associative and commutative, so max-of-maxes is the
global max bit-for-bit on identical inputs.  (This argument does NOT transfer to a
sum.)  Per-element scores are bit-identical between arms because the QK MMA
accumulates over the same D=128 in the same order regardless of how the KV axis is
tiled.

NaN IS NOT ALLOWED TO PROPAGATE.  The ICP score planes are pre-filled with NaN and
the validity planes with 9, both impossible values: ABI refined-icp-v1.abi.1 W5
says D writes EVERY advertised score and validity cell on EVERY invocation, so any
survivor is an unwritten cell, which is a failure and is reported as its own class
rather than being folded into the max-reduce comparison.

WHY pack_factor IS PINNED TO 1 ON BOTH ARMS.  ``_fmha_sm100_plan_impl`` forces
``pack_factor = 1`` when ``icp_c > 1`` (api.py, the ``if icp_c > 1`` branch), but
would pick 4 for the reference arm at Q <= 32 (h_r = 4).  A packed reference has a
different max_score row/head mapping, so the two arms would no longer be
comparable.  ``num_kv_heads=-1`` in the PLAN makes ``_compute_pack_factor`` return
1; it is used for nothing else in the dense planner, and the kernel takes the real
head counts from the tensor shapes.  Both arms assert ``pack_factor == 1``.

WHY num_kv_splits IS PINNED TO 1.  ``api.py`` auto-selects split-KV when
``num_kv_splits < 1 and qo_tile_size == 128``, and ICP x split-KV is asserted
UNSUPPORTED in ``_fmha_sm100``.  Passing ``num_kv_splits=1`` skips the
auto-selector on both arms, so the two arms differ in the ICP geometry and
nothing else.

Run standalone (prints the sweep table) or under pytest.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "python"))

import torch

from fmha_sm100.icp.scorer.prefill.api import _fmha_sm100, _fmha_sm100_plan
from fmha_sm100.icp.scorer.prefill import api as _fmha_api

HEAD_DIM = 128
NUM_QO_HEADS = 4
NUM_KV_HEADS = 1  # the index KV head; the byte-identity view below needs it to be 1
NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------
def build_index_keys(kv_len: int, device, seed: int = 20260913) -> torch.Tensor:
    """ONE immutable fp8_e4m3 index-key buffer, ``[nblk, 1, 128, D]``."""
    assert kv_len % 128 == 0, "the fixture is whole 128-token logical blocks"
    nblk = kv_len // 128
    gen = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(nblk, NUM_KV_HEADS, 128, HEAD_DIM, generator=gen) * 0.7
    return raw.to(torch.float8_e4m3fn).to(device).contiguous()


def build_queries(qo_len: int, device, seed: int = 4242) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(qo_len, NUM_QO_HEADS, HEAD_DIM, generator=gen) * 0.7
    return raw.to(torch.float8_e4m3fn).to(device).contiguous()


def fragment_view(keys: torch.Tensor, icp_c: int) -> torch.Tensor:
    """The SAME bytes as ``[nblk*C, 1, 128//C, D]``.  No copy, no reordering."""
    nblk = keys.shape[0]
    assert keys.shape[1] == 1, "the zero-copy fragment view needs exactly one KV head"
    return keys.view(nblk * icp_c, 1, 128 // icp_c, HEAD_DIM)


# ---------------------------------------------------------------------------
# analytic ABI predicates (V2 / V3 of refined-icp-v1.abi.1, and the C == 1 case)
# ---------------------------------------------------------------------------
def n_valid_columns(n_eff: int, icp_c: int, icp_rank: int) -> int:
    """ABI V2/V3: how many advertised columns rank ``icp_rank`` may mark valid."""
    r_frag = 128 // icp_c
    rem = min(max(n_eff % 128 - icp_rank * r_frag, 0), r_frag)
    l_r = (n_eff // 128) * r_frag + rem
    return (l_r + r_frag - 1) // r_frag


def run_arm(keys, q, qo_len, kv_len, icp_c, icp_rank, device):
    """One scorer call.  Returns (max_score, valid_score|None, plan_dict)."""
    nblk = kv_len // 128
    r_frag = 128 // icp_c
    kv = fragment_view(keys, icp_c)

    # This rank's LOCAL page table: local page l is global fragment l*C + r,
    # which is the physical page index in the fragment view above.
    kv_indices = torch.arange(nblk, dtype=torch.int32, device=device) * icp_c + icp_rank

    qo_t = torch.tensor([qo_len], dtype=torch.int32)
    kv_t = torch.tensor([kv_len], dtype=torch.int32)

    plan = _fmha_sm100_plan(
        qo_t,
        kv_t,
        NUM_QO_HEADS,
        num_kv_heads=-1,  # -> pack_factor 1 on both arms; see module docstring
        qo_offset=kv_t - qo_t,
        page_size=r_frag,
        output_maxscore=True,
        causal=True,
        num_kv_splits=1,  # -> never the split-KV auto-selector
        icp_c=icp_c,
        icp_rank=icp_rank,
        # the planner's workspace cache is indexed by an INT device id, not a
        # torch.device (api.py `_alloc_workspace_buf`), so hand it the ordinal.
        device=torch.cuda.current_device(),
    )
    plan_dict = plan
    assert plan_dict["pack_factor"] == 1, f"pack_factor={plan_dict['pack_factor']}"
    assert plan_dict["num_kv_splits"] == 1, (
        f"num_kv_splits={plan_dict['num_kv_splits']}"
    )
    max_k_tiles = plan_dict["max_k_tiles"]
    assert max_k_tiles > 0

    if icp_c == 1:
        # The unwritten-cell convention of the C == 1 path IS -inf (api.py fills the
        # buffer it allocates with -inf), so pre-fill with -inf and nothing else.
        ms = torch.full(
            (qo_len, NUM_QO_HEADS, max_k_tiles),
            NEG_INF,
            dtype=torch.float32,
            device=device,
        )
        vs = None
        extra = {}
    else:
        # NaN / 9 are impossible values under W5.  Any survivor is an unwritten cell.
        ms = torch.full(
            (qo_len, NUM_QO_HEADS, max_k_tiles),
            float("nan"),
            dtype=torch.float32,
            device=device,
        )
        vs = torch.full(
            (qo_len, NUM_QO_HEADS, max_k_tiles), 9, dtype=torch.uint8, device=device
        )
        extra = {"valid_score": vs, "icp_c": icp_c, "icp_rank": icp_rank}

    # Record the variant the dispatch ACTUALLY resolved to, by observing the real
    # `get_fmha_variant` call rather than re-deriving it here.  The tile the ICP arm
    # runs on is the whole question, and it is visible nowhere else in the outputs.
    dispatched = {}
    real_get_variant = _fmha_api.get_fmha_variant

    def _recording_get_variant(
        dtype_code, qo_tile_size, single_wg, sparse_mode, variant_page_size, *a, **kw
    ):
        dispatched.update(
            qo_tile_size=qo_tile_size,
            sparse_mode=sparse_mode,
            page_size=variant_page_size,
            single_wg=single_wg,
        )
        return real_get_variant(
            dtype_code,
            qo_tile_size,
            single_wg,
            sparse_mode,
            variant_page_size,
            *a,
            **kw,
        )

    _fmha_api.get_fmha_variant = _recording_get_variant
    try:
        out, ms_out = _fmha_sm100(
            q,
            kv,
            kv,
            plan,
            kv_indices=kv_indices,
            max_score=ms,
            output_o=False,
            output_maxscore=True,
            **extra,
        )
    finally:
        _fmha_api.get_fmha_variant = real_get_variant
    torch.cuda.synchronize()
    assert out is None
    assert ms_out.data_ptr() == ms.data_ptr()
    assert dispatched["page_size"] == r_frag
    assert dispatched["sparse_mode"] == (4 if icp_c > 1 else 2)
    plan_dict = dict(plan_dict)
    plan_dict["dispatched"] = dispatched
    return ms, vs, plan_dict


def expected_validity(qo_len, kv_len, max_k_tiles, icp_c, icp_rank, device):
    """[T, K] bool: the ABI's own predicate, computed on the host side."""
    qo_offset = kv_len - qo_len  # the caller's explicit causal offset
    t = torch.arange(qo_len, device=device)
    p = t + qo_offset
    n_eff = torch.clamp(p + 1, min=0).clamp(max=kv_len)
    r_frag = 128 // icp_c
    rem = ((n_eff % 128) - icp_rank * r_frag).clamp(0, r_frag)
    l_r = (n_eff // 128) * r_frag + rem
    n_valid = (l_r + r_frag - 1) // r_frag
    cols = torch.arange(max_k_tiles, device=device)
    return cols[None, :] < n_valid[:, None]


def measure(qo_len, kv_len, icp_c, device, verbose=True):
    """Returns a dict describing ONE (Q, W) point.  Never raises on mismatch."""
    keys = build_index_keys(kv_len, device)
    q = build_queries(qo_len, device)

    ref_ms, _, ref_plan = run_arm(keys, q, qo_len, kv_len, 1, 0, device)
    max_k_tiles = ref_plan["max_k_tiles"]

    rec = {
        "Q": qo_len,
        "W": icp_c,
        "kv_len": kv_len,
        "max_k_tiles": max_k_tiles,
        "ref_qo_tile": ref_plan["dispatched"]["qo_tile_size"],
        "icp_qo_tile": None,
        "error": None,
    }

    # The reference's own validity: column b is scored iff b*128 < min(kv_len, p+1).
    ref_valid_expect = expected_validity(qo_len, kv_len, max_k_tiles, 1, 0, device)
    rec["ref_finite_matches_causal"] = bool(
        torch.equal(torch.isfinite(ref_ms).all(dim=1), ref_valid_expect)
    )

    parts, valids = [], []
    for r in range(icp_c):
        ms_r, vs_r, icp_plan = run_arm(keys, q, qo_len, kv_len, icp_c, r, device)
        parts.append(ms_r)
        valids.append(vs_r)
        rec["icp_qo_tile"] = icp_plan["dispatched"]["qo_tile_size"]

    # ---- class 1: unwritten cells (ABI W5) --------------------------------
    rec["nan_cells"] = int(sum(int(torch.isnan(p).sum()) for p in parts))
    rec["unwritten_valid_cells"] = int(sum(int((v == 9).sum()) for v in valids))
    rec["bad_valid_values"] = int(sum(int(((v != 0) & (v != 1)).sum()) for v in valids))

    # ---- class 2: the validity plane vs the ABI predicate ------------------
    bad_pred = 0
    for r, v in enumerate(valids):
        want = expected_validity(qo_len, kv_len, max_k_tiles, icp_c, r, device)
        bad_pred += int((v.bool() != want[:, None, :].expand_as(v)).sum())
    rec["validity_vs_abi_mismatch"] = bad_pred

    # ---- class 3: invalid cells must carry -inf (W5) -----------------------
    bad_inv = 0
    for v, p in zip(valids, parts):
        inv = v == 0
        bad_inv += int((inv & ~(p == NEG_INF)).sum())
    rec["invalid_cells_not_neg_inf"] = bad_inv

    # ---- class 4: THE differential ----------------------------------------
    combined = parts[0]
    for p in parts[1:]:
        combined = torch.maximum(combined, p)  # NaN-propagating, deliberately
    exact = combined == ref_ms  # NaN != anything -> counted as mismatch
    rec["mismatch_cells"] = int((~exact).sum())
    rec["total_cells"] = int(exact.numel())
    rec["exact"] = (
        rec["mismatch_cells"] == 0
        and rec["nan_cells"] == 0
        and rec["unwritten_valid_cells"] == 0
    )

    finite_both = torch.isfinite(combined) & torch.isfinite(ref_ms)
    rec["max_abs_diff"] = (
        float((combined[finite_both] - ref_ms[finite_both]).abs().max())
        if int(finite_both.sum())
        else 0.0
    )

    # ---- class 5: the vacuity gate ----------------------------------------
    # If a single rank's partial already equalled the reference, the max-reduce
    # above would be proving nothing.  It must NOT.
    rec["rank0_alone_equals_ref"] = bool(torch.equal(parts[0], ref_ms))
    rec["ref_scored_cells"] = int(torch.isfinite(ref_ms).sum())

    if verbose and rec["mismatch_cells"]:
        bad = (~exact).nonzero()[:5]
        rec["examples"] = [
            (
                tuple(int(x) for x in idx),
                float(ref_ms[tuple(idx)]),
                float(combined[tuple(idx)]),
                [float(p[tuple(idx)]) for p in parts],
                [int(v[tuple(idx)]) for v in valids],
            )
            for idx in bad
        ]
    return rec


# ---------------------------------------------------------------------------
# sweep driver
# ---------------------------------------------------------------------------
Q_SWEEP = [1, 8, 128, 129, 192, 256]
W_SWEEP = [2, 4]


def sweep(kv_len=1024, q_list=Q_SWEEP, w_list=W_SWEEP, device="cuda"):
    rows = []
    for q in q_list:
        for w in w_list:
            try:
                rec = measure(q, kv_len, w, device)
            except Exception as exc:  # noqa: BLE001 - we REPORT failures
                import traceback

                traceback.print_exc()
                rec = {
                    "Q": q,
                    "W": w,
                    "kv_len": kv_len,
                    "exact": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(rec)
            print(f"[point] Q={q:<4} W={w}  -> {rec}", flush=True)
    print("\n=== Q vs exactness (kv_len=%d) ===" % kv_len)
    hdr = "  Q    ref/icp qo_tile  " + "  ".join(f"W={w} exact?" for w in w_list)
    print(hdr)
    for q in q_list:
        cells = []
        tile = ""
        for w in w_list:
            rec = next(r for r in rows if r["Q"] == q and r["W"] == w)
            tile = f"{rec.get('ref_qo_tile', '?')}/{rec.get('icp_qo_tile', '?')}"
            if rec.get("error"):
                cells.append("ERROR")
            elif rec["exact"]:
                cells.append("EXACT")
            else:
                cells.append(
                    f"NO({rec.get('mismatch_cells', '?')}/{rec.get('total_cells', '?')}"
                    f",nan={rec.get('nan_cells', '?')})"
                )
        print(f"  {q:<4} {tile:<16} " + "  ".join(f"{c:<14}" for c in cells))
    return rows


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------
def _assert_point(rec):
    assert not rec.get("error"), rec["error"]
    assert rec["nan_cells"] == 0, (
        f"ABI W5: {rec['nan_cells']} score cells never written"
    )
    assert rec["unwritten_valid_cells"] == 0, (
        f"ABI W5: {rec['unwritten_valid_cells']} validity cells never written"
    )
    assert rec["bad_valid_values"] == 0
    assert rec["validity_vs_abi_mismatch"] == 0
    assert rec["invalid_cells_not_neg_inf"] == 0
    assert not rec["rank0_alone_equals_ref"], (
        "vacuity: one rank's partial already equals the reference, so the "
        "max-reduce proves nothing"
    )
    assert rec["ref_scored_cells"] > 0
    assert rec["mismatch_cells"] == 0, (
        f"{rec['mismatch_cells']}/{rec['total_cells']} cells differ; "
        f"max_abs_diff={rec['max_abs_diff']}; examples={rec.get('examples')}"
    )


def test_icp_fragment_maxscore_sweep():
    # Packed-list ICP route (kv_indices), which keeps host planning; only the
    # direct-table route is device-planned.
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        import pytest

        pytest.skip("Requires the SM100/SM103 FMHA target")
    for q in Q_SWEEP:
        for w in W_SWEEP:
            rec = measure(q, 1024, w, "cuda")
            print(f"[Q={q} W={w}] {rec}", flush=True)
            _assert_point(rec)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-len", type=int, default=1024)
    ap.add_argument("--q", type=int, nargs="*", default=Q_SWEEP)
    ap.add_argument("--w", type=int, nargs="*", default=W_SWEEP)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    rows = sweep(args.kv_len, args.q, args.w, args.device)
    bad = [r for r in rows if not r.get("exact")]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} points EXACT")
    sys.exit(1 if bad else 0)
