"""refined-icp-v1 ABI — the single machine-checkable definition.

Version string every lane cites:  refined-icp-v1.abi.1

This module is the executable form of ``ABI/refined-icp-v1.md``. It is pure
Python (no torch, no numpy, no GPU) so that every lane can import it in a unit
test and assert against it without a device or a build.

Provenance
----------
Contract family : refined-icp-v1  (docs/history/CONTRACT.md, amendment at :3-24)
Frozen by       : lane A0, 2026-09-12
Source baseline : vLLM  26b68add0690efe272c43420cf9607e193e8ec0c
                  + campaigns/refined-icp/A0/baseline/tracked-modifications.patch
                  + campaigns/refined-icp/A0/baseline/untracked/

Run ``python3 refined_icp_v1.py`` to execute the self-check.

NOTHING IN THIS FILE LAUNCHES WORK. It computes shapes and predicates only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict

# --------------------------------------------------------------------------
# 0. Version
# --------------------------------------------------------------------------

ABI_VERSION = "refined-icp-v1.abi.1"
CONTRACT_FAMILY = "refined-icp-v1"
FROZEN_DATE = "2026-09-12"
SOURCE_BASELINE_COMMIT = "26b68add0690efe272c43420cf9607e193e8ec0c"

#: Any lane that changes a value in this module MUST bump ABI_VERSION and get
#: review from every dependent owner (EXECUTION_LANES.md:86).

# --------------------------------------------------------------------------
# 1. Geometry  (clause G)
# --------------------------------------------------------------------------

B = 128                      # G1. logical tokens per selection block. Immutable.
D_HEAD = 128                 # G2. index key/query head dimension.
K_SELECT = 16                # G3. selected blocks per query/head.
GLOBAL_MAIN_KV_HEADS = 4     # G4.
GLOBAL_INDEX_Q_HEADS = 4     # G5. every rank scores ALL four. Never reduced to Hlocal.
INDEX_KEY_HEADS = 1          # G6.
SUPPORTED_W = (2, 4)         # G7. W = TP = indexer CP. Manager/main DCP is 1.

CANDIDATE_SIZE_BYTES = 8     # G8. sizeof(Candidate). Frozen.
CANDIDATE_SCORE_OFFSET = 0   # G9. float32
CANDIDATE_ID_OFFSET = 4      # G10. int32
INVALID_BLOCK_ID = -1        # G11.

#: G12. Persistent selected-ID row capacity MUST be a multiple of 4.
#: Source-backed, NOT a design-document invention:
#: vllm/models/minimax_m3/nvidia/model.py:1024-1025
#:     padded_num_tokens = (max_num_batched_tokens + 3) // 4 * 4
#: "Pad tokens to a multiple of 4 so the buffer head stride stays int4-aligned
#:  for build_k2q_csr's vectorised int4 loads."
#: No design document propagates this. It is frozen here so no lane loses it.
QCAPACITY_ALIGNMENT = 4


def R(W: int) -> int:
    """G13. Index rows stored per rank per logical block."""
    _check_W(W)
    return B // W


def h_local(W: int) -> int:
    """G14. Local main-KV / selection head-group count owned by each rank."""
    _check_W(W)
    return GLOBAL_MAIN_KV_HEADS // W


def _check_W(W: int) -> None:
    if W not in SUPPORTED_W:
        raise ValueError(f"refined-icp-v1 supports W in {SUPPORTED_W}, got {W!r}")


# --------------------------------------------------------------------------
# 2. Fragment placement  (clause P)
# --------------------------------------------------------------------------

def owner_rank(u: int, W: int) -> int:
    """P1. Token offset u in [0,128) is owned by rank u//R."""
    if not 0 <= u < B:
        raise ValueError(f"offset u must be in [0,{B}), got {u}")
    return u // R(W)


def owner_row(u: int, W: int) -> int:
    """P2. ...and stored at that rank's local index row u%R."""
    if not 0 <= u < B:
        raise ValueError(f"offset u must be in [0,{B}), got {u}")
    return u % R(W)


def global_token_position(logical_block: int, rank: int, local_row: int, W: int) -> int:
    """P3. Inverse of P1/P2:  128*b + r*R + j."""
    _check_W(W)
    return B * logical_block + rank * R(W) + local_row


def slot_to_page_and_offset(slot: int) -> tuple[int, int]:
    """P4. For a VALID ordinary write slot s:  page = s//128, u = s%128.

    Callers MUST test validity (slot < 0 means PAD) BEFORE calling this.
    A PAD slot forms no cache address and performs no store.
    """
    if slot < 0:
        raise ValueError("PAD slot: test validity before forming an address")
    return slot // B, slot % B


def writes_index_row(slot: int, rank: int, W: int) -> bool:
    """P5. Rank writes the index row for this slot only if u//R == rank.

    Main K/V for the rank's local heads is written for EVERY valid slot on
    EVERY rank. Not owning the index row NEVER suppresses index-query
    production for that row (LANE_INTERFACES.md:20).
    """
    _, u = slot_to_page_and_offset(slot)
    return owner_rank(u, W) == rank


# --------------------------------------------------------------------------
# 3. Visibility / causality  (clause V)
# --------------------------------------------------------------------------

def is_visible(logical_block: int, local_row: int, rank: int, p: int, W: int,
               kv_visible_end: int) -> bool:
    """V1. The normative visibility predicate.

        global_key = 128*b + r*R + j
        visible iff  0 <= global_key < kv_visible_end  and  global_key <= p

    ``p`` is THIS ROW's exact global query position. A final sequence length is
    never a substitute (UNIFIED_KERNEL_DESIGN.md:151).
    """
    g = global_token_position(logical_block, rank, local_row, W)
    return 0 <= g < kv_visible_end and g <= p


def compact_prefix_length(p: int, rank: int, W: int) -> int:
    """V2. Closed form for ACTIVE CONTIGUOUS CAUSAL rows only.

        L_r(N) = floor(N/128)*R + clamp(N%128 - r*R, 0, R),  N = p+1

    Piecewise flat/rising, NOT affine. Transitions at 31/32, 63/64, 95/96 (W4)
    and 63/64 (W2).
    """
    _check_W(W)
    r_, N = R(W), p + 1
    return (N // B) * r_ + max(0, min(N % B - rank * r_, r_))


def is_visible_compact(logical_block: int, local_row: int, rank: int, p: int, W: int) -> bool:
    """V3. Equivalent to V1 for contiguous causal rows: x < L_r(p+1)."""
    x = logical_block * R(W) + local_row
    return x < compact_prefix_length(p, rank, W)


def forced_block_id(p: int) -> int:
    """V4. The visible current block, excluded from ordinary ranking on EVERY
    rank and injected exactly once at the destination."""
    return p // B


def valid_selection_count(p: int, active: bool = True) -> int:
    """V5. min(16, ceil((p+1)/128)) for an active contiguous causal row; 0 if
    inactive."""
    if not active:
        return 0
    return min(K_SELECT, (p + B) // B)   # == ceil((p+1)/128)


# --------------------------------------------------------------------------
# 4. Score wave, D -> E  (clause W)   *** A0.3 / A0.4 RESOLUTION ***
# --------------------------------------------------------------------------

SCORE_DTYPE = "float32"
VALID_DTYPE = "uint8"

#: W1. Column c of a wave names logical block  scan_begin + c.
#: W2. Requirement: scan_end - scan_begin <= Pwave.
#: W3. *** A0.4 RESOLVED *** "Advertised cells" are bounded to the WAVE:
#:     Qchunk x 4 x Pwave, the CALLER-SUPPLIED wave capacity.
#:     They are NEVER Kglobal-wide. A [Tcap, 4, Kglobal] score buffer is
#:     FORBIDDEN by this ABI.
#: W4. *** A0.3 RESOLVED *** Validity is an OUT-OF-BAND uint8 plane,
#:     shape-matched to the score plane. An in-band sentinel is FORBIDDEN
#:     because -inf is a valid representable score and NaN is a failure.
#:     A prefix count is FORBIDDEN because validity is not a prefix.
#: W5. D writes EVERY advertised score AND validity cell each invocation,
#:     including inactive rows, empty fragments, absent pages and capacity
#:     tails. Reused scratch must never leak a previous wave's values.
#: W6. E masks every score load by validity before access.
#: W7. No blanket score-buffer fill is required OR permitted as a substitute
#:     for W5.


def score_wave_shape(qchunk: int, pwave: int) -> tuple[int, int, int]:
    """W8. FP32 score plane, caller-owned, explicit strides."""
    return (qchunk, GLOBAL_INDEX_Q_HEADS, pwave)


def valid_wave_shape(qchunk: int, pwave: int) -> tuple[int, int, int]:
    """W9. uint8 validity plane. Shape-matched to W8. NOT optional."""
    return (qchunk, GLOBAL_INDEX_Q_HEADS, pwave)


def score_wave_bytes(qchunk: int, npart: int, ppart: int) -> int:
    """W10. Total wave scratch = score bytes + validity bytes.

    16*Qchunk*Npart*Ppart  (FP32 score)
    + 4*Qchunk*Npart*Ppart (uint8 validity)
    Context-independent once the plan capacities are fixed.
    """
    cells = qchunk * GLOBAL_INDEX_Q_HEADS * npart * ppart
    return 4 * cells + 1 * cells


def wave_column_to_block(column: int, scan_begin: int) -> int:
    """W11."""
    return scan_begin + column


# --------------------------------------------------------------------------
# 5. Candidates and carrier  (clause C)
# --------------------------------------------------------------------------

CARRIER_DTYPE = "int32"
CARRIER_WORD_SCORE_BITS = 0   # C1. raw FP32 bits, BITCAST. Never a numeric cast.
CARRIER_WORD_BLOCK_ID = 1     # C2. int32 logical block ID.


def local_semantic_shape(qchunk: int) -> tuple[int, int, int]:
    """C3. The rank's own logical result, before routing."""
    return (qchunk, GLOBAL_INDEX_Q_HEADS, K_SELECT)


def send_carrier_shape(W: int, qchunk: int) -> tuple[int, int, int, int, int]:
    """C4. DESTINATION-major at send.  axis 0 = destination rank."""
    return (W, qchunk, h_local(W), K_SELECT, 2)


def recv_carrier_shape(W: int, qchunk: int) -> tuple[int, int, int, int, int]:
    """C5. SOURCE-major at receive.  axis 0 = source rank."""
    return (W, qchunk, h_local(W), K_SELECT, 2)


def peer_split_int32_elements(W: int, qchunk: int) -> int:
    """C6. 2 * Qchunk * Hlocal * 16 int32 words per peer."""
    return 2 * qchunk * h_local(W) * K_SELECT


def peer_split_bytes(W: int, qchunk: int) -> int:
    """C7."""
    return 4 * peer_split_int32_elements(W, qchunk)


def total_carrier_bytes(W: int, qchunk: int) -> int:
    """C8. 512*Qchunk bytes, independent of W (because W*Hlocal == 4)."""
    return W * peer_split_bytes(W, qchunk)


def owned_global_heads(rank: int, W: int) -> range:
    """C9. Destination d owns contiguous global heads [d*Hlocal,(d+1)*Hlocal)."""
    hl = h_local(W)
    return range(rank * hl, (rank + 1) * hl)


#: C10. A contiguous [Qchunk,4,16] tensor MAY NOT be `.view`-ed into
#:      destination-major order. Q and destination axes change order. Producers
#:      emit destination-major directly or perform a real transpose/pack.
#:      Q>=2 at BOTH W=2 and W=4 is the gate that catches a view.
#: C11. Invalid record  <=>  global_block_id == -1. The score field of an
#:      invalid record is ignored and carries no meaning.
#: C12. Validity is determined by the ID, NEVER by the score alone.

# --------------------------------------------------------------------------
# 6. Ordering, reduction, forcing  (clause O)
# --------------------------------------------------------------------------

def canonical_score(score: float) -> float:
    """O1. Signed-zero normalization: -0.0 flushes to +0.0.

    NORMATIVE FORM IS +0 (CONTRACT.md:15). "Canonicalize consistently" is not
    checkable; this is. A max reduction over a different grouping can otherwise
    yield opposite zero signs for the same block and break a bit-exact gate.
    """
    return 0.0 if score == 0.0 else score


def is_nan(score: float) -> bool:
    return score != score


def sort_key(score: float, block_id: int) -> tuple[float, int]:
    """O2. Total order: score DESCENDING, then global ID ASCENDING.

    Returned as a key for ``sorted(..., key=...)`` ascending, so the score is
    negated. Caller MUST have rejected NaN first (O3).
    """
    return (-canonical_score(score), block_id)


#: O3. A NaN partial score FAILS THE INVOCATION. It is never ordered, never
#:     silently dropped, and never given a backend-dependent position.
#: O4. Valid +/-inf participate normally in the order.
#: O5. The receiver considers ALL W*16 incoming records per query/head and
#:     MAX-REDUCES duplicate global IDs BEFORE selecting distinct winners.
#:     Truncating to 16 records before deduplication is WRONG.
#:     (Sorting all records and taking the first distinct IDs is equivalent.)
#: O6. Forcing is by RESERVED SLOT, never by score +inf. Ordinary +inf scores
#:     plus the ID tie-break can otherwise displace the forced block.
#: O7. Exclude forced id p//128 from ordinary ranking on EVERY rank, select
#:     min(15, M-1) ordinary winners where M = ceil((p+1)/128), inject the
#:     forced ID exactly once, then sort the result ascending.

# --------------------------------------------------------------------------
# 7. Output  (clause S)
# --------------------------------------------------------------------------

def selected_blocks_shape(qcapacity: int, W: int) -> tuple[int, int, int]:
    """S1. int32 [Qcapacity, Hlocal, 16].

    Qcapacity is the FULL invocation's persistent selected-ID capacity and is
    DISTINCT from the bounded Qchunk transport scratch (S4).
    """
    if qcapacity % QCAPACITY_ALIGNMENT:
        raise ValueError(
            f"Qcapacity must be a multiple of {QCAPACITY_ALIGNMENT} "
            f"(model.py:1024-1025 int4-aligned head stride); got {qcapacity}")
    return (qcapacity, h_local(W), K_SELECT)


#: S2. Valid prefix is DISTINCT and ASCENDING, of length
#:     valid_selection_count(p); the remainder of the 16 slots is -1.
#: S3. A negative ID anywhere INSIDE the valid prefix is a correctness
#:     failure. Main readers dereference the prefix without a -1 guard.
#: S4. Each chunk writes its ORIGINAL invocation row slice before its scratch
#:     is reused.
#: S5. IDs always name GLOBAL LOGICAL 128-token blocks. Never a physical page
#:     ID, never the historical `local_block*W + rank` striped ID.
#: S6. Inactive rows have a valid count of zero.

# --------------------------------------------------------------------------
# 8. Units and address rules  (clause U)
# --------------------------------------------------------------------------

#: U1. Byte quantities are named *_bytes. Element quantities are named
#:     *_elements. A field name without a unit suffix is a contract violation.
#: U2. Address products are 64-bit. Cast BEFORE multiplying:
#:       int64(physical_page_id) * int64(page_stride_bytes)
#:     int32 logical block IDs do NOT justify an int32 address product.
#: U3. Component pointers are ALREADY REBASED to their region. Do NOT add the
#:     component's offset a second time.
#: U4. The index alias' outer stride is the WHOLE COMPOUND PAGE byte stride,
#:     never R*128.
#: U5. Aliases retain the ACTUAL outer stride. A legal larger stride (an
#:     enclosing slab) does not change semantic spec identity.
#: U6. No flatten()/reshape(-1)/contiguous() to manufacture a dense cross-page
#:     input.

PAGE_BYTES = {
    # (main_format, W) -> (main_bytes, index_bytes, compound_bytes)
    ("nvfp4",     2): (36 * 1024, 8 * 1024, 44 * 1024),
    ("nvfp4",     4): (18 * 1024, 4 * 1024, 22 * 1024),
    ("fp8_e4m3",  2): (64 * 1024, 8 * 1024, 72 * 1024),
    ("fp8_e4m3",  4): (32 * 1024, 4 * 1024, 36 * 1024),
}


def index_region_bytes(W: int) -> int:
    """U7. R rows x 128 bytes. FP8 E4M3, one head, no per-row scale."""
    return R(W) * D_HEAD


# --------------------------------------------------------------------------
# 9. Self-check
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class _Summary:
    abi_version: str
    contract_family: str
    frozen_date: str
    source_baseline_commit: str
    checks: int


def self_check() -> _Summary:
    n = 0

    def ok(cond, msg):
        nonlocal n
        assert cond, msg
        n += 1

    # --- geometry ---
    for W in SUPPORTED_W:
        ok(R(W) * W == B, f"R*W must be B at W={W}")
        ok(h_local(W) * W == GLOBAL_MAIN_KV_HEADS, f"Hlocal*W must be 4 at W={W}")
        ok(R(W) in (32, 64), f"R must be 32 or 64, got {R(W)}")
    ok(CANDIDATE_SIZE_BYTES == 8, "Candidate must be 8 bytes")
    ok(CANDIDATE_ID_OFFSET == CANDIDATE_SCORE_OFFSET + 4, "id follows score")

    # --- placement round-trip: every offset has exactly one owner ---
    for W in SUPPORTED_W:
        seen = {}
        for u in range(B):
            key = (owner_rank(u, W), owner_row(u, W))
            ok(key not in seen, f"two offsets map to {key} at W={W}")
            seen[key] = u
            ok(global_token_position(0, key[0], key[1], W) == u,
               f"placement inverse failed at u={u}, W={W}")
        ok(len(seen) == B, f"placement is not a bijection at W={W}")

    # --- V1 vs V3 must agree on the contiguous causal domain ---
    for W in SUPPORTED_W:
        for p in (0, 30, 31, 32, 62, 63, 64, 94, 95, 96, 126, 127, 128, 129, 1023):
            for rank in range(W):
                for b in range(0, p // B + 2):
                    for j in range(R(W)):
                        a = is_visible(b, j, rank, p, W, kv_visible_end=p + 1)
                        c = is_visible_compact(b, j, rank, p, W)
                        ok(a == c,
                           f"V1/V3 disagree: W={W} p={p} r={rank} b={b} j={j} "
                           f"exact={a} compact={c}")
    # --- the documented ownership transitions actually transition ---
    ok(compact_prefix_length(63, 1, 2) == 0, "W2 rank1 must see 0 rows at p=63")
    ok(compact_prefix_length(64, 1, 2) == 1, "W2 rank1 must see 1 row at p=64")
    for rank, edge in ((1, 32), (2, 64), (3, 96)):
        ok(compact_prefix_length(edge - 1, rank, 4) == 0,
           f"W4 rank{rank} must see 0 rows at p={edge-1}")
        ok(compact_prefix_length(edge, rank, 4) == 1,
           f"W4 rank{rank} must see 1 row at p={edge}")

    # --- selection count ---
    for p, expect in ((0, 1), (127, 1), (128, 2), (255, 2), (256, 3),
                      (16 * B - 1, 16), (16 * B, 16), (10 ** 6, 16)):
        ok(valid_selection_count(p) == expect,
           f"valid_selection_count({p}) != {expect}")
    ok(valid_selection_count(5000, active=False) == 0, "inactive count must be 0")

    # --- carrier sizing ---
    for W in SUPPORTED_W:
        for q in (1, 2, 5, 8):
            ok(send_carrier_shape(W, q) == recv_carrier_shape(W, q),
               "send/recv carrier SHAPES match; only the axis-0 MEANING differs")
            ok(peer_split_bytes(W, q) * W == total_carrier_bytes(W, q), "split sums")
            ok(total_carrier_bytes(W, q) == 512 * q,
               f"carrier must be 512*Qchunk bytes, got {total_carrier_bytes(W,q)}")
        # head ownership is a partition of the 4 global heads
        covered = [h for r in range(W) for h in owned_global_heads(r, W)]
        ok(sorted(covered) == list(range(GLOBAL_INDEX_Q_HEADS)),
           f"head ownership is not a partition at W={W}")

    # --- C10: the view that must not work ---
    for W in SUPPORTED_W:
        q = 3
        ok(local_semantic_shape(q)[0] != send_carrier_shape(W, q)[0],
           "Q-major and destination-major leading axes must differ, so a "
           "reshape cannot silently succeed for Q>1")

    # --- ordering ---
    ok(canonical_score(-0.0) == 0.0, "-0.0 must flush to +0.0")
    ok(str(canonical_score(-0.0)) == "0.0", "-0.0 must flush to +0.0 by SIGN too")
    ok(sort_key(-0.0, 5) == sort_key(0.0, 5), "signed zeros must key identically")
    ok(sort_key(3.0, 9) < sort_key(1.0, 0), "higher score sorts first")
    ok(sort_key(3.0, 2) < sort_key(3.0, 7), "equal score: lower ID first")
    ok(is_nan(float("nan")) and not is_nan(float("-inf")),
       "NaN detection must not catch -inf")

    # --- output ---
    for W in SUPPORTED_W:
        ok(selected_blocks_shape(64, W) == (64, h_local(W), K_SELECT), "S1 shape")
    try:
        selected_blocks_shape(63, 4)
        ok(False, "Qcapacity=63 must be rejected (not a multiple of 4)")
    except ValueError:
        ok(True, "Qcapacity alignment enforced")

    # --- page bytes ---
    for (fmt, W), (main_b, idx_b, comp_b) in PAGE_BYTES.items():
        ok(main_b + idx_b == comp_b, f"page bytes do not sum for {fmt}/W{W}")
        ok(idx_b == index_region_bytes(W),
           f"index region mismatch for {fmt}/W{W}: {idx_b} vs {index_region_bytes(W)}")

    # --- wave scratch accounting includes the validity plane ---
    ok(score_wave_bytes(4, 2, 8) == 5 * 4 * GLOBAL_INDEX_Q_HEADS * 2 * 8,
       "wave bytes must be 4 B score + 1 B validity per cell")
    ok(score_wave_shape(4, 8) == valid_wave_shape(4, 8),
       "validity plane must be shape-matched to the score plane")
    ok(wave_column_to_block(3, 1000) == 1003, "W11")

    return _Summary(ABI_VERSION, CONTRACT_FAMILY, FROZEN_DATE,
                    SOURCE_BASELINE_COMMIT, n)


def abi_digest() -> str:
    """The ABI digest: sha256 of the canonical JSON, INCLUDING its trailing
    newline, so that `sha256sum refined-icp-v1.json` agrees with this value.
    (They disagreed once; see JOURNAL.md dead end DE-3.)"""
    import hashlib
    return hashlib.sha256(to_json().encode()).hexdigest()


def to_json() -> str:
    """Canonical machine-readable projection. Ends with a newline."""
    doc = {
        "abi_version": ABI_VERSION,
        "contract_family": CONTRACT_FAMILY,
        "frozen_date": FROZEN_DATE,
        "source_baseline_commit": SOURCE_BASELINE_COMMIT,
        "geometry": {
            "B": B, "D_head": D_HEAD, "K_select": K_SELECT,
            "global_main_kv_heads": GLOBAL_MAIN_KV_HEADS,
            "global_index_q_heads": GLOBAL_INDEX_Q_HEADS,
            "index_key_heads": INDEX_KEY_HEADS,
            "supported_W": list(SUPPORTED_W),
            "R": {str(w): R(w) for w in SUPPORTED_W},
            "H_local": {str(w): h_local(w) for w in SUPPORTED_W},
            "qcapacity_alignment": QCAPACITY_ALIGNMENT,
        },
        "candidate": {
            "size_bytes": CANDIDATE_SIZE_BYTES,
            "score_offset": CANDIDATE_SCORE_OFFSET, "score_dtype": "float32",
            "id_offset": CANDIDATE_ID_OFFSET, "id_dtype": "int32",
            "invalid_block_id": INVALID_BLOCK_ID,
        },
        "carrier": {
            "dtype": CARRIER_DTYPE,
            "send_shape": "[W, Qchunk, Hlocal, 16, 2]  axis0 = destination",
            "recv_shape": "[W, Qchunk, Hlocal, 16, 2]  axis0 = source",
            "words": ["score_bits_bitcast", "global_block_id"],
            "peer_split_int32_elements": "2 * Qchunk * Hlocal * 16",
            "total_bytes": "512 * Qchunk",
            "view_reshape_permitted": False,
        },
        "score_wave": {
            "score": {"dtype": SCORE_DTYPE, "shape": "[Qchunk, 4, Pwave]"},
            "valid": {"dtype": VALID_DTYPE, "shape": "[Qchunk, 4, Pwave]"},
            "validity_is_out_of_band": True,
            "in_band_sentinel_permitted": False,
            "prefix_count_permitted": False,
            "advertised_cells": "Qchunk * 4 * Pwave   (wave-bounded, NOT Kglobal)",
            "kglobal_wide_buffer_permitted": False,
            "bytes": "16*Qchunk*Npart*Ppart score + 4*Qchunk*Npart*Ppart validity",
        },
        "output": {
            "dtype": "int32", "shape": "[Qcapacity, Hlocal, 16]",
            "valid_prefix": "distinct ascending, length min(16, ceil((p+1)/128))",
            "suffix": INVALID_BLOCK_ID,
            "inactive_count": 0,
            "negative_id_in_prefix_permitted": False,
        },
        "ordering": {
            "primary": "score descending",
            "secondary": "global block id ascending",
            "signed_zero": "-0.0 flushes to +0.0",
            "nan": "fails the invocation",
            "dedup": "max-reduce by global id BEFORE selecting distinct winners",
            "forcing": "reserved final slot, injected once; never score +inf",
        },
        "placement": {
            "owner_rank": "u // R", "owner_row": "u % R",
            "global_token": "128*b + r*R + j",
            "visibility": "0 <= 128*b + r*R + j < kv_visible_end  and  <= p",
            "compact_prefix": "L_r(N) = floor(N/128)*R + clamp(N%128 - r*R, 0, R)",
        },
        "page_bytes": {f"{fmt}/W{w}": {"main": m, "index": i, "compound": c}
                       for (fmt, w), (m, i, c) in PAGE_BYTES.items()},
    }
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    summary = self_check()
    print(json.dumps(asdict(summary), indent=2))
    print("SELF-CHECK PASS")
