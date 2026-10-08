"""Immutable encoded inputs and independent oracles for full-invocation prefill.

One request has ``history`` old tokens and ``query_len`` new tokens. Query row
``t`` is at absolute position ``history+t``; the materialized cache contains
``history+query_len`` tokens. TP2 rank r owns rows 64*r..64*r+63 of *each* B128
block. Rank-local storage is never interpreted as a shorter request.

This module constructs inputs only; it does not import or call an MSA scorer.
The original H2 view and both ICP H4 views share identical encoded FP8 Q/K bytes.
"""

from __future__ import annotations

import ctypes
import hashlib
from dataclasses import dataclass

import torch

VERSION = "icp-full-prefill-fixture-v1"
PAGE, FRAGMENT, DIM, HEADS = 128, 64, 128, 4
CAPACITY = 8192
PAGE_BYTES, INDEX_OFFSET = 45056, 36864
PAD_PAGE = -2147480000


def tensor_digest(tensor):
    """Hash logical contiguous tensor bytes, without a NumPy dependency."""
    host = tensor.detach().cpu().contiguous()
    return hashlib.sha256(
        ctypes.string_at(host.data_ptr(), host.numel() * host.element_size())
    ).hexdigest()


def causal_geometry(query_len, history, rank):
    if rank not in (0, 1) or query_len < 1 or history < 0:
        raise ValueError("Expected positive Q, nonnegative history and TP2 rank")
    positions = torch.arange(query_len, dtype=torch.int64) + history
    visible = positions + 1
    prefix = (visible // PAGE + (visible % PAGE > rank * FRAGMENT)).to(torch.int32)
    current = (positions // PAGE).to(torch.int32)
    forced = torch.where(current < prefix, current, -1).to(torch.int32)
    return {
        "positions": positions,
        "prefix": prefix,
        "forced": forced,
        "whole_prefix": (positions // PAGE + 1).to(torch.int32),
        "active": torch.ones(query_len, dtype=torch.bool),
    }


@dataclass(frozen=True)
class PrefillFixture:
    query_len: int
    history: int
    seed: int
    q_bytes: torch.Tensor
    key_bytes: torch.Tensor
    block_table: torch.Tensor

    @property
    def total_len(self):
        return self.history + self.query_len

    @property
    def live_blocks(self):
        return (self.total_len + PAGE - 1) // PAGE

    def payload(self):
        return {
            "version": VERSION,
            "query_len": self.query_len,
            "history": self.history,
            "seed": self.seed,
            "q_bytes": self.q_bytes,
            "key_bytes": self.key_bytes,
            "block_table": self.block_table,
        }

    def fingerprints(self):
        return {
            name: tensor_digest(getattr(self, name))
            for name in ("q_bytes", "key_bytes", "block_table")
        }

    def materialize(self, rank, device):
        geometry = {
            k: v.to(device)
            for k, v in causal_geometry(self.query_len, self.history, rank).items()
        }
        q = self.q_bytes.view(torch.float8_e4m3fn).to(device)
        whole = self.key_bytes.view(torch.float8_e4m3fn).to(device)
        backing = torch.full(
            (whole.shape[0] * PAGE_BYTES,), 0x7F, dtype=torch.uint8, device=device
        )
        local = backing.view(torch.float8_e4m3fn).as_strided(
            (whole.shape[0], 1, FRAGMENT, DIM),
            (PAGE_BYTES, FRAGMENT * DIM, DIM, 1),
            INDEX_OFFSET,
        )
        local[:, 0].copy_(whole[:, rank * FRAGMENT : (rank + 1) * FRAGMENT])
        return {
            "q": q,
            "original_q": q[:, rank * 2 : (rank + 1) * 2].contiguous(),
            "whole_keys": whole.unsqueeze(1),
            "local_keys": local,
            "compound_storage": backing,
            "table": self.block_table.to(device),
            "geometry": geometry,
        }

    def numerical_reference(self, rank=None, query_rows=None):
        """Independent CPU FP32 dots on sampled queries and EVERY visible block.

        Uses full logical pages and the ordinary table, never the scorer's
        address helpers or local compact storage. Results include exact causal
        masking of materialized future rows. This is a numerical spot gate;
        same-input producer repeatability and selector exactness cover all rows.
        """
        rows = (
            query_rows
            if query_rows is not None
            else sorted(
                {
                    0,
                    min(1, self.query_len - 1),
                    min(63, self.query_len - 1),
                    min(64, self.query_len - 1),
                    min(127, self.query_len - 1),
                    min(128, self.query_len - 1),
                    self.query_len // 2,
                    self.query_len - 1,
                }
            )
        )
        ids = self.block_table[0, : self.live_blocks].long()
        keys = self.key_bytes.view(torch.float8_e4m3fn)[ids].float()
        queries = self.q_bytes.view(torch.float8_e4m3fn)[rows].float()
        dots = torch.matmul(queries, keys.flatten(0, 1).T).reshape(
            len(rows), HEADS, self.live_blocks, PAGE
        )
        key_positions = torch.arange(self.live_blocks * PAGE).reshape(
            self.live_blocks, PAGE
        )
        visible = (
            key_positions[None] <= (torch.tensor(rows) + self.history)[:, None, None]
        )
        visible &= key_positions[None] < self.total_len
        if rank is not None:
            visible &= (torch.arange(PAGE) // FRAGMENT == rank)[None, None]
        expected = torch.where(visible[:, None], dots, float("-inf")).amax(dim=-1)
        return rows, expected, visible.any(dim=-1)


def make_prefill_fixture(*, query_len, history, seed=20260915):
    if query_len < 1 or history < 0 or query_len + history > PAGE * CAPACITY:
        raise ValueError("Fixture must fit the configured 1M context")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    blocks = (history + query_len + PAGE - 1) // PAGE
    pages = blocks + 7
    order = torch.randperm(pages, generator=gen)
    keys = torch.full((pages, PAGE, DIM), float("nan"), dtype=torch.float8_e4m3fn)
    for begin in range(0, blocks, 64):
        target = order[begin : min(begin + 64, blocks)]
        raw = torch.randn(target.numel(), PAGE, DIM, generator=gen) * 0.7
        keys[target] = raw.to(torch.float8_e4m3fn)
    # Invalid physical pages are poisoned. Future tokens of the final live page
    # are finite, as a production page may contain stale/rejected write data;
    # exact causal masks must exclude them. Only table-tail entries are invalid.
    table = torch.full((1, CAPACITY), PAD_PAGE, dtype=torch.int32)
    table[0, :blocks] = order[:blocks].to(torch.int32)
    q = (torch.randn(query_len, HEADS, DIM, generator=gen) * 0.7).to(
        torch.float8_e4m3fn
    )
    return PrefillFixture(
        query_len, history, seed, q.view(torch.uint8), keys.view(torch.uint8), table
    )


def from_payload(payload):
    if payload.get("version") != VERSION:
        raise ValueError("Wrong prefill fixture version")
    return PrefillFixture(
        **{
            k: payload[k]
            for k in (
                "query_len",
                "history",
                "seed",
                "q_bytes",
                "key_bytes",
                "block_table",
            )
        }
    )


def expected_candidates(scores, prefix, forced, topk=16):
    """Exact finite-score C4 oracle, returning canonical int32 record bits.

    Called on the ACTUAL frozen producer scores, never another GPU scorer.
    Every readable fixture score must be finite. Stable score order gives
    ascending global IDs for ties. Inputs may contain poisoned excluded tails.
    """
    rows, heads, blocks = scores.shape
    result = torch.empty((rows, heads, topk, 2), dtype=torch.int32)
    result[..., 0] = -8388608  # float32 -inf bit pattern
    result[..., 1] = -1
    ties = 0
    for begin in range(0, rows, 128):
        end = min(begin + 128, rows)
        count, excluded = prefix[begin:end], forced[begin:end]
        col = torch.arange(blocks)[None, None]
        live = col < count[:, None, None]
        x = scores[begin:end].clone()
        if not bool(torch.isfinite(x[live.expand_as(x)]).all()):
            raise AssertionError("Readable real-fixture score is nonfinite")
        ordinary = live & (col != excluded[:, None, None])
        x.masked_fill_(~ordinary, float("-inf"))
        ids = torch.argsort(x, dim=-1, descending=True, stable=True)[..., : topk + 1]
        vals = x.gather(-1, ids)
        wanted = (count - (excluded >= 0).to(count.dtype)).clamp(0, topk)
        if blocks > topk:
            ties += int(
                (
                    (wanted[:, None] == topk)
                    & (vals[..., topk - 1] == vals[..., topk])
                    & torch.isfinite(vals[..., topk])
                ).sum()
            )
        width = min(topk, blocks)
        valid = torch.arange(width)[None, None] < wanted[:, None, None]
        vals = vals[..., :width]
        vals = torch.where(vals == 0, 0.0, vals).contiguous().view(torch.int32)
        result[begin:end, :, :width, 0] = torch.where(valid, vals, -8388608)
        result[begin:end, :, :width, 1] = torch.where(valid, ids[..., :width], -1).int()
    return result, ties


def canonical_records(bits):
    ids = bits[..., 1]
    order = torch.argsort(torch.where(ids >= 0, ids, 2147483647), dim=-1)
    return bits.gather(-2, order[..., None].expand_as(bits))


def check_candidates(output, expected):
    actual = output.detach().cpu().contiguous().view(torch.int32)
    if not torch.equal(canonical_records(actual), canonical_records(expected)):
        raise AssertionError(
            "Candidate score/ID bits differ from exact independent C4 oracle"
        )


def check_original_ids(output, scores, whole_prefix):
    """Check original Top15+forced IDs, admitting its different cutoff tie rule.

    Original MSA returns IDs, not C4 records, and its tied subset follows legacy
    staging order. Its result is a comparator, never the refined C4 oracle.
    """
    ids = output.detach().cpu()
    tied = 0
    for begin in range(0, ids.shape[0], 128):
        end = min(begin + 128, ids.shape[0])
        n = whole_prefix[begin:end]
        got = ids[begin:end]
        valid = got >= 0
        wanted = n.clamp(max=16)[:, None].expand(got.shape[:2])
        if not torch.equal(valid.sum(dim=-1), wanted):
            raise AssertionError("Original selector returned wrong valid-ID count")
        if bool((valid & (got >= n[:, None, None])).any()):
            raise AssertionError("Original selector returned out-of-range ID")
        sorted_ids = got.sort(dim=-1).values
        if bool(
            (
                (sorted_ids[..., 1:] == sorted_ids[..., :-1])
                & (sorted_ids[..., 1:] >= 0)
            ).any()
        ):
            raise AssertionError("Original selector returned duplicate IDs")
        forced = n - 1
        if not bool((got == forced[:, None, None]).any(dim=-1).all()):
            raise AssertionError("Original selector omitted current forced block")
        x = scores[begin:end]
        cols = torch.arange(x.shape[-1])[None, None]
        ordinary = cols < forced[:, None, None]
        ranked = (
            x.masked_fill(~ordinary, float("-inf")).sort(descending=True, dim=-1).values
        )
        count = forced.clamp(max=15)
        if x.shape[-1] > 15:
            tied += int(
                (
                    (count[:, None] == 15)
                    & (ranked[..., 14] == ranked[..., 15])
                    & torch.isfinite(ranked[..., 15])
                ).sum()
            )
        chosen = x.gather(-1, got.clamp(0, x.shape[-1] - 1))
        chosen.masked_fill_(~valid | (got == forced[:, None, None]), float("-inf"))
        chosen = chosen.sort(descending=True, dim=-1).values
        width = min(15, x.shape[-1])
        mask = torch.arange(width)[None, None] < count[:, None, None]
        if not torch.equal(
            chosen[..., :width][mask.expand_as(chosen[..., :width])],
            ranked[..., :width][mask.expand_as(ranked[..., :width])],
        ):
            raise AssertionError(
                "Original selected ordinary score multiset is not Top15"
            )
    return tied


def check_rank_candidate_union(rank0, rank1, dense_rank_max, whole_prefix):
    """Exact all-row union gate on the SAME evaluated rank scores.

    Local inputs are actual C4 int32 record bits. Sort by score descending,
    global ID ascending, keep the first (maximum) record for each duplicate
    global ID, select 15 ordinary blocks, then inject the current block. The
    independent dense oracle sees ``max(rank0_scores, rank1_scores)`` over ALL
    columns. No original GPU scorer or numerical approximation is involved.
    """
    if (
        rank0.dtype != torch.int32
        or rank1.dtype != torch.int32
        or rank0.shape != rank1.shape
    ):
        raise ValueError("Expected shape-matched int32 C4 bit records from two ranks")
    if tuple(rank0.shape) != tuple(dense_rank_max.shape[:2]) + (16, 2):
        raise ValueError("Candidate rows/heads do not match the frozen score plane")
    forced = whole_prefix - 1
    expected, ties = expected_candidates(dense_rank_max, whole_prefix, forced, topk=15)
    observed = torch.empty_like(expected)
    for begin in range(0, rank0.shape[0], 128):
        end = min(begin + 128, rank0.shape[0])
        records = torch.cat((rank0[begin:end], rank1[begin:end]), dim=-2)
        ids = records[..., 1]
        if bool(((ids >= whole_prefix[begin:end, None, None]) & (ids >= 0)).any()):
            raise AssertionError(
                "Rank candidate union contains an out-of-range global ID"
            )
        if bool((ids == forced[begin:end, None, None]).any()):
            raise AssertionError(
                "A rank admitted the globally forced block as ordinary"
            )
        # Establish ID order first, then use stable score sort for exact ties.
        order = torch.argsort(
            torch.where(ids >= 0, ids, 2147483647), dim=-1, stable=True
        )
        records = records.gather(-2, order[..., None].expand_as(records))
        scores = records[..., 0].contiguous().view(torch.float32)
        scores = torch.where(scores == 0, 0.0, scores)
        records[..., 0] = scores.contiguous().view(torch.int32)
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        records = records.gather(-2, order[..., None].expand_as(records))
        ids = records[..., 1]
        pos = torch.arange(ids.shape[-1])
        previous = pos[None, :] < pos[:, None]
        duplicate = ((ids[..., :, None] == ids[..., None, :]) & previous).any(dim=-1)
        keep = (ids >= 0) & ~duplicate
        order = torch.argsort(
            keep.to(torch.int32), dim=-1, descending=True, stable=True
        )[..., :15]
        picked = records.gather(-2, order[..., None].expand(*order.shape, 2))
        valid = keep.gather(-1, order)
        picked[..., 0] = torch.where(valid, picked[..., 0], -8388608)
        picked[..., 1] = torch.where(valid, picked[..., 1], -1)
        observed[begin:end] = picked
    if not torch.equal(canonical_records(observed), canonical_records(expected)):
        raise AssertionError(
            "Local candidate union differs from dense frozen-rank-max ordinary Top15"
        )
    final_ids = torch.cat(
        (observed[..., 1], forced[:, None, None].expand(*observed.shape[:2], 1)), dim=-1
    )
    expected_ids = torch.cat(
        (expected[..., 1], forced[:, None, None].expand(*expected.shape[:2], 1)), dim=-1
    )
    final_ids = torch.sort(
        torch.where(final_ids >= 0, final_ids, 2147483647), dim=-1
    ).values
    expected_ids = torch.sort(
        torch.where(expected_ids >= 0, expected_ids, 2147483647), dim=-1
    ).values
    final_ids = torch.where(final_ids == 2147483647, -1, final_ids)
    expected_ids = torch.where(expected_ids == 2147483647, -1, expected_ids)
    if not torch.equal(final_ids, expected_ids):
        raise AssertionError(
            "Candidate union with forced injection differs from dense oracle"
        )
    return {
        "final_ids": final_ids,
        "ordinary_record_bits": observed,
        "dense_expected_record_bits": expected,
        "tied_cutoff_rows": ties,
        "rows": rank0.shape[0],
        "head_rows": rank0.shape[0] * rank0.shape[1],
    }
