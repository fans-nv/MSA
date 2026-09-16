# SPDX-License-Identifier: MIT
"""Correctness-only tests: no warmup/timing loops or serving benchmarks."""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from fmha_sm100.nvfp4 import validate_nvfp4_kv


def make_cache(heads, pages=3, *, device="cpu", padding=256):
    """Construct independently addressed packed regions, including storage offset."""
    stride = 18432 * heads + padding
    backing = torch.full((16 + pages * stride,), 0xCD, dtype=torch.uint8, device=device)
    cache = {"layout": "vllm"}
    regions = (
        ("k_data", 0, 64),
        ("k_scale", 8192 * heads, 8),
        ("v_data", 9216 * heads, 64),
        ("v_scale", 17408 * heads, 8),
    )
    for name, offset, width in regions:
        cache[name] = backing.as_strided(
            (pages, heads, 128, width), (stride, 128 * width, width, 1), 16 + offset
        )
    cache["k_global_scale"] = torch.tensor([0.5], dtype=torch.float32, device=device)
    cache["v_global_scale"] = torch.tensor([2.0], dtype=torch.float32, device=device)
    return cache, backing


@pytest.mark.parametrize("heads", [1, 2])
def test_layout_goldens_and_page_copy(heads):
    cache, backing = make_cache(heads)
    stride = validate_nvfp4_kv(cache, torch.device("cpu"))
    assert stride == 18432 * heads + 256
    for (token, group), koff, voff in [
        ((0, 0), 0, 0),
        ((1, 0), 8, 1),
        ((0, 1), 1, 4),
        ((3, 7), 31, 31),
        ((4, 0), 32, 32),
        ((5, 3), 43, 45),
        ((127, 7), 1023, 1023),
    ]:
        assert token * 8 + group == koff
        assert (token // 4) * 32 + 4 * group + token % 4 == voff
        cache["k_scale"][1, heads - 1].flatten()[koff] = token
        cache["v_scale"][1, heads - 1].flatten()[voff] = token
        assert backing[16 + stride + 8192 * heads + 1024 * (heads - 1) + koff] == token
        assert backing[16 + stride + 17408 * heads + 1024 * (heads - 1) + voff] == token
    assert sorted(
        (t // 4) * 32 + s * 4 + t % 4 for t in range(128) for s in range(8)
    ) == list(range(1024))
    backing[16 + 2 * stride : 16 + 3 * stride].copy_(
        backing[16 + stride : 16 + 2 * stride]
    )
    for name in ("k_data", "v_data", "k_scale", "v_scale"):
        torch.testing.assert_close(cache[name][2], cache[name][1])
    assert torch.all(backing[:16] == 0xCD)


def test_layout_rejects_incompatible_strides_and_scale_roles():
    cache, _ = make_cache(2)
    with pytest.raises(ValueError, match="HND"):
        validate_nvfp4_kv(
            {**cache, "k_scale": cache["k_scale"].contiguous()}, torch.device("cpu")
        )
    with pytest.raises(TypeError, match="uint8"):
        validate_nvfp4_kv(
            {**cache, "v_scale": cache["v_scale"].view(torch.float8_e4m3fn)},
            torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="missing"):
        validate_nvfp4_kv(
            {k: v for k, v in cache.items() if k != "k_global_scale"},
            torch.device("cpu"),
        )
    with pytest.raises(TypeError, match="float32 scalar"):
        validate_nvfp4_kv({**cache, "k_global_scale": 0.5}, torch.device("cpu"))


def require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires a supported SM100-family GPU")


def populated_cache(heads, pages=3):
    cache, backing = make_cache(heads, pages=pages, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(817)
    logical_sf = {}
    codebook = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda"
    )
    raw = {}
    for side in ("k", "v"):
        data = cache[side + "_data"]
        data.copy_(
            torch.randint(
                0,
                256,
                data.shape,
                dtype=torch.uint8,
                device="cuda",
                generator=generator,
            )
        )
        sf = torch.tensor([0.5, 0.75, 1, 1.5, 2], device="cuda")[
            torch.randint(
                0, 5, (pages, heads, 128, 8), device="cuda", generator=generator
            )
        ]
        logical_sf[side] = sf.to(torch.float8_e4m3fn).view(torch.uint8)
        if side == "k":
            cache["k_scale"].copy_(logical_sf[side])
        else:
            # Independent 4x8 transpose inside each token quad.
            physical = (
                logical_sf[side]
                .reshape(pages, heads, 32, 4, 8)
                .transpose(-1, -2)
                .reshape(pages, heads, 128, 8)
            )
            cache["v_scale"].copy_(physical)
        codes = torch.stack((data & 15, data >> 4), dim=-1).flatten(-2).long()
        raw[side] = codebook[codes] * sf.repeat_interleave(16, dim=-1)
    return cache, raw, logical_sf


def legacy_scales(logical):
    rows = logical.numel() // 8
    row = torch.arange(rows, device=logical.device)[:, None]
    col = torch.arange(8, device=logical.device)[None, :]
    offset = (
        ((row // 128) * 2 + col // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + col % 4
    )
    result = torch.empty_like(logical.reshape(rows, 8))
    result.flatten()[offset.flatten()] = logical.flatten()
    return result


def reference(q, cache, raw, pages, used, qlen, *, staged=False, qscale=1.0):
    k, v = (
        raw[name][pages.long()]
        .permute(0, 2, 1, 3)
        .reshape(-1, q.shape[1] // 16, 128)[:used]
        for name in ("k", "v")
    )
    if staged:
        # Mirror only the explicit public numerical contract, not kernel indexing.
        k = (k / 6).to(torch.float8_e4m3fn).float() * 6
        v = (v / 6).to(torch.float8_e4m3fn).float() * 6
    k *= cache["k_global_scale"]
    v *= cache["v_global_scale"]
    k, v = k.repeat_interleave(16, dim=1), v.repeat_interleave(16, dim=1)
    logits = torch.einsum("qhd,khd->hqk", q.float() * qscale, k) / math.sqrt(128)
    mask = (
        torch.arange(used, device=q.device)[None, :]
        <= (used - qlen + torch.arange(qlen, device=q.device))[:, None]
    )
    probabilities = logits.masked_fill(~mask[None], -torch.inf).softmax(-1)
    return torch.einsum("hqk,khd->qhd", probabilities, v)


@pytest.mark.parametrize(
    "heads,qlen,used", [(1, 1, 127), (2, 33, 129), (1, 33, 257), (2, 1, 128)]
)
def test_prefill_layout_equivalence(heads, qlen, used, scalar_scales=False):
    require_sm100()
    from fmha_sm100 import build_k2q_csr, sparse_atten_nvfp4_kv_func

    cache, raw, sf = populated_cache(heads)
    if scalar_scales:
        for key in ("k_global_scale", "v_global_scale"):
            scale = cache[key]
            cache[key] = scale.reshape(())
            assert cache[key].data_ptr() == scale.data_ptr()
    torch.manual_seed(12)
    q = (torch.randn((qlen, heads * 16, 128), device="cuda") * 0.1).bfloat16()
    cuq = torch.tensor([0, qlen], device="cuda", dtype=torch.int32)
    cuk = torch.tensor([0, used], device="cuda", dtype=torch.int32)
    pages = torch.tensor([[2, 0, 1]], device="cuda", dtype=torch.int32)
    used_tensor = torch.tensor([used], device="cuda", dtype=torch.int32)
    q2k = torch.full((heads, qlen, 16), -1, device="cuda", dtype=torch.int32)
    for row in range(qlen):
        n = (used - qlen + row) // 128 + 1
        q2k[:, row, :n] = torch.arange(n, device="cuda", dtype=torch.int32)
    csr, indices, schedule = build_k2q_csr(
        q2k,
        cuq,
        cuk,
        128,
        total_k=used,
        max_seqlen_k=used,
        max_seqlen_q=qlen,
        qhead_per_kv=16,
        return_schedule=True,
    )
    kwargs = dict(
        cu_seqlens_q=cuq,
        cu_seqlens_k=cuk,
        max_seqlen_q=qlen,
        max_seqlen_k=used,
        causal=True,
        page_table=pages,
        seqused_k=used_tensor,
        schedule=schedule,
    )
    old = sparse_atten_nvfp4_kv_func(
        q,
        cache["k_data"].contiguous(),
        cache["v_data"].contiguous(),
        legacy_scales(sf["k"]),
        legacy_scales(sf["v"]),
        cache["k_global_scale"],
        cache["v_global_scale"],
        csr,
        indices,
        16,
        **kwargs,
    )
    out = torch.empty_like(q)
    new = sparse_atten_nvfp4_kv_func(
        q,
        cache["k_data"],
        cache["v_data"],
        cache["k_scale"],
        cache["v_scale"],
        cache["k_global_scale"],
        cache["v_global_scale"],
        csr,
        indices,
        16,
        kv_layout="vllm",
        out=out,
        **kwargs,
    )
    assert new.data_ptr() == out.data_ptr()
    torch.testing.assert_close(new, old, atol=0, rtol=0)
    expected = reference(q, cache, raw, pages[0], used, qlen)
    torch.testing.assert_close(new.float(), expected, atol=0.025, rtol=0.025)
    if scalar_scales:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sparse_atten_nvfp4_kv_func(
                q,
                cache["k_data"],
                cache["v_data"],
                cache["k_scale"],
                cache["v_scale"],
                cache["k_global_scale"],
                cache["v_global_scale"],
                csr,
                indices,
                16,
                kv_layout="vllm",
                out=out,
                **kwargs,
            )
        q.mul_(0.5)
        graph.replay()
        expected = reference(q, cache, raw, pages[0], used, qlen)
        torch.testing.assert_close(out.float(), expected, atol=0.025, rtol=0.025)


@pytest.mark.parametrize("heads", [1, 2])
def test_prefill_scalar_scales_and_capture(heads):
    test_prefill_layout_equivalence(heads, 33, 257, scalar_scales=True)


@pytest.mark.parametrize(
    "heads,batch,qlen,splits",
    [
        (1, 1, 1, 2),
        (2, 2, 4, 2),
        (1, 15, 1, 1),
        (2, 16, 1, 2),
        (1, 32, 1, 2),
        (2, 2, 32, 2),
    ],
)
def test_decode_device_scales_and_split_kv(heads, batch, qlen, splits, fp8_first=False):
    require_sm100()
    from fmha_sm100 import fmha_sm100, fmha_sm100_plan

    cache, raw, _ = populated_cache(heads)
    torch.manual_seed(33)
    q = (torch.randn((batch * qlen, heads * 16, 128), device="cuda") * 0.1).to(
        torch.float8_e4m3fn
    )
    used = 257
    pages = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int32)
    kv_indices = pages.repeat(batch)
    selected = torch.full(
        (batch * qlen, heads, 16), -1, device="cuda", dtype=torch.int32
    )
    for b in range(batch):
        for row in range(qlen):
            n = (used - qlen + row) // 128 + 1
            selected[b * qlen + row, :, :n] = torch.arange(
                n, device="cuda", dtype=torch.int32
            )
    plan = fmha_sm100_plan(
        torch.full((batch,), qlen, dtype=torch.int32),
        torch.full((batch,), used, dtype=torch.int32),
        heads * 16,
        num_kv_heads=heads,
        page_size=128,
        kv_block_num=16,
        num_kv_splits=splits,
        sparse_kernel_mode="decode",
        output_maxscore=False,
        device=q.device,
    )
    out = torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)
    fp8_k = (raw["k"] / 6).to(torch.float8_e4m3fn).contiguous()
    fp8_v = (raw["v"] / 6).to(torch.float8_e4m3fn).contiguous()

    def run_fp8(alpha_k, alpha_v):
        ordinary, _ = fmha_sm100(
            q,
            fp8_k,
            fp8_v,
            plan,
            kv_indices=kv_indices,
            kv_block_indexes=selected,
            q_scale=0.75,
            k_scale=6 * alpha_k,
            v_scale=6 * alpha_v,
            o_scale=0.5,
            out=out,
            output_maxscore=False,
        )
        return ordinary.clone()

    original_fp8 = run_fp8(0.5, 2.0) if fp8_first else None
    for alpha_k, alpha_v in ((0.5, 2.0), (1.25, 0.75)):
        cache["k_global_scale"].fill_(alpha_k)
        cache["v_global_scale"].fill_(alpha_v)
        actual, _ = fmha_sm100(
            q,
            cache["k_data"],
            cache["v_data"],
            plan,
            kv_indices=kv_indices,
            kv_block_indexes=selected,
            nvfp4_kv=cache,
            q_scale=0.75,
            o_scale=0.5,
            out=out,
            output_maxscore=False,
        )
        actual = actual.clone()
        expected_fp8 = (
            original_fp8 if original_fp8 is not None else run_fp8(alpha_k, alpha_v)
        )
        original_fp8 = None
        # The existing attention kernel rounds probabilities to FP8 before PV.
        # Require bit-exact agreement with that route on independently staged
        # KV; report float-attention drift separately instead of loosening it.
        torch.testing.assert_close(actual, expected_fp8, atol=0, rtol=0)
        expected = (
            torch.cat(
                [
                    reference(
                        q[b * qlen : (b + 1) * qlen],
                        cache,
                        raw,
                        pages,
                        used,
                        qlen,
                        staged=True,
                        qscale=0.75,
                    )
                    for b in range(batch)
                ]
            )
            * 0.5
        )
        direct = (
            torch.cat(
                [
                    reference(
                        q[b * qlen : (b + 1) * qlen],
                        cache,
                        raw,
                        pages,
                        used,
                        qlen,
                        qscale=0.75,
                    )
                    for b in range(batch)
                ]
            )
            * 0.5
        )
        error, direct_error = actual.float() - expected, actual.float() - direct
        print(
            json.dumps(
                dict(
                    heads=heads,
                    batch=batch,
                    qlen=qlen,
                    splits=splits,
                    alphas=[alpha_k, alpha_v],
                    fp8_staging_max_abs_error=0.0,
                    max_abs_error=error.abs().max().item(),
                    relative_l2_error=(error.norm() / expected.norm()).item(),
                    float_reference_rows_over_025=(
                        (error.abs() > 0.025 + 0.025 * expected.abs()).any(dim=-1)
                    )
                    .sum()
                    .item(),
                    direct_max_abs_error=direct_error.abs().max().item(),
                    direct_relative_l2_error=(
                        direct_error.norm() / direct.norm()
                    ).item(),
                )
            )
        )
    if batch == 1 and qlen == 1:
        # Warm variants were loaded above. Replay reads current device buffers.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fmha_sm100(
                q,
                cache["k_data"],
                cache["v_data"],
                plan,
                kv_indices=kv_indices,
                kv_block_indexes=selected,
                nvfp4_kv=cache,
                q_scale=0.75,
                o_scale=0.5,
                out=out,
                output_maxscore=False,
            )
        q.fill_(0.125)
        pages = pages.flip(0)
        kv_indices.copy_(pages)
        cache["k_global_scale"].fill_(0.5)
        cache["v_global_scale"].fill_(2.0)
        graph.replay()
        expected = (
            reference(q, cache, raw, pages, used, qlen, staged=True, qscale=0.75) * 0.5
        )
        torch.testing.assert_close(out.float(), expected, atol=0.025, rtol=0.025)


def test_fp8_then_nvfp4_symbols_in_fresh_process():
    require_sm100()
    source = Path(__file__).resolve()
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy; tests = runpy.run_path(" + repr(str(source)) + "); "
            "tests['test_decode_device_scales_and_split_kv'](1, 1, 1, 2, fp8_first=True)",
        ],
        check=True,
        timeout=180,
    )


@pytest.mark.parametrize("q_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_prefill_adapter_ragged_current_lengths(q_dtype):
    require_sm100()
    from fmha_sm100 import build_k2q_csr, sparse_atten_nvfp4_kv_func
    from fmha_sm100.sparse_fmha_adapter import sparse_fmha, sparse_fmha_plan

    heads = 2
    cache, raw, sf = populated_cache(heads)
    qlens = [1, 33]
    used = [127, 257]
    mappings = [
        torch.tensor([2], device="cuda", dtype=torch.int32),
        torch.tensor([0, 2, 1], device="cuda", dtype=torch.int32),
    ]
    plan = sparse_fmha_plan(
        torch.tensor(qlens, dtype=torch.int32),
        torch.tensor(used, dtype=torch.int32),
        heads * 16,
        num_kv_heads=heads,
        page_size=128,
        kv_block_num=16,
        output_maxscore=False,
    )
    torch.manual_seed(97)
    q = (torch.randn((sum(qlens), heads * 16, 128), device="cuda") * 0.1).to(q_dtype)
    indices = torch.cat(mappings)
    selected = torch.full((sum(qlens), heads, 16), -1, device="cuda", dtype=torch.int32)
    out = torch.empty(q.shape, dtype=torch.bfloat16, device="cuda")
    for current_lengths in (used, [126, 256]):
        plan["seqused_k"].copy_(
            torch.tensor(current_lengths, device="cuda", dtype=torch.int32)
        )
        offset = 0
        selected.fill_(-1)
        for qlen, current in zip(qlens, current_lengths):
            for row in range(qlen):
                n = (current - qlen + row) // 128 + 1
                selected[offset + row, :, :n] = torch.arange(
                    n, device="cuda", dtype=torch.int32
                )
            offset += qlen
        actual, _ = sparse_fmha(
            q,
            cache["k_data"],
            cache["v_data"],
            plan,
            kv_indices=indices,
            kv_block_indexes=selected,
            nvfp4_kv=cache,
            q_scale=0.75,
            o_scale=0.5,
            out=out,
            output_maxscore=False,
        )
        assert actual.data_ptr() == out.data_ptr()
        actual_snapshot = actual.clone()
        csr, csr_indices, schedule = build_k2q_csr(
            selected.permute(1, 0, 2).contiguous(),
            plan["cu_seqlens_q"],
            plan["cu_seqlens_k"],
            128,
            total_k=sum(used),
            total_rows=plan["total_rows"],
            max_seqlen_k=max(used),
            max_seqlen_q=max(qlens),
            qhead_per_kv=16,
            return_schedule=True,
        )
        page_table = torch.tensor(
            [[2, -1, -1], [0, 2, 1]], device="cuda", dtype=torch.int32
        )
        legacy = sparse_atten_nvfp4_kv_func(
            q,
            cache["k_data"].contiguous(),
            cache["v_data"].contiguous(),
            legacy_scales(sf["k"]),
            legacy_scales(sf["v"]),
            cache["k_global_scale"],
            cache["v_global_scale"],
            csr,
            csr_indices,
            16,
            cu_seqlens_q=plan["cu_seqlens_q"],
            cu_seqlens_k=plan["cu_seqlens_k"],
            max_seqlen_q=max(qlens),
            max_seqlen_k=max(used),
            causal=True,
            softmax_scale=0.75 / math.sqrt(128),
            page_table=page_table,
            seqused_k=plan["seqused_k"],
            schedule=schedule,
        ).mul_(0.5)
        torch.testing.assert_close(actual_snapshot, legacy, atol=0, rtol=0)
        expected = (
            torch.cat(
                [
                    reference(part, cache, raw, mapping, current, qlen, qscale=0.75)
                    for part, mapping, current, qlen in zip(
                        q.split(qlens), mappings, current_lengths, qlens
                    )
                ]
            )
            * 0.5
        )
        if q_dtype == torch.bfloat16:
            torch.testing.assert_close(
                actual_snapshot.float(), expected, atol=0.025, rtol=0.025
            )
        error = actual_snapshot.float() - expected
        print(
            json.dumps(
                dict(
                    prefill_q_dtype=str(q_dtype),
                    current_lengths=current_lengths,
                    legacy_max_abs_error=0.0,
                    float_max_abs_error=error.abs().max().item(),
                    float_relative_l2_error=(error.norm() / expected.norm()).item(),
                )
            )
        )


def test_legacy_aot_template_generation():
    import runpy

    import jinja2

    root = Path(__file__).resolve().parents[2]
    variants = runpy.run_path(str(root / "scripts/warmup_fmha_sm100.py"))[
        "enumerate_all_variants"
    ]()
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(root / "python/fmha_sm100/csrc"),
        undefined=jinja2.StrictUndefined,
    )
    templates = [
        env.get_template(name)
        for name in ("fmha_sm100_inst.jinja", "fmha_sm100_variant_run.cu.jinja")
    ]
    assert variants
    for variant in variants:
        assert "kv_mode" not in variant
        for template in templates:
            rendered = template.render(**variant)
            assert variant["func_name"] in rendered
            assert "maybe_nvfp4_k_data" not in rendered
    print(f"Rendered both legacy templates for {len(variants)} ordinary variants")
