# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
"""Public 128x4 scales and ICP cache scales share one attention implementation.

Host tests use real tensors and CuTe imports. Only the compiled/AOT callable is
replaced when checking dispatch; no CUDA initialization or compilation is needed.
GPU cases compare both layouts and an independently dequantized attention result.
"""

import ast
import importlib
import inspect
import textwrap
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def interface(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("host contract initialized CUDA or compiled a kernel")

    monkeypatch.setenv("MSA_REGISTER_TVM_FFI", "0")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    pytest.importorskip("cutlass.cute")
    module = importlib.import_module("fmha_sm100.cute.interface")
    monkeypatch.setattr(module.cute, "compile", forbidden)
    return module


def _inputs(*, paged=True, layout="legacy"):
    tokens, heads, dim, topk = 256, 2, 128, 4
    kv_shape = (2, heads, 128, dim // 2) if paged else (tokens, heads, dim // 2)
    scale_shape = (
        (tokens * heads, dim // 16)
        if layout == "legacy"
        else (*kv_shape[:-1], dim // 16)
    )
    return dict(
        q=torch.zeros((2, 32, dim), dtype=torch.bfloat16),
        k=torch.zeros(kv_shape, dtype=torch.uint8),
        v=torch.zeros(kv_shape, dtype=torch.uint8),
        k_scale_128x4=torch.zeros(scale_shape, dtype=torch.uint8),
        v_scale_128x4=torch.zeros(scale_shape, dtype=torch.uint8),
        k_global_scale=None,
        v_global_scale=None,
        k2q_row_ptr=torch.tensor([[0, 2, 4]] * heads, dtype=torch.int32),
        k2q_q_indices=torch.tensor(
            [[0, 1, 0, 1, 0, 0, 0, 0]] * heads, dtype=torch.int32
        ),
        topK=topk,
        blk_kv=128,
        page_table=torch.tensor([[1, 0]], dtype=torch.int32) if paged else None,
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, tokens], dtype=torch.int32),
        seqused_k=torch.tensor([tokens], dtype=torch.int32) if paged else None,
    )


@pytest.mark.parametrize("layout", ["legacy", "vllm"])
@pytest.mark.parametrize("paged", [False, True])
def test_both_scale_layouts_are_validated_without_cuda(interface, layout, paged):
    arguments = _inputs(paged=paged, layout=layout)
    validate = interface._validate_csr_varlen_nvfp4_kv_inputs
    assert validate(**arguments, kv_layout=layout) == (1, 2)
    wrong = _inputs(paged=paged, layout=("vllm" if layout == "legacy" else "legacy"))
    arguments.update(
        k_scale_128x4=wrong["k_scale_128x4"], v_scale_128x4=wrong["v_scale_128x4"]
    )
    with pytest.raises(ValueError, match="rank-2|shape"):
        validate(**arguments, kv_layout=layout)


def test_legacy_validation_retains_padded_scale_capacity(interface):
    arguments = _inputs(paged=False)
    arguments.update(
        k=arguments["k"][:130],
        v=arguments["v"][:130],
        cu_seqlens_k=torch.tensor([0, 130], dtype=torch.int32),
        k_scale_128x4=torch.zeros((384, 8), dtype=torch.uint8),
        v_scale_128x4=torch.zeros((384, 8), dtype=torch.uint8),
    )
    validate = interface._validate_csr_varlen_nvfp4_kv_inputs
    assert validate(**arguments, kv_layout="legacy") == (1, 2)
    arguments["k_scale_128x4"] = arguments["k_scale_128x4"][:260]
    with pytest.raises(ValueError, match="128x4|padded"):
        validate(**arguments, kv_layout="legacy")


@pytest.mark.parametrize("paged", [False, True])
def test_cache_wrapper_preserves_dev_entry_and_scale_objects(
    interface, monkeypatch, paged
):
    arguments = _inputs(layout="vllm", paged=paged)
    arguments.update(max_seqlen_q=2, max_seqlen_k=256)
    signature = inspect.signature(interface.sparse_atten_nvfp4_kv_func)
    assert signature.parameters["kv_layout"].default == "legacy"
    calls = []
    sentinel = object()

    def record(*args, **kwargs):
        calls.append(signature.bind(*args, **kwargs).arguments)
        return sentinel

    monkeypatch.setattr(interface, "sparse_atten_nvfp4_kv_func", record)
    scales = arguments.pop("k_scale_128x4"), arguments.pop("v_scale_128x4")
    assert (
        interface.sparse_atten_nvfp4_kv_cache_func(
            **arguments, k_scale=scales[0], v_scale=scales[1]
        )
        is sentinel
    )
    assert len(calls) == 1
    assert calls[0]["kv_layout"] == "vllm"
    assert calls[0]["k_scale_128x4"] is scales[0]
    assert calls[0]["v_scale_128x4"] is scales[1]


def test_icp_facade_selects_explicit_cache_layout_entry(interface):
    facade = importlib.import_module("fmha_sm100.icp.attention.nvfp4_prefill")
    assert (
        facade.sparse_atten_nvfp4_kv_func is interface.sparse_atten_nvfp4_kv_cache_func
    )
    assert "k_scale" in inspect.signature(facade.sparse_atten_nvfp4_kv_func).parameters
    assert facade.AOT_FAMILIES == (
        "sparse_forward_sm100_csr_varlen_nvfp4_kv_cache",
        "combine",
    )


def test_layouts_cannot_reuse_each_others_memory_or_aot_cache(interface, monkeypatch):
    aot = importlib.import_module("fmha_sm100.cute.src.common.aot_cache")
    monkeypatch.setattr(interface, "_compile_cache", {})
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda name: nullcontext())
    loads, launches = [], []

    def load(key):
        loads.append(key)

        def launch(*args):
            launches.append((key, args))

        return launch

    monkeypatch.setattr(aot, "try_load_aot", load)
    schedule = SimpleNamespace(
        enabled=True,
        scheduler_metadata=torch.zeros((1, 8), dtype=torch.int32),
        work_count=torch.ones(1, dtype=torch.int32),
        work_capacity=1,
    )
    for layout in ("legacy", "vllm", "legacy", "vllm"):
        args = _inputs(layout=layout)
        args.pop("topK")
        args.update(
            k2q_qsplit_indices=torch.zeros((2, 8), dtype=torch.int32),
            split_counts=torch.ones((2, 2), dtype=torch.int32),
            O_partial=torch.empty((4, 2, 32, 128), dtype=torch.bfloat16),
            LSE_partial=torch.empty((4, 2, 32), dtype=torch.float32),
            LSE_temperature_partial=None,
            softmax_scale=128**-0.5,
            lse_temperature_scale=1.0,
            return_temperature_lse=False,
            max_num_kv_blocks=2,
            head_kv=2,
            max_seqlen_q=2,
            schedule=schedule,
            kv_layout=layout,
        )
        assert (
            interface._call_sparse_forward_sm100_csr_varlen_nvfp4_kv(**args) is schedule
        )
        assert launches[-1][1][2] is args["k_scale_128x4"]
        assert launches[-1][1][3] is args["v_scale_128x4"]
    assert len(launches) == 4
    assert len(loads) == len(interface._compile_cache) == 2
    assert loads[0] != loads[1]
    assert aot._key_to_path(loads[0]) != aot._key_to_path(loads[1])
    assert {key[0] for key in loads} == {
        "sparse_forward_sm100_csr_varlen_nvfp4_kv",
        "sparse_forward_sm100_csr_varlen_nvfp4_kv_cache",
    }


def test_kernel_layout_selection_is_explicit_and_validated(interface):
    kernel = interface.SparseAttentionForwardNvfp4KvSm100
    assert not kernel().vllm_layout
    assert kernel(kv_layout="vllm").vllm_layout
    with pytest.raises(ValueError, match="kv_layout"):
        kernel(kv_layout="ambiguous")


def _integer_method(kernel, name):
    """Evaluate only a scale-address helper, with integer DSL casts as integers."""
    source = ast.parse(textwrap.dedent(inspect.getsource(type(kernel))))
    method = next(
        node
        for node in source.body[0].body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    method.decorator_list = []
    method.returns = None
    for argument in method.args.args:
        argument.annotation = None
    namespace = {"Int32": int, "Int64": int, "const_expr": bool}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "<scale-address>", "exec"),
        namespace,
    )
    return namespace[name]


@pytest.mark.parametrize("is_v", [False, True])
def test_flat_cache_addresses_match_independent_tensor_packing(interface, is_v):
    kernel = interface.SparseAttentionForwardNvfp4KvSm100(kv_layout="vllm")
    offset = _integer_method(kernel, "_flat_vllm_scale_offset")
    tokens, heads, columns = 12, 3, 8
    logical = torch.arange(tokens * heads * columns).reshape(tokens, heads, columns)
    packed = (
        (
            logical.reshape(tokens // 4, 4, heads, columns)
            .permute(0, 2, 3, 1)
            .contiguous()
            .flatten()
        )
        if is_v
        else logical.flatten()
    )
    for token in range(tokens):
        for head in range(heads):
            base = offset(kernel, token, head, heads, is_v)
            addresses = base + torch.arange(columns) * (4 if is_v else 1)
            torch.testing.assert_close(packed[addresses], logical[token, head])


def test_cache_validation_accepts_compound_page_and_head_gaps(interface):
    args = _inputs(layout="vllm")
    for name in ("k", "v", "k_scale_128x4", "v_scale_128x4"):
        tensor = args[name]
        parent = torch.empty((4, 4, *tensor.shape[2:]), dtype=tensor.dtype)
        args[name] = parent[::2, ::2]
    assert interface._validate_csr_varlen_nvfp4_kv_inputs(**args, kv_layout="vllm") == (
        1,
        2,
    )


def test_icp_prewarm_rejects_legacy_only_attention_export(tmp_path):
    from fmha_sm100.icp.attention.nvfp4_prefill._prewarm import _import_aot_objects

    source = tmp_path / "source/sm_107a/nvfp4-digest/aot/v2/toolchain"
    source.mkdir(parents=True)
    for family in ("sparse_forward_sm100_csr_varlen_nvfp4_kv", "combine"):
        (source / f"{family}_fixture.o").write_bytes(b"object")
    target = tmp_path / "target/sm_107a/nvfp4-digest/aot/v2/toolchain"
    with pytest.raises(SystemExit, match="missing AOT families.*nvfp4_kv_cache"):
        _import_aot_objects(source, target)
    assert not target.exists()


@pytest.mark.gpu
@pytest.mark.parametrize("paged", [False, True])
def test_nvfp4_layouts_match_reference_attention(paged):
    if not torch.cuda.is_available():
        pytest.skip("NVFP4 attention layout comparison requires an SM10x CUDA GPU")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("NVFP4 attention layout comparison targets SM10x")
    from fmha_sm100.cute import interface as interface_module
    from fmha_sm100.cute.quantize import (
        Nvfp4QuantizedTensor,
        dequantize_nvfp4_128x4_to_bf16,
        swizzle_nvfp4_scale_to_128x4,
    )

    args = {
        key: value.cuda() if isinstance(value, torch.Tensor) else value
        for key, value in _inputs(paged=paged).items()
    }
    generator = torch.Generator(device="cuda").manual_seed(20261007)
    args["q"] = torch.randn(
        args["q"].shape, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    reference, tiled, cache = {}, {}, {}
    for name in ("k", "v"):
        data = torch.randint(
            0,
            256,
            args[name].shape,
            device="cuda",
            dtype=torch.uint8,
            generator=generator,
        )
        # Exact E4M3 powers of two, varied across tokens/heads/scale columns.
        scale = (
            2.0 ** torch.randint(-3, 1, (512, 8), device="cuda", generator=generator)
        ).to(torch.float8_e4m3fn)
        tiled[name] = swizzle_nvfp4_scale_to_128x4(
            scale.view(torch.uint8), rows=512, cols=8
        )
        quantized = Nvfp4QuantizedTensor(
            data=data,
            scale_128x4=tiled[name],
            global_scale=torch.ones(1, device="cuda"),
            logical_scale_shape=(512, 8),
            original_shape=(*data.shape[:-1], 128),
        )
        reference[name] = dequantize_nvfp4_128x4_to_bf16(quantized).float()
        args[name] = data
        logical = scale.view(torch.uint8).reshape(*data.shape[:-1], 8)
        if name == "k":
            cache[name] = logical
        elif paged:
            cache[name] = (
                logical.reshape(2, 2, 32, 4, 8)
                .transpose(-1, -2)
                .contiguous()
                .reshape(2, 2, 128, 8)
            )
        else:
            cache[name] = (
                logical.reshape(64, 4, 2, 8)
                .permute(0, 2, 3, 1)
                .contiguous()
                .reshape(256, 2, 8)
            )
    args.pop("k_scale_128x4")
    args.pop("v_scale_128x4")
    args.update(max_seqlen_q=2, max_seqlen_k=256)
    legacy = interface_module.sparse_atten_nvfp4_kv_func(
        **args, k_scale_128x4=tiled["k"], v_scale_128x4=tiled["v"]
    )
    current = interface_module.sparse_atten_nvfp4_kv_cache_func(
        **args, k_scale=cache["k"], v_scale=cache["v"]
    )
    if paged:
        reference = {
            name: value[[1, 0]].permute(0, 2, 1, 3).reshape(256, 2, 128)
            for name, value in reference.items()
        }
    keys = reference["k"].repeat_interleave(16, dim=1)
    values = reference["v"].repeat_interleave(16, dim=1)
    scores = torch.einsum("qhd,khd->hqk", args["q"].float(), keys) * (128**-0.5)
    expected = torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), values)
    torch.testing.assert_close(legacy.float(), expected, atol=0.02, rtol=0.02)
    torch.testing.assert_close(current.float(), expected, atol=0.02, rtol=0.02)
    torch.testing.assert_close(current.float(), legacy.float(), atol=0.01, rtol=0.01)
