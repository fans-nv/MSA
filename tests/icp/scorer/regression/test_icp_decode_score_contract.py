# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""CPU checks for the prepared refined-icp-v1 decode API.

The fake tensors expose only host metadata: attempts to read or transform their
contents fail. GPU arithmetic, exact live geometry and graph replay are
qualified by tests/integration/test_icp_decode_score.py, not by these tests.
"""

import importlib.util
from contextlib import nullcontext
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[4] / "python"
DECODE = REPO / "fmha_sm100/icp/scorer/decode"


@pytest.fixture(scope="module")
def api():
    # Load this lightweight module without importing the package's optional FFI
    # registration path. This fixture works without torch, CUDA or CuTe installed.
    source = DECODE / "icp_decode_score.py"
    name = "_icp_decode_score_contract_api"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


class MetadataTensor:
    def __init__(self, shape, dtype, *, strides=None, device=0, pointer=4096):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = SimpleNamespace(type="cuda", index=device)
        self.pointer = pointer
        if strides is None:
            strides = [1] * len(shape)
            for axis in range(len(shape) - 2, -1, -1):
                strides[axis] = strides[axis + 1] * shape[axis + 1]
        self.strides = tuple(strides)

    def stride(self, axis):
        return self.strides[axis]

    def data_ptr(self):
        return self.pointer

    def __getitem__(self, key):
        raise AssertionError("execution must not index live metadata on the host")

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise AssertionError(f"execution attempted a tensor operation: {name}")


@pytest.fixture
def torch_metadata():
    return SimpleNamespace(
        Tensor=MetadataTensor,
        float8_e4m3fn="fp8",
        float32="fp32",
        uint8="uint8",
        int32="int32",
        int64="int64",
        bool="bool",
    )


@pytest.fixture
def tensors():
    return [
        MetadataTensor((64, 4, 128), "fp8", strides=(1024, 256, 1)),
        MetadataTensor((64, 64, 128), "fp8", strides=(65536, 128, 1)),
        MetadataTensor((16, 8192), "int32", strides=(8224, 1)),
        MetadataTensor((17,), "int32"),
        MetadataTensor((16,), "int32"),
        MetadataTensor((64,), "int64"),
        # The activity plane is uint8, not bool: see the rejection case below.
        MetadataTensor((64,), "uint8"),
        MetadataTensor((128, 4, 8192), "fp32", strides=(32800, 8192, 1)),
        MetadataTensor((128, 4, 8192), "uint8", strides=(32960, 8240, 1)),
    ]


@pytest.fixture
def prepared(api, torch_metadata):
    launches = []
    profile = api._DecodeProfile(4, 1, 2, 256)
    scorer = api._PreparedIcpDecodeScorer(
        profile,
        0,
        (10, 3),
        lambda *args: launches.append(("scan", args)),
        torch_metadata,
    ).bind(token_begin=0, token_count=64, request_begin=0, request_count=16)
    return scorer, launches


def test_package_exports_a_lazy_module_without_gpu_dependencies():
    code = """
import builtins
import sys
sys.path.insert(0, sys.argv[1])
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'cutlass', 'quack', 'vllm'}:
        raise ImportError('dependency blocked by CPU import test: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import fmha_sm100
assert 'fmha_sm100.icp_decode_score' not in sys.modules
assert 'icp_decode_score' in fmha_sm100.__all__
from fmha_sm100 import icp_decode_score
assert icp_decode_score.__name__ == 'fmha_sm100.icp.scorer.decode.icp_decode_score'
assert not callable(icp_decode_score)
assert icp_decode_score.ICP_DECODE_SCORE_ABI_VERSION == 2
assert not icp_decode_score.supports_icp_decode_score(query_len=4)
"""
    subprocess.run(
        [sys.executable, "-c", code, str(REPO)],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("query_len,max_dql", [(1, 2), (2, 2), (3, 4), (4, 4)])
def test_every_mma_query_tile_has_a_complete_eight_column_group(
    api, query_len, max_dql
):
    profile = api._DecodeProfile(query_len, 0, 2, 256)
    assert profile.max_decode_query_len == max_dql
    assert 4 * profile.max_decode_query_len % 8 == 0


@pytest.mark.parametrize(
    "change,match",
    [
        ({"query_len": 0}, "query_len"),
        ({"query_len": 5}, "query_len"),
        ({"query_len": True}, "query_len"),
        ({"world_size": 4}, "TP=ICP2"),
        ({"rank": 2}, "rank"),
        ({"rank": -1}, "rank"),
        ({"token_begin": -1}, "token_begin"),
        ({"token_count": 0}, "token_count"),
        ({"request_begin": -1}, "request_begin"),
        ({"request_count": 0}, "request_count"),
        ({"split_k": -1}, "split_k"),
        ({"split_k": 65536}, "split_k"),
        ({"num_stages": 0}, "num_stages"),
        ({"num_stages": 9}, "num_stages"),
        # The non-PDL 4-stage path scored wrong cells in v9 GPU verification.
        ({"use_pdl": False}, "use_pdl=False is refused"),
    ],
)
def test_bad_fixed_profile_fails_before_device_or_compiler_access(
    api, monkeypatch, change, match
):
    def forbidden(*args):
        pytest.fail("invalid CPU profile reached CUDA resolution or compilation")

    monkeypatch.setattr(api, "_cuda_device_info", forbidden)
    monkeypatch.setattr(api, "_compile_profile", forbidden)
    kwargs = dict(
        query_len=4,
        token_begin=0,
        token_count=64,
        request_begin=0,
        request_count=16,
        rank=1,
    )
    kwargs.update(change)
    with pytest.raises(ValueError, match=match):
        api.get_icp_decode_scorer(**kwargs)


@pytest.mark.parametrize(
    "architecture,expected",
    [
        ((10, 0), True),
        ((10, 3), True),
        ((10, 7), True),
        ((9, 0), False),
        ((12, 0), False),
    ],
)
def test_startup_architecture_admission(api, monkeypatch, architecture, expected):
    monkeypatch.setattr(api, "_cuda_device_info", lambda device: (0, architecture))
    assert api.supports_icp_decode_score(query_len=4) is expected


@pytest.mark.parametrize(
    "change",
    [
        {"query_len": 5},
        {"world_size": 4},
        {"page_size": 256},
        {"num_heads": 2},
        {"head_dim": 64},
        {"dtype": "torch.bfloat16"},
    ],
)
def test_out_of_profile_startup_admission_never_probes_cuda(api, monkeypatch, change):
    def forbidden(device):
        pytest.fail("out-of-profile input reached CUDA")

    monkeypatch.setattr(api, "_cuda_device_info", forbidden)
    kwargs = {"query_len": 4}
    kwargs.update(change)
    assert not api.supports_icp_decode_score(**kwargs)


def test_execution_launches_only_the_scan_without_gpu_values_or_compilation(
    api, prepared, tensors, monkeypatch
):
    scorer, launches = prepared

    def forbidden(*args):
        pytest.fail("prepared execution tried to compile or resolve CUDA")

    monkeypatch.setattr(api, "_compile_profile", forbidden)
    monkeypatch.setattr(api, "_cuda_device_info", forbidden)
    api.icp_decode_score(*tensors, scorer=scorer)
    q, k, table, offsets, lengths, positions, active, score, valid = tensors
    assert launches == [
        (
            "scan",
            (
                q,
                k,
                table,
                score,
                lengths,
                offsets,
                positions,
                active,
                valid,
                0,
                64,
                0,
                16,
                256,
            ),
        ),
    ]
    assert not hasattr(scorer, "initialize")
    assert not hasattr(scorer, "_initialize")


def test_call_and_scan_each_submit_one_scan_without_an_initialization_precondition(
    prepared, tensors
):
    scorer, launches = prepared
    scorer(*tensors)
    scorer.scan(*tensors)
    assert [name for name, _ in launches] == ["scan", "scan"]
    assert launches[0][1] == launches[1][1]


def test_factory_compiles_and_caches_only_the_native_scan(
    api, monkeypatch, torch_metadata, tensors, tmp_path
):
    """Exercise the real factory with compiler spies and no CUDA installation.

    A hidden initializer compile or fallback tensor fill must fail this gate,
    even when its work was omitted from the normal execution wrapper.
    """
    compilations, launches, fake_tensors = [], [], []

    class NativeScan:
        def __init__(self, *args, **kwargs):
            self.args, self.kwargs = args, kwargs

    def compile_kernel(kernel, *args, **kwargs):
        compilations.append((kernel, args, kwargs))
        return lambda *runtime_args: launches.append(runtime_args)

    def make_fake_tensor(dtype, shape, **kwargs):
        tensor = SimpleNamespace(dtype=dtype, shape=shape, kwargs=kwargs)
        fake_tensors.append(tensor)
        return tensor

    torch_metadata.cuda = SimpleNamespace(
        device=lambda index: nullcontext(),
        is_current_stream_capturing=lambda: False,
    )
    cutlass = ModuleType("cutlass")
    for name in ("Boolean", "Float8E4M3FN", "Float32", "Int64", "Uint8"):
        setattr(cutlass, name, name)
    cutlass.Int32 = lambda value: ("Int32", value)
    cutlass.cute = SimpleNamespace(
        sym_int64=object,
        runtime=SimpleNamespace(make_fake_stream=lambda **kwargs: "ffi-stream"),
        compile=compile_kernel,
    )
    quack = ModuleType("quack")
    quack.__path__ = []
    compile_utils = ModuleType("quack.compile_utils")
    compile_utils.make_fake_tensor = make_fake_tensor
    kernel_module = ModuleType("_icp_contract_package._icp_decode_score_kernel")
    kernel_module.IndexDecodeScoreKernel = NativeScan
    monkeypatch.setitem(sys.modules, "torch", torch_metadata)
    monkeypatch.setitem(sys.modules, "cutlass", cutlass)
    monkeypatch.setitem(sys.modules, "quack", quack)
    monkeypatch.setitem(sys.modules, "quack.compile_utils", compile_utils)
    monkeypatch.setitem(sys.modules, kernel_module.__name__, kernel_module)
    monkeypatch.setattr(api, "__package__", "_icp_contract_package")
    monkeypatch.setattr(api, "_cuda_device_info", lambda device: (0, (10, 3)))
    # An empty cache root: no prewarmed AOT object may stand in for the compile.
    monkeypatch.setenv("ICP_CACHE_ROOT", str(tmp_path))
    kwargs = dict(
        query_len=4,
        token_begin=0,
        token_count=64,
        request_begin=0,
        request_count=16,
        rank=1,
        device=0,
        split_k=128,
    )
    api._compile_scan.cache_clear()
    try:
        scorer = api.get_icp_decode_scorer(**kwargs)
        assert api.get_icp_decode_scorer(**kwargs)._scan is scorer._scan
        assert len(compilations) == 1
        assert isinstance(compilations[0][0], NativeScan)
        # The window and split_k are five dynamic Int32 launch scalars before
        # the stream, never constructor constants of the native kernel.
        assert compilations[0][1][9:14] == tuple(
            ("Int32", value) for value in (0, 1, 0, 1, 1)
        )
        assert set(compilations[0][0].kwargs) == {"query_len", "rank", "num_stages"}
        assert compilations[0][2] == {"options": "--enable-tvm-ffi"}
        assert len(fake_tensors) == 9
        assert all(tensor.kwargs["leading_dim"] == -1 for tensor in fake_tensors)
        assert not hasattr(scorer, "initialize")
        assert not hasattr(api, "_make_initializer")
        scorer(*tensors)
        assert len(launches) == 1
        assert launches[0][9:] == (0, 64, 0, 16, 128)
        # Another window or split of the same (q, rank) compiles nothing.
        other = api.get_icp_decode_scorer(
            **{**kwargs, "token_count": 8, "request_count": 2, "split_k": 64}
        )
        assert len(compilations) == 1
        assert other._scan is scorer._scan
    finally:
        api._compile_scan.cache_clear()


def test_dynamic_outer_strides_and_larger_retained_outputs_are_accepted(
    prepared, tensors
):
    scorer, launches = prepared
    # The full invocation and 128-row retained allocation stay bound while this
    # graph executes only its 64-row window. No view/copy may be created here.
    scorer(*tensors)
    assert scorer.window.token_count == 64
    assert tensors[-1].shape[0] == 128
    assert launches[0][1][3] is tensors[-2]
    # HTK physical order exposed as THK metadata also has dynamic outer strides.
    tensors[-2] = MetadataTensor((128, 4, 8192), "fp32", strides=(8192, 128 * 8192, 1))
    scorer(*tensors)


@pytest.mark.parametrize(
    "slot,replacement,match",
    [
        (0, MetadataTensor((64, 2, 128), "fp8"), "index_q must have shape"),
        (0, MetadataTensor((64, 4, 128), "fp8", pointer=4097), "aligned TMA base"),
        (
            0,
            MetadataTensor((64, 4, 128), "fp8", strides=(1032, 256, 1)),
            "multiples of 16",
        ),
        (1, MetadataTensor((64, 128, 128), "fp8"), "index_k must have shape"),
        (
            1,
            MetadataTensor((64, 64, 128), "fp8", strides=(4096, 128, 1)),
            "overlapping",
        ),
        (2, MetadataTensor((8, 8192), "int32"), "request range"),
        (2, MetadataTensor((16, 8192), "int32", strides=(8192, 2)), "unit stride"),
        (3, MetadataTensor((16,), "int32"), "end offset"),
        (4, MetadataTensor((16,), "int64"), "seq_lens must be"),
        (5, MetadataTensor((63,), "int64"), "exactly index_q's row count"),
        # OVER-LONG planes, both slots. These are the cases the integration
        # fixtures structurally cannot reach: they size positions/active_rows
        # exactly to index_q, so only the `==` case that every contract accepts
        # is ever exercised, and 71 passing GPU cases could not see a validator
        # that promised `>=` while the compiled signature required `==`. A
        # retained-buffer caller hits this immediately -- it passes a
        # capacity-length plane against a `[:num_tokens]` index_q -- and used
        # to get an opaque tvm-ffi bind error instead of this message.
        (5, MetadataTensor((128,), "int64"), "exactly index_q's row count"),
        (6, MetadataTensor((128,), "uint8"), "exactly index_q's row count"),
        (6, MetadataTensor((64,), "uint8", device=1), "cuda:0"),
        # The scorer's compiled signature declares the activity plane uint8, so
        # the tvm-ffi call rejects a torch `bool` tensor on DLPack dtype code
        # alone. Catching it here keeps that failure at the API boundary with
        # the argument's name, rather than at the first invocation -- which for
        # a decode-only model is inside cudagraph capture.
        (6, MetadataTensor((64,), "bool"), "active_rows must be a 1D uint8 tensor"),
        (7, MetadataTensor((64, 4, 8192), "fp32"), "matching THK shapes"),
        (
            7,
            MetadataTensor((128, 4, 8192), "fp32", strides=(32768, 8000, 1)),
            "overlapping",
        ),
        (8, MetadataTensor((128, 4, 8192), "int32"), "valid_out must be"),
    ],
)
def test_invalid_buffers_are_rejected_before_the_scan_launch(
    prepared, tensors, slot, replacement, match
):
    scorer, launches = prepared
    tensors[slot] = replacement
    with pytest.raises(ValueError, match=match):
        scorer(*tensors)
    assert launches == []


def test_allocation_bound_rejects_a_score_plane_narrower_than_the_native_table(
    prepared, tensors
):
    scorer, launches = prepared
    tensors[-2] = MetadataTensor((64, 4, 1024), "fp32")
    tensors[-1] = MetadataTensor((64, 4, 1024), "uint8")
    with pytest.raises(ValueError, match="full block_table column capacity"):
        scorer(*tensors)
    assert launches == []


def test_q3_second_window_uses_full_invocation_bases_and_explicit_request_range(
    api, torch_metadata
):
    launches = []
    profile = api._DecodeProfile(3, 0, 2, 256)
    scorer = api._PreparedIcpDecodeScorer(
        profile,
        0,
        (10, 3),
        lambda *args: launches.append(args),
        torch_metadata,
    ).bind(token_begin=128, token_count=16, request_begin=42, request_count=6)
    q = MetadataTensor((144, 4, 128), "fp8")
    k = MetadataTensor((24, 64, 128), "fp8", strides=(32768, 128, 1))
    table = MetadataTensor((48, 8192), "int32")
    offsets = MetadataTensor((49,), "int32")
    lengths = MetadataTensor((48,), "int32")
    positions = MetadataTensor((144,), "int64")
    active = MetadataTensor((144,), "uint8")
    score = MetadataTensor((128, 4, 8192), "fp32")
    valid = MetadataTensor((128, 4, 8192), "uint8")
    scorer(q, k, table, offsets, lengths, positions, active, score, valid)
    assert len(launches) == 1
    assert launches[0][0] is q
    assert launches[0][2] is table
    assert launches[0][5] is offsets
    assert launches[0][6] is positions


def test_index_q_may_end_before_the_window(prepared, tensors, torch_metadata):
    """A rung window past the live rows is legal: the kernel stops at
    ``t < index_q.shape[0]``. Positions/activity still match index_q exactly."""
    scorer, launches = prepared
    tensors[0] = MetadataTensor((32, 4, 128), "fp8")
    tensors[5] = MetadataTensor((32,), "int64")
    tensors[6] = MetadataTensor((32,), "uint8")
    scorer(*tensors)
    assert len(launches) == 1


def test_launch_is_unchecked_and_passes_runtime_window_scalars(prepared, tensors):
    scorer, launches = prepared
    tensors[3] = MetadataTensor((3,), "int32")  # would fail validate
    scorer.launch(*tensors)
    window = scorer.window.__class__(8, 4, 2, 1)
    scorer.launch(*tensors, window=window)
    assert [args[9:] for _, args in launches] == [
        (0, 64, 0, 16, 256),
        (8, 4, 2, 1, 256),
    ]


def test_window_is_a_launch_argument_not_a_compile_key(api, torch_metadata):
    launches = []
    base = api._PreparedIcpDecodeScorer(
        api._DecodeProfile(1, 0, 2, 128),
        0,
        (10, 3),
        lambda *args: launches.append(args),
        torch_metadata,
    )
    with pytest.raises(ValueError, match="no token window"):
        base.validate(*[None] * 9)
    first = base.bind(token_begin=0, token_count=4, request_begin=0, request_count=4)
    second = base.bind(token_begin=0, token_count=64, request_begin=0, request_count=64)
    assert first._scan is second._scan is base._scan
    assert first.profile == second.profile == base.profile
    assert not hasattr(base.profile, "token_count")
    with pytest.raises(ValueError, match="request_count"):
        base.bind(token_begin=0, token_count=4, request_begin=0, request_count=0)


def test_native_kernel_takes_the_window_as_runtime_int32_launch_scalars():
    """Launch ABI 2 in the kernel source itself: no window in the constructor
    (the compile key), four Int32 parameters on the launch, grid.x from the
    runtime request count, and no `self.` window reads left in the body."""
    import ast

    source = (DECODE / "_icp_decode_score_kernel.py").read_text()
    tree = ast.parse(source)
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "IndexDecodeScoreKernel"
    )
    methods = {
        item.name: item for item in kernel.body if isinstance(item, ast.FunctionDef)
    }
    window = ("token_begin", "token_count", "request_begin", "request_count")
    init_args = {arg.arg for arg in methods["__init__"].args.kwonlyargs}
    assert not init_args & set(window)
    call_args = {
        arg.arg: ast.unparse(arg.annotation)
        for arg in methods["__call__"].args.args
        if arg.annotation is not None
    }
    assert all(call_args.get(name) == "Int32" for name in window)
    assert call_args.get("split_count") == "Int32"
    assert "grid = (request_count, split_count, 1)" in source
    for name in window:
        assert f"self.{name}" not in source


def test_auto_split_is_at_most_one_resident_wave(api):
    # 4-stage Q4 CTA is ~37 KiB of smem: 6 fit a 228 KiB SM, not 8.
    ctas = api.decode_ctas_per_sm(4, 4, smem_per_sm=228 * 1024, max_threads_per_sm=2048)
    assert ctas == 6
    for batch in (1, 2, 4, 8, 12, 16):
        split = api.auto_split_k(batch, 212, ctas)
        assert batch * split <= ctas * 212 or split == api.AUTO_SPLIT_MIN
    # The measured v9 150K sweep optimum per batch.
    best = {1: 512, 4: 256, 8: 128, 12: 96, 16: 64}
    assert {b: api.auto_split_k(b, 212, ctas) for b in best} == best
