"""The prewarmed AOT decode scan must be the JIT scan, bit for bit, on device.

GPU gate for fmha_sm100.icp.prewarm's decode artifacts: the object is exported on
the host with no device (CUTE_DSL_ARCH), linked, and loaded through tvm-ffi;
this proves that path launches the same kernel as the in-process compile.
"""

import os
import subprocess
import sys

import pytest
import torch

from icp_decode_fixtures import make_decode_fixture


@pytest.fixture(scope="module")
def aot_root(tmp_path_factory):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Requires an SM10x CUDA GPU")
    from fmha_sm100.icp import _cache

    root = tmp_path_factory.mktemp("icp-aot")
    arch = _cache.normalize_arch(tuple(torch.cuda.get_device_capability()))
    env = {**os.environ, "ICP_CACHE_ROOT": str(root)}
    env.pop("CUTE_DSL_ARCH", None)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "fmha_sm100.icp.prewarm",
            "build",
            "--arch",
            arch,
            "--components",
            "decode",
        ],
        env=env,
        check=True,
    )
    return root


def _run(scan, api, profile, fixture):
    device_index = fixture.full_q.device.index
    scorer = api._PreparedIcpDecodeScorer(
        profile,
        device_index,
        tuple(torch.cuda.get_device_capability(device_index)),
        scan,
        torch,
    ).bind(
        token_begin=fixture.token_begin,
        token_count=fixture.token_count,
        request_begin=fixture.request_begin,
        request_count=fixture.request_count,
    )
    fixture.score_out.fill_(float("nan"))
    fixture.valid_out.fill_(7)
    scorer(*fixture.inputs)
    torch.cuda.synchronize()
    rows = slice(0, fixture.token_count)
    return fixture.score_out[rows].clone(), fixture.valid_out[rows].clone()


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("query_len", [1, 2, 3, 4])
def test_aot_scan_matches_jit_bit_for_bit(aot_root, monkeypatch, query_len, rank):
    use_pdl = True  # the only admitted profile
    from fmha_sm100.icp.scorer.decode import icp_decode_score as api

    monkeypatch.setenv("ICP_CACHE_ROOT", str(aot_root))
    device = torch.device("cuda", torch.cuda.current_device())
    capability = tuple(torch.cuda.get_device_capability(device))
    stages = api.DEFAULT_NUM_STAGES
    profile = api._DecodeProfile(query_len, rank, 2, 0, stages, use_pdl)
    aot = api._load_aot_scan(query_len, rank, stages, use_pdl, capability)
    assert aot is not None, "prewarm did not produce this profile"
    jit = api._jit_compile_scan(query_len, rank, stages, use_pdl, device.index)
    fixture = make_decode_fixture(
        query_len=query_len,
        batch_size=8,
        rank=rank,
        kv_lens=[32, 33, 64, 65, 128, 129, 0, 385],
        capacity_blocks=8,
        device=device,
        random_seed=1234,
    )
    aot_score, aot_valid = _run(aot, api, profile, fixture)
    jit_score, jit_valid = _run(jit, api, profile, fixture)
    assert torch.equal(aot_score.view(torch.int32), jit_score.view(torch.int32))
    assert torch.equal(aot_valid, jit_valid)
    expected_score, expected_valid = fixture.reference()
    live = expected_valid.bool()
    assert bool((aot_valid.cpu()[live] == 1).all())
    torch.testing.assert_close(
        aot_score.cpu()[live], expected_score[live], rtol=2e-5, atol=1e-4
    )
