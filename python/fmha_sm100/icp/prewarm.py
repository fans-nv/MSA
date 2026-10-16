"""Prewarm by execution: load or compile every kernel MiniMax-M3 TP2/ICP2 serving runs.

``python -m <pkg>.icp.prewarm build --arch 107a`` compiles them into the image;
``python -m <pkg>.icp.prewarm verify`` loads them with ICP_RUNTIME_JIT=fail.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from functools import partial

from . import _build

# Production config: two local KV heads of D128, GQA 16, top-16 of page 128, FP8 scale shift 3.
_HEAD_KV, _HEAD_DIM, _GQA, _TOPK, _BLK, _SCALE_SHIFT = 2, 128, 16, 16, 128, 3
_SLOT_BYTES = _BLK * (_HEAD_DIM // 2 + _HEAD_DIM // 16)
_COMPOUND_PAGE_BYTES = 2 * _HEAD_KV * _SLOT_BYTES + (_BLK // 2) * _HEAD_DIM


def _call(module: str, attr: str, *args, **kwargs):
    return getattr(importlib.import_module(module, __package__), attr)(*args, **kwargs)


def _scorer_variant(index: int):
    from .. import api, jit

    return jit.get_fmha_variant(*api._icp_scorer_variants()[index][1])


def _decode_scan(scorer, query_len: int, rank: int, arch: str, device):
    if not scorer._aot_path(query_len, rank, arch).is_file():
        _build.refuse_jit("decode", scorer._aot_function_name(query_len, rank))
        # The DSL must target the keyed arch; CLI build sets it before CUTLASS is imported.
        os.environ.setdefault("CUTE_DSL_ARCH", f"sm_{_build.normalize_arch(arch)}")
        scorer._export_aot_scan(query_len, rank, arch)
    scorer._compile_scan(query_len, rank, getattr(device, "index", None) or 0, arch)


def _head_slot_views(storage):
    from ..nvfp4_kv import nvfp4_head_slot_views

    # Strided off the backing storage so every view keeps the compound page stride.
    slots = storage.as_strided(
        (storage.shape[0], 2 * _HEAD_KV, _BLK, _SLOT_BYTES // _BLK),
        (_COMPOUND_PAGE_BYTES, _SLOT_BYTES, _SLOT_BYTES // _BLK, 1),
        storage.storage_offset())
    return nvfp4_head_slot_views(slots[:, 0::2], slots[:, 1::2])


def _nvfp4_prefill(device):
    """One serving-form prefill (CSR schedule, caller-owned out): k2q builder + CuTe AOT."""
    import torch

    from ..sparse import build_k2q_csr, sparse_atten_nvfp4_kv_func

    if device is None:
        raise RuntimeError("the NVFP4 prefill needs a CUDA device")
    # Host draws: vLLM constructs the model under a CUDA default device.
    g = torch.Generator(device="cpu").manual_seed(0)
    q_lens, k_lens = (200, 64), (2000, 600)
    pages = [-(-k // _BLK) for k in k_lens]
    table = torch.zeros((len(k_lens), max(pages)), dtype=torch.int32)
    perm = torch.randperm(sum(pages) + 3, generator=g, device="cpu")
    for b, n in enumerate(pages):
        table[b, :n] = perm[sum(pages[:b]):sum(pages[:b]) + n].to(torch.int32)
    q2k = torch.full((sum(q_lens), _HEAD_KV, _TOPK), -1, dtype=torch.int32)
    row = 0
    for q_len, k_len in zip(q_lens, k_lens, strict=True):
        for t in range(q_len):
            visible = -(-(k_len - q_len + t + 1) // _BLK)
            pick = torch.randperm(visible, generator=g, device="cpu")[:_TOPK].sort().values
            q2k[row + t, :, :pick.numel()] = pick.to(torch.int32)
        row += q_len
    heads = _HEAD_KV * _GQA
    q = torch.randn((sum(q_lens), heads, _HEAD_DIM), generator=g, device="cpu").to(torch.bfloat16)
    storage = torch.randint(0, 256, (sum(pages) + 3, _COMPOUND_PAGE_BYTES), generator=g,
                            device="cpu", dtype=torch.uint8)
    for scales in _head_slot_views(storage)[1::2]:
        scales.fill_(0x38)
    cu_q, cu_k = (torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)
                  for lens in (q_lens, k_lens))
    q, q2k, table, cu_q, cu_k = (t.to(device) for t in (q, q2k, table, cu_q, cu_k))
    k, k_sf, v, v_sf = _head_slot_views(storage.to(device))
    rowptr, qidx, schedule = build_k2q_csr(
        q2k.transpose(0, 1), cu_q, cu_k, _BLK, total_k=0, max_seqlen_k=max(k_lens),
        max_seqlen_q=max(q_lens), total_rows=sum(pages), qhead_per_kv=_GQA,
        return_schedule=True)
    sparse_atten_nvfp4_kv_func(
        q, k, v, k_sf, v_sf,
        torch.tensor([0.75], dtype=torch.float32, device=device),
        torch.tensor([1.25], dtype=torch.float32, device=device),
        rowptr, qidx, topK=_TOPK, blk_kv=_BLK, causal=True, softmax_scale=_HEAD_DIM ** -0.5,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens), page_table=table,
        seqused_k=torch.tensor(k_lens, dtype=torch.int32, device=device), schedule=schedule,
        out=torch.empty_like(q), kv_layout="vllm")
    torch.cuda.synchronize(device)


def _artifacts(device) -> list[tuple[str, partial]]:
    from .scorer.decode import icp_decode_score as scorer

    q8 = {"topk": _TOPK, "gqa_ratio": _GQA, "device": device, "block_scale_shift": _SCALE_SHIFT}
    return [
        *((f"icp:{name}", partial(_build._load_extension, name)) for name in _build.EXTENSIONS),
        *((f"fmha:OnlyScoreIcp[{i}]", partial(_scorer_variant, i)) for i in range(2)),
        ("fmha:plan", partial(_call, "..jit", "get_plan_fn")),
        *((f"decode:{scorer._aot_function_name(q, r)}",
           partial(_decode_scan, scorer, q, r, _build.arch(), device))
          for q, r in scorer._AOT_PROFILES),
        ("q8kv4:cpp_ext", partial(_call, "..decode_q8kv4.interface", "_get_cpp")),
        ("q8kv4:plan", partial(_call, "..decode_q8kv4.jit", "get_plan_fn", device=device)),
        *((f"q8kv4:forward[split_kv={split}]",
           partial(_call, "..decode_q8kv4.jit", "get_fmha_fwd_variant", split_kv=split, **q8))
          for split in (False, True)),
        ("nvfp4:prefill", partial(_nvfp4_prefill, device)),
    ]


def _run(policy: str | None) -> tuple[int, list[str]]:
    import torch

    saved = os.environ.pop("ICP_RUNTIME_JIT", None)
    if policy:
        os.environ["ICP_RUNTIME_JIT"] = policy
    try:
        device = None
        if torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        artifacts, problems = _artifacts(device), []
        for label, load in artifacts:
            try:
                if device is None:
                    load()
                else:
                    with torch.cuda.device(device):
                        load()
            except Exception as exc:  # noqa: BLE001 - every artifact is reported
                problems.append(f"{label}: {exc}")
        return len(artifacts), problems
    finally:
        os.environ.pop("ICP_RUNTIME_JIT", None)
        if saved is not None:
            os.environ["ICP_RUNTIME_JIT"] = saved


def warm() -> None:
    """Load every serving artifact on the current device, compiling what is missing; raise on failure."""
    problems = _run(None)[1]
    if problems:
        raise RuntimeError("ICP prewarm failed:\n" + "\n".join(problems))


def check() -> list[str]:
    """Load every serving artifact with ICP_RUNTIME_JIT=fail; empty iff nothing would compile."""
    return _run("fail")[1]


def _main(argv=None) -> int:
    """CLI: ``build --arch A`` warms with JIT allowed; ``verify`` checks with JIT refused."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build").add_argument("--arch", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--arch")
    verify.add_argument("--policy", choices=("fail",), default="fail")
    args = parser.parse_args(argv)
    arch = (args.arch or _build.arch()).strip().lower().removeprefix("sm_")
    if args.command == "build":
        if "cutlass" in sys.modules and os.environ.get("CUTE_DSL_ARCH") != f"sm_{arch}":
            parser.error("CUTLASS DSL was imported before CUTE_DSL_ARCH was set")
        # Every compiler targets the requested arch, with or without a device.
        os.environ.update(ICP_KERNEL_ARCH=arch, CUTE_DSL_ARCH=f"sm_{arch}",
                          FMHA_SM100_DECODE_Q8KV4_ARCH=arch)
    elif arch != _build.arch():
        parser.error(f"--arch {args.arch} but this process targets sm_{_build.arch()}")
    count, problems = _run(None if args.command == "build" else "fail")
    for problem in problems:
        print(f"ICP_PREWARM_PROBLEM {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"ICP_PREWARM_{'BUILT' if args.command == 'build' else 'VERIFIED'} "
          f"sm_{arch} {count} artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
