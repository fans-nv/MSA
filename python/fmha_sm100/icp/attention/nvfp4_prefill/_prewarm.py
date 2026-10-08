"""Prewarm of the NVFP4 sparse prefill: the k2q CSR extension and the CuTe AOT objects.

The k2q builder is plain nvcc and builds with no GPU. The CuTe kernels compile
from real tensor arguments in ``interface.py`` (a CPU tensor would compile a
CPU-device ABI), so their AOT objects come from one production-form call on a
CUDA device. A build step with no GPU instead imports the objects of such a run
from ``ICP_NVFP4_AOT_IMPORT``; they are only accepted under this component's
own keyed directory name, i.e. for byte-identical sources, the same arch and
the same CUTLASS DSL release.
"""

from __future__ import annotations

import os
import hashlib
import json
import pathlib
import shutil

from . import AOT_FAMILIES, COMPONENT, PREWARM_VARIANTS
from . import _configure, _load_k2q_extension, aot_dir
from . import k2q_extension_path

IMPORT_ENV = "ICP_NVFP4_AOT_IMPORT"
_BLK = 128
_HEAD_KV = 2
_HEAD_DIM = 128
_SIDE_DATA_BYTES = _HEAD_KV * _BLK * (_HEAD_DIM // 2)
_SIDE_SCALE_BYTES = _HEAD_KV * _BLK * (_HEAD_DIM // 16)
_SIDE_BYTES = _SIDE_DATA_BYTES + _SIDE_SCALE_BYTES
_INDEX_BYTES = (_BLK // 2) * _HEAD_DIM
_COMPOUND_PAGE_BYTES = 2 * _SIDE_BYTES + _INDEX_BYTES
PREWARM_LAYOUTS = ("v13", "public")


def _compound_cache_views(storage, layout="v13"):
    """The admitted TP2/P128 compound pages, with historical or public slots.

    Transfer the backing storage before creating target-device views; copying
    each non-dense view independently would discard the compound page stride.
    """
    import torch

    if (storage.dtype != torch.uint8 or storage.ndim != 2
            or storage.shape[1] != _COMPOUND_PAGE_BYTES or not storage.is_contiguous()):
        raise ValueError("prewarm storage must be dense uint8 [pages,45056]")
    pages = storage.shape[0]
    if layout == "public":
        from ....nvfp4_kv import nvfp4_head_slot_views

        slot_bytes = _BLK * (_HEAD_DIM // 2 + _HEAD_DIM // 16)
        slots = storage.as_strided(
            (pages, 2 * _HEAD_KV, _BLK, 72),
            (_COMPOUND_PAGE_BYTES, slot_bytes, 72, 1),
            storage.storage_offset(),
        )
        return dict(zip(("k", "k_sf", "v", "v_sf"),
                        nvfp4_head_slot_views(slots[:, 0::2], slots[:, 1::2])))
    if layout != "v13":
        raise ValueError(f"unknown compound prewarm layout: {layout!r}")
    views = {}
    for name, offset, width in (
        ("k", 0, _HEAD_DIM // 2),
        ("k_sf", _SIDE_DATA_BYTES, _HEAD_DIM // 16),
        ("v", _SIDE_BYTES, _HEAD_DIM // 2),
        ("v_sf", _SIDE_BYTES + _SIDE_DATA_BYTES, _HEAD_DIM // 16),
    ):
        views[name] = storage.as_strided(
            (pages, _HEAD_KV, _BLK, width),
            (_COMPOUND_PAGE_BYTES, _BLK * width, width, 1),
            storage.storage_offset() + offset,
        )
    return views


def required_aot_keys(arch, *, layouts=PREWARM_LAYOUTS):
    """Exact serving keys, using the same tuple builders as the dev loaders.

    This is host-only: Q/output lengths are dynamic, while TP2 head geometry,
    paged strides, dtypes, causal/schedule/global-scale flags and combine
    architecture are compile inputs. The current vLLM prefill uses BF16 Q;
    its separate FP8 Q buffer goes to Q8KV4 decode, not this facade.
    """
    import torch

    from ... import _cache
    from ....cute.interface import _nvfp4_forward_key
    from ....cute.src.sm100.fwd.combine import _combine_key, _get_cutlass_dtype

    sm = int(_cache.normalize_arch(arch).removesuffix("a"))
    capability = (sm // 10, sm % 10)
    keys = []
    for layout in layouts:
        views = _compound_cache_views(
            torch.empty((1, _COMPOUND_PAGE_BYTES), dtype=torch.uint8), layout)
        has_global_scales = layout == "public"
        for q_dtype, qhead_per_kv, topk in PREWARM_VARIANTS:
            keys.append(_nvfp4_forward_key(
                "vllm", _HEAD_KV, views["k"], views["v"], views["k_sf"], views["v_sf"],
                _HEAD_DIM, _BLK, qhead_per_kv, getattr(torch, q_dtype), torch.bfloat16,
                True, True, True, _BLK, object(), False,
                os.environ.get("MINIMAX_KVFP4_FP8_PAIR_DEQUANT", "1") != "0",
                has_global_scales, False,
            ))
            keys.append(_combine_key(
                capability, 3 if capability == (10, 7) else 2, _HEAD_DIM, 128, 64,
                topk, _get_cutlass_dtype(torch.bfloat16), _get_cutlass_dtype(torch.bfloat16),
                True, False, True, False, True, has_global_scales, True,
                3 if has_global_scales and capability != (10, 7) else 0,
            ))
    return tuple(dict.fromkeys(keys))


def required_aot_entries(arch):
    """Exact loader filenames and serialized keys, independent of placement."""
    from ....cute.src.common import aot_cache

    return {pathlib.Path(aot_cache._key_to_path(key)).name: repr(key)
            for key in required_aot_keys(arch)}


def production_call(q_dtype="bfloat16", qhead_per_kv=16, topk=16, *,
                    head_kv=2, q_lens=(200, 64), k_lens=(2000, 600), seed=0,
                    device="cuda", layout="v13"):
    """Serving call: transposed q2k, total_k=0, CSR schedule, caller-owned out.

    The public layout includes calibrated K/V tensor scales; v13 omits them. Returns
    ``(out, inputs)``; K/V bytes are random with unit block scales."""
    import torch  # noqa: PLC0415

    from . import build_k2q_csr, sparse_atten_nvfp4_kv_func  # noqa: PLC0415

    if head_kv != _HEAD_KV:
        raise ValueError("the production prewarm profile requires TP2 with two local KV heads")
    g = torch.Generator(device="cpu").manual_seed(seed)
    dtype = getattr(torch, q_dtype)
    heads, dim = head_kv * qhead_per_kv, 128
    pages = [-(-k // _BLK) for k in k_lens]
    num_pages = sum(pages) + 3
    table = torch.full((len(k_lens), max(pages)), 0, dtype=torch.int32)
    perm = torch.randperm(num_pages, generator=g)
    cursor = 0
    for b, n in enumerate(pages):
        table[b, :n] = perm[cursor:cursor + n].to(torch.int32)
        cursor += n
    total_q = sum(q_lens)
    q2k = torch.full((total_q, head_kv, topk), -1, dtype=torch.int32)
    row = 0
    for q_len, k_len in zip(q_lens, k_lens, strict=True):
        for t in range(q_len):
            visible = -(-(k_len - q_len + t + 1) // _BLK)
            pick = torch.randperm(visible, generator=g)[:topk].sort().values
            q2k[row + t, :, :pick.numel()] = pick.to(torch.int32)
        row += q_len
    q = torch.randn((total_q, heads, dim), generator=g).to(dtype)
    storage = torch.randint(0, 256, (num_pages, _COMPOUND_PAGE_BYTES), generator=g,
                            dtype=torch.uint8)
    cpu_views = _compound_cache_views(storage, layout)
    cpu_views["k_sf"].fill_(0x38)
    cpu_views["v_sf"].fill_(0x38)
    cu_q = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32)
    cu_k = torch.tensor([0, *torch.tensor(k_lens).cumsum(0).tolist()], dtype=torch.int32)
    seq = torch.tensor(k_lens, dtype=torch.int32)
    inputs = {name: t.to(device) for name, t in dict(
        q=q, q2k=q2k, page_table=table,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, seq_lens=seq).items()}
    inputs.update(_compound_cache_views(storage.to(device), layout))
    inputs["k_global_scale"] = (
        torch.tensor([0.75], dtype=torch.float32, device=device) if layout == "public" else None)
    inputs["v_global_scale"] = (
        torch.tensor([1.25], dtype=torch.float32, device=device) if layout == "public" else None)
    rowptr, qidx, schedule = build_k2q_csr(
        inputs["q2k"].transpose(0, 1), inputs["cu_seqlens_q"], inputs["cu_seqlens_k"],
        _BLK, total_k=0, max_seqlen_k=max(k_lens), max_seqlen_q=max(q_lens),
        total_rows=sum(pages), qhead_per_kv=qhead_per_kv, return_schedule=True)
    out = torch.empty((total_q, heads, dim), dtype=torch.bfloat16, device=device)
    sparse_atten_nvfp4_kv_func(
        inputs["q"], inputs["k"], inputs["v"], inputs["k_sf"], inputs["v_sf"],
        inputs["k_global_scale"], inputs["v_global_scale"],
        rowptr, qidx, topK=topk, blk_kv=_BLK, causal=True, softmax_scale=dim ** -0.5,
        cu_seqlens_q=inputs["cu_seqlens_q"], cu_seqlens_k=inputs["cu_seqlens_k"],
        max_seqlen_q=max(q_lens), max_seqlen_k=max(k_lens),
        page_table=inputs["page_table"], seqused_k=inputs["seq_lens"],
        schedule=schedule, out=out)
    inputs.update(k2q_row_ptr=rowptr, k2q_q_indices=qidx)
    return out, inputs


def _aot_objects(directory: pathlib.Path, arch: str | None = None) -> list[pathlib.Path]:
    objects = sorted(directory.glob("*.o"))
    missing = [family for family in AOT_FAMILIES
               if not any(obj.stem.startswith(family + "_") for obj in objects)]
    if missing:
        raise SystemExit(f"{COMPONENT}: missing AOT families {missing} under {directory}")
    from ....cute.src.common import aot_cache  # noqa: PLC0415
    from ... import _cache  # noqa: PLC0415

    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest != aot_cache._MANIFEST:
            raise ValueError("AOT toolchain differs")
        available_keys = {}
        for obj in objects:
            entry = json.loads(obj.with_suffix(".json").read_text())
            if entry["schema"] != aot_cache._SCHEMA or not entry["inputs"]:
                raise ValueError(f"missing source inventory for {obj.name}")
            key_hash = hashlib.sha256(entry["key"].encode()).hexdigest()[:16]
            if not obj.stem.endswith("_" + key_hash):
                raise ValueError(f"compile key differs for {obj.name}")
            available_keys[obj.stem] = entry["key"]
            for relative, expected in entry["inputs"].items():
                source = (aot_cache._CUTE_ROOT / relative).resolve()
                if not source.is_relative_to(aot_cache._CUTE_ROOT.resolve()):
                    raise ValueError(f"source outside cute/: {relative}")
                if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"AOT source differs: {relative}")
        required = required_aot_entries(arch or _cache.key_arch())
        missing = [name for name, key in required.items()
                   if available_keys.get(name) != key]
        if missing:
            raise ValueError(f"missing required production AOT keys/files: {sorted(missing)}")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise SystemExit(f"{COMPONENT}: invalid AOT metadata under {directory}: {exc}") from exc
    return objects


def _import_aot_objects(source: pathlib.Path, target: pathlib.Path) -> None:
    # The component digest includes sources/DSL, while its parent carries arch.
    # Comparing only nvfp4-<digest> would accept a different architecture's AOT.
    # dev AOT appends its schema and toolchain namespace to the ICP placement.
    source_key = source.parts[-5:]
    target_key = target.parts[-5:]
    if (source_key != target_key or len(target_key) != 5
            or not target_key[0].startswith("sm_")
            or not target_key[1].startswith("nvfp4-")
            or target_key[2:4] != ("aot", "v2")):
        raise SystemExit(
            f"{IMPORT_ENV}={source} identifies {'/'.join(source_key)}, these "
            f"sources require {'/'.join(target_key)}: sources, arch or DSL differ; "
            "use the keyed ICP_CACHE_ROOT placement")
    objects = _aot_objects(source, source_key[0])
    target.mkdir(parents=True, exist_ok=True)
    for obj in objects:
        shutil.copyfile(obj, target / obj.name)
        metadata = obj.with_suffix(".json")
        shutil.copyfile(metadata, target / metadata.name)
    shutil.copyfile(source / "manifest.json", target / "manifest.json")


def build(arch: str) -> list[tuple[str, pathlib.Path]]:
    """Build every artifact; returns ``(name, path)`` pairs for the manifest."""
    _configure()
    _load_k2q_extension()
    target = aot_dir(arch)
    import torch  # noqa: PLC0415

    if torch.cuda.is_available():
        for layout in PREWARM_LAYOUTS:
            for variant in PREWARM_VARIANTS:
                production_call(*variant, layout=layout)
        torch.cuda.synchronize()
    else:
        source = os.environ.get(IMPORT_ENV)
        if not source:
            raise SystemExit(
                f"{COMPONENT}: the CuTe AOT objects need one CUDA-device run "
                f"(`python -m fmha_sm100.icp.prewarm build --components {COMPONENT}` "
                f"on a GPU), or {IMPORT_ENV}=<the aot dir of such a run>")
        _import_aot_objects(pathlib.Path(source), target)
    objects = _aot_objects(target, arch)
    return ([("k2q_ext", k2q_extension_path()),
             ("aot-manifest.json", target / "manifest.json")]
            + [(o.stem, o) for o in objects]
            + [(o.stem + ".json", o.with_suffix(".json")) for o in objects])
