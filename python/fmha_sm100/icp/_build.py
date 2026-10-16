"""ICP extension loader, digest-keyed cache directories and the runtime-JIT refusal."""

from __future__ import annotations

import functools
import hashlib
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_CSRC = _HERE / "csrc"

# sm_103 takes the plain (non-`a`) gencode form.
_PLAIN_FORM_FAMILIES = frozenset({"103"})
_FALLBACK_ARCH = "107a"

#: extension name -> sources under ``csrc/``.
EXTENSIONS: dict[str, list[str]] = {
    "icp_k2": ["k2_merge.cu"],
    "icp_fused": ["fused_exchange.cu"],
    "icp_pack": ["carrier_pack.cu"],
    "icp_select": ["local_candidates.cu"],
}

_loaded: dict[str, object] = {}


def arch() -> str:
    """Target arch such as ``"107a"``: ICP_KERNEL_ARCH, else the current torch device."""
    override = os.environ.get("ICP_KERNEL_ARCH")
    if override:
        return override.strip().lower().removeprefix("sm_")
    import torch

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return f"{major}{minor}a"
    return _FALLBACK_ARCH


def normalize_arch(value) -> str:
    """``"sm_107a"``, ``"107"`` or ``(10, 7)`` -> ``"107a"``: the arch half of a cache key."""
    if isinstance(value, tuple):
        return f"{value[0]}{value[1]}a"
    text = str(value).strip().lower().removeprefix("sm_")
    return text if text.endswith("a") else text + "a"


def _gencode(arch_str: str) -> str:
    family = arch_str.rstrip("a")
    if family in _PLAIN_FORM_FAMILIES:
        return f"-gencode=arch=compute_{family},code=sm_{family}"
    return f"-gencode=arch=compute_{arch_str},code=sm_{arch_str}"


def refuse_jit(component: str, name: str) -> None:
    """Raise RuntimeError instead of compiling ``component:name`` when ICP_RUNTIME_JIT=fail."""
    if os.environ.get("ICP_RUNTIME_JIT") == "fail":
        raise RuntimeError(
            f"runtime JIT of {component}:{name} refused (ICP_RUNTIME_JIT=fail)")


def _inputs(component: str) -> list[pathlib.Path]:
    if component == "icp":
        return [*(p for p in _CSRC.iterdir() if p.suffix in (".cu", ".cuh")),
                _HERE / "_build.py"]
    return list((_HERE / "scorer" / "decode").glob("*.py"))


@functools.cache
def _digest(component: str) -> str:
    files = {p.relative_to(_HERE.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in _inputs(component) if p.is_file()}
    h = hashlib.sha256()
    for rel, digest in sorted(files.items()):
        h.update(f"{rel}\0{digest}\n".encode())
    if component == "decode":
        # The DSL emits the decode scan's machine code, so its release keys the cache too.
        try:
            from importlib.metadata import version

            h.update(f"nvidia-cutlass-dsl=={version('nvidia-cutlass-dsl')}".encode())
        except Exception:  # noqa: BLE001 - absent DSL: nothing can compile anyway
            h.update(b"nvidia-cutlass-dsl==absent")
    return h.hexdigest()


def component_dir(component: str, arch_str) -> pathlib.Path:
    """``$ICP_CACHE_ROOT/sm_<arch>/<component>-<digest16>`` for component "icp" or "decode"."""
    root = os.environ.get("ICP_CACHE_ROOT") or os.path.expanduser("~/.cache/minfer/msa_icp")
    key = normalize_arch(arch_str)
    return pathlib.Path(root) / f"sm_{key}" / f"{component}-{_digest(component)[:16]}"


def _load_extension(name: str):
    """Load one :data:`EXTENSIONS` module, compiling it into its keyed directory if absent."""
    module = _loaded.get(name)
    if module is not None:
        return module
    from torch.utils.cpp_extension import load

    arch_str = arch()
    directory = component_dir("icp", arch_str) / f"{name}_sm{arch_str}"
    if not (directory / f"{name}.so").is_file():
        refuse_jit("icp", name)
        print(f"[fmha_sm100.icp] building {name} for sm_{arch_str} ({_gencode(arch_str)})",
              file=sys.stderr)
    directory.mkdir(parents=True, exist_ok=True)
    module = _loaded[name] = load(
        name=name,
        sources=[str(_CSRC / source) for source in EXTENSIONS[name]],
        # No `-std=`: cpp_extension supplies the one torch's headers need.
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", _gencode(arch_str),
                           "--expt-relaxed-constexpr"],
        extra_include_paths=[str(_CSRC)],
        build_directory=str(directory),
    )
    return module


def _k2():
    """The K2 merge of the NCCL route."""
    return _load_extension("icp_k2")


def _fused():
    """The D3 fused decode exchange."""
    return _load_extension("icp_fused")


def _pack():
    """The C4 send-carrier pack."""
    return _load_extension("icp_pack")


def _select_ext():
    """The prefill local-candidate selector."""
    return _load_extension("icp_select")
