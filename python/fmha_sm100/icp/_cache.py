"""Where every ICP JIT artifact lives: keyed by component source digest and arch.

One cache root, ``$ICP_CACHE_ROOT`` (default ``~/.cache/minfer/msa_icp``), laid out

    <root>/sm_<arch>/<component>-<digest16>/...     one dir per component
    <root>/sm_<arch>/PREWARM-MANIFEST.json          written by fmha_sm100.icp.prewarm

``<digest16>`` is over the component's own compile inputs (below), so an edit
moves the component to a new, empty directory: a stale object cannot be served
for new sources, and mounting a newer checkout over a prewarmed image recompiles
exactly the components whose sources changed. ``<arch>`` is ``107a``-style.
The shared MSA JIT and AOT caches additionally validate their own toolchain and
transitive source records before loading an artifact.

Components and their inputs (relative to ``fmha_sm100/``):

    icp     icp/csrc/*.cu, *.cuh, icp/_build.py  selection/merge/transport
    fmha    csrc/**, jit.py, _jit_cache.py, CUTLASS shared FMHA/plan/top-k
    decode  icp/scorer/decode/*.py + DSL version CuTe decode scorer
    nvfp4   cute/**/*.py, *.cu + DSL version    shared sparse attention/CSR

The NVFP4 CSR extension uses a fingerprinted ``cpp_extension`` directory
(``TORCH_EXTENSIONS_DIR``). Its effective artifact path is recorded and checked.
The KV writer belongs to the serving engine and is not built by this package.

Explicit per-component overrides (``ICP_KERNEL_CACHE``,
``MINFER_FMHA_CACHE_DIR``) are honoured for development and are UNKEYED;
:mod:`fmha_sm100.icp.prewarm` rejects them for strict serving prewarm.

Importing this module imports nothing outside the standard library.
"""

from __future__ import annotations

import hashlib
import os
import pathlib

PACKAGE = pathlib.Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE.parent
ROOT_ENV = "ICP_CACHE_ROOT"
MANIFEST_NAME = "PREWARM-MANIFEST.json"
FALLBACK_ARCH = "107a"

_FMHA_SUFFIXES = (".h", ".hpp", ".cuh", ".cu", ".jinja")

#: Explicit unkeyed overrides, per component.
LEGACY_ENV = {"icp": "ICP_KERNEL_CACHE", "fmha": "MINFER_FMHA_CACHE_DIR"}


def _component_files(component: str) -> list[pathlib.Path]:
    if component == "icp":
        csrc = PACKAGE / "csrc"
        files = [p for p in csrc.iterdir() if p.suffix in (".cu", ".cuh")]
        files.append(PACKAGE / "_build.py")
    elif component == "fmha":
        prefill = SOURCE_ROOT
        files = [p for p in (prefill / "csrc").rglob("*")
                 if p.is_file() and p.suffix in _FMHA_SUFFIXES]
        # A header/toolchain replacement must not reuse the previous cubins.
        for relative in ("cutlass/include", "cutlass/tools/util/include"):
            files.extend(p for p in (prefill / relative).rglob("*") if p.is_file())
        files.extend(prefill / name for name in ("jit.py", "_jit_cache.py"))
    elif component == "decode":
        files = list((PACKAGE / "scorer" / "decode").glob("*.py"))
    elif component == "nvfp4":
        root = SOURCE_ROOT / "cute"
        files = [p for p in root.rglob("*")
                 if p.suffix in (".py", ".cu") and "__pycache__" not in p.parts]
    else:
        raise KeyError(f"unknown cache component {component!r}; known: {COMPONENTS}")
    files = sorted(p for p in files if p.is_file())
    if not files:
        raise RuntimeError(f"component {component!r} has no source files under "
                           f"{PACKAGE}; its digest would certify anything")
    return files


COMPONENTS = ("icp", "fmha", "decode", "nvfp4")


def _extra_key(component: str) -> str:
    # The decode scorer's machine code is produced by the DSL, not by nvcc, so
    # the DSL release is a compile input like any source file.
    if component in ("decode", "nvfp4"):
        try:
            from importlib.metadata import version  # noqa: PLC0415

            return f"nvidia-cutlass-dsl=={version('nvidia-cutlass-dsl')}"
        except Exception:  # noqa: BLE001 - absent DSL: nothing to compile anyway
            return "nvidia-cutlass-dsl==absent"
    return ""


def file_hashes(component: str) -> dict[str, str]:
    """``{path relative to fmha_sm100.icp/: sha256}`` for one component's inputs."""
    return {p.relative_to(SOURCE_ROOT).as_posix():
            hashlib.sha256(p.read_bytes()).hexdigest()
            for p in _component_files(component)}


def component_digest(component: str) -> str:
    h = hashlib.sha256()
    for rel, digest in sorted(file_hashes(component).items()):
        h.update(f"{rel}\0{digest}\n".encode())
    h.update(_extra_key(component).encode())
    return h.hexdigest()


def normalize_arch(value) -> str:
    """``"sm_107a"``, ``"107A"``, ``(10, 7)`` -> ``"107a"``."""
    if isinstance(value, tuple):
        major, minor = value
        return f"{major}{minor}a"
    text = str(value).strip().lower().removeprefix("sm_")
    if not text.rstrip("a").isdigit():
        raise ValueError(f"not an arch string: {value!r}")
    return text if text.endswith("a") else text + "a"


def _nvml_arch() -> str | None:
    try:
        import pynvml  # noqa: PLC0415
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        try:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible == "":
                return None
            first = (visible or "0").split(",")[0].strip()
            if first.isdigit():
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(first))
            else:
                handle = pynvml.nvmlDeviceGetHandleByUUID(first)
            return normalize_arch(pynvml.nvmlDeviceGetCudaComputeCapability(handle))
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001 - no driver / no device: fall through
        return None


def key_arch() -> str:
    """The arch half of the key, WITHOUT creating a CUDA context.

    ``ICP_KERNEL_ARCH`` > an already-initialised torch device > NVML >
    :data:`FALLBACK_ARCH`. Import-time callers (the FMHA ``jit`` module) must
    not initialise CUDA, which is why torch is consulted only if it already did.
    """
    override = os.environ.get("ICP_KERNEL_ARCH")
    if override:
        return normalize_arch(override)
    try:
        import sys  # noqa: PLC0415

        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            return normalize_arch(tuple(torch.cuda.get_device_capability()))
    except Exception:  # noqa: BLE001 - a stub or broken torch: fall through
        pass
    return _nvml_arch() or FALLBACK_ARCH


def root() -> pathlib.Path:
    return pathlib.Path(
        os.environ.get(ROOT_ENV) or os.path.expanduser("~/.cache/minfer/msa_icp"))


def arch_dir(arch: str | None = None) -> pathlib.Path:
    return root() / f"sm_{normalize_arch(arch or key_arch())}"


def component_dir(component: str, arch: str | None = None) -> pathlib.Path:
    """The keyed directory; honours the unkeyed legacy override if one is set."""
    legacy = LEGACY_ENV.get(component)
    if legacy and os.environ.get(legacy):
        return pathlib.Path(os.environ[legacy])
    return arch_dir(arch) / f"{component}-{component_digest(component)[:16]}"


def manifest_path(arch: str | None = None) -> pathlib.Path:
    return arch_dir(arch) / MANIFEST_NAME
