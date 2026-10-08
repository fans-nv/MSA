"""Runtime-JIT loader and arch handling for the ICP kernels.

Everything in :mod:`fmha_sm100.icp` is built with
``torch.utils.cpp_extension.load`` at first use rather than ahead of time. The
reason is recorded in the MiniMax-M3 kernel work: an AOT object cache that keys
on neither the arch nor the source hash silently serves a stale ``.o``, and a
stale ``.o`` is indistinguishable from a correct build until the numbers are
wrong. JIT with an explicit arch is the only form that cannot lie about what
ran.

Arch handling
-------------

**The arch must be spelled out, and it must be derived from the live device.**
No container's ``TORCH_CUDA_ARCH_LIST`` has carried the targets this repo cares
about: the Rubin image's has no 10.7, and the GB300 image's is
``8.0 8.7 8.9 9.0 10.0 11.0 12.0`` -- no 10.3. A prior version of this loader
hard-coded ``_ARCH = "107a"`` and died on GB300 with "no kernel image is
available for execution on the device", which is what :func:`arch` and its
``ICP_KERNEL_ARCH`` override exist to prevent.

The ``-gencode`` *form* is per-arch data, not a style choice. The rule, the
per-arch exception and the evidence for both live at :func:`gencode_flags` and
:data:`_PLAIN_FORM_FAMILIES`, next to the code that applies them.

Environment
-----------

**Never set ``TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES``.** It looks like free
insurance against "detected allocations from overlapping devices from different
ranks", and it is not: ``CUDASymmetricMemory.cu:853`` skips
``init_multicast_for_block`` when it is set, which **silently disables NVLS
multicast**. The overlapping-devices error is the allocator correctly refusing
two ranks that both report CUDA ordinal 0; the fix is
``torch.cuda.set_device(local_rank)`` *before* ``init_process_group()`` and no
per-rank ``CUDA_VISIBLE_DEVICES`` masking. See :class:`fmha_sm100.icp.IcpExchange`.

``TORCH_SYMMMEM`` and ``NVSHMEM_MAX_TEAMS`` stay unset: this repo uses the CUDA
symmetric-memory backend, and the NVSHMEM backend cannot fit its team pool on
NVLS hardware at any DP/EP size.

CPU-only build gate
-------------------

::

    python -m fmha_sm100.icp._build              # every extension, live arch
    ICP_KERNEL_ARCH=90a python -m fmha_sm100.icp._build --compile-only k2 k5

Every accessor here is PRIVATE and returns the raw pybind module, which
carries neither band's rules. The typed modules -- :mod:`fmha_sm100.icp.candidates`,
:mod:`fmha_sm100.icp.exchange`, :mod:`fmha_sm100.icp.merge` -- are the public path.

``--compile-only`` shells out to ``nvcc -c`` with torch's include paths and then
prints ``cuobjdump -res-usage``. It never creates a CUDA context and never links
against libtorch, so it is safe to run on a node whose GPUs another lane is
timing on -- which is the standing check after touching a tuned kernel, because
a ``__launch_bounds__`` or register-spill regression is otherwise silent.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import re
import subprocess
import sys

from . import _cache, _jit_guard

_HERE = pathlib.Path(__file__).resolve().parent
_CSRC = _HERE / "csrc"
# The cache root is keyed by source digest and arch in `_cache.component_dir`.

#: Arch families whose ``-gencode`` must use the plain (non-``a``) form. See the
#: module docstring: on these, the ``a`` form additionally emits a PTX entry,
#: i.e. a silent driver-JIT fallback.
_PLAIN_FORM_FAMILIES = frozenset({"103"})

#: Used only when there is no device to ask and no override -- a docs build or
#: ``--compile-only`` on a workstation. A *run* that landed here would build for
#: the wrong GPU, so :func:`arch` prefers the live capability and the env
#: override, and :func:`load` says loudly which arch it used.
_FALLBACK_ARCH = "107a"

# No `-std=` on purpose. `cpp_extension.load` appends the standard *its own*
# headers require -- c++17 on torch 2.6, c++20 from 2.9 -- but only if the
# caller did not already pass one, and a caller-supplied `-std=c++17` wins on
# both gcc and nvcc. Pinning c++17 here therefore breaks the build on any
# modern torch with "#error C++20 or later compatible compiler is required to
# use PyTorch", which is a confusing way to say "this flag list is stale".
_CXX_FLAGS = ["-O3"]

#: Standards to try, newest first, for the toolchain-independent
#: :func:`compile_check` path (which assembles its own command line and so does
#: not get torch's answer for free).
_STD_CANDIDATES = ("c++20", "c++17")

#: name -> sources, relative to ``csrc/``. ``merge_topk.cuh`` is compiled once
#: per extension **on purpose**: sharing the *source* is what makes K2, K5 and
#: the fused path bit-identical, and a shared object would not be a stronger
#: guarantee than a shared header.
EXTENSIONS: dict[str, list[str]] = {
    "icp_k2": ["k2_merge.cu"],
    "icp_k5": ["k5_exchange.cu"],
    "icp_k5t": ["k5t_exchange.cu"],
    "icp_fused": ["fused_exchange.cu"],
    "icp_pack": ["carrier_pack.cu"],
    "icp_select": ["local_candidates.cu"],
    "icp_select_control": ["local_candidates_rs_control.cu"],
}

#: Extra preprocessor definitions, per extension. Keyed by extension *name*, so
#: a defined and an undefined build of the same sources are two names, two build
#: directories and two sets of objects -- never one cache entry serving both.
DEFINES: dict[str, list[str]] = {
    "icp_select_control": ["-DICP_RS_NEGATIVE_CONTROL=1"],
}

#: Short names for the CLI, so `python -m fmha_sm100.icp._build k5` works.
ALIASES = {
    "k2": "icp_k2",
    "k5": "icp_k5",
    "k5t": "icp_k5t",
    "fused": "icp_fused",
    "pack": "icp_pack",
    "select": "icp_select",
    "control": "icp_select_control",
}

_loaded: dict[str, object] = {}
_arch_reported: set[str] = set()


# --------------------------------------------------------------------------
# arch
# --------------------------------------------------------------------------


def arch() -> str:
    """The arch string to build for, e.g. ``"107a"``, ``"103a"``, ``"90a"``.

    Order: ``ICP_KERNEL_ARCH`` > the live device capability > ``_FALLBACK_ARCH``.
    Deriving from ``torch.cuda.get_device_capability`` is what lets the same
    tree build ``sm_107a`` on Rubin and ``sm_103`` on GB300 with no edit.
    """
    override = os.environ.get("ICP_KERNEL_ARCH")
    if override:
        # Normalised, so "sm_107A", "SM_107a" and "107a" are one arch and one
        # build directory rather than three.
        return override.strip().lower().removeprefix("sm_")
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            # Always the `a` (architecture-specific) suffix from detection:
            # every target this repo builds for is an `a`-form part, and
            # `gencode_flags` strips it back off for the families that need the
            # plain form. Detection therefore never has to know the exception.
            return f"{major}{minor}a"
    except Exception:  # pragma: no cover - no torch, no driver
        pass
    return _FALLBACK_ARCH


def gencode_flags(arch_str: str | None = None) -> list[str]:
    """The ``-gencode`` flags for one arch, in the form that arch requires.

    One rule and one exception, both about never emitting a **PTX entry**. A
    PTX entry is not a safety net: it turns a loud build failure into a silent
    multi-hundred-millisecond driver JIT on the first launch, which is
    invisible in the output and shows up only as a slow first iteration.

    The rule -- the ``a``-form targets, e.g. ``sm_107a``
        Two-part ``-gencode=arch=compute_107a,code=sm_107a``, which emits a
        *cubin only*. The two rejected spellings both emit PTX:
        ``code=compute_107a`` is PTX-only, and ``-arch=sm_107a`` emits both
        entries.

    The exception -- ``_PLAIN_FORM_FAMILIES``, currently ``sm_103`` (GB300)
        The **plain** form ``-gencode=arch=compute_103,code=sm_103``, because
        on this family it is the ``a`` spelling that adds the unwanted
        ``sm_103`` PTX entry.

        **Measured on a GB300, 2026-09-12.** All four extensions build for
        ``sm_103a`` and ``cuobjdump`` reports exactly one ELF entry and no PTX
        at all::

            cuobjdump -lelf icp_k2.so  ->  ELF file 1: icp_k2.1.sm_103.cubin
            cuobjdump -lptx icp_k2.so  ->  No PTX file found to extract

        which is the property the plain form is chosen for. A kernel that
        needed an ``a``-form-only instruction (``cvt.rn.satfinite.e2m1x2.f32``,
        for instance) would have to take the ``a`` form instead, and under nvcc
        13.2.51 ``-arch=sm_103a`` writes ``.target sm_103a`` into the PTX but
        then calls ptxas with ``-arch=sm_103``. Nothing in this repo's CUDA
        needs such an instruction, so the plain form is both correct and free
        of that trap -- but the choice is per-arch data in
        :data:`_PLAIN_FORM_FAMILIES`, one line to change, and
        ``ICP_KERNEL_ARCH`` overrides the detection entirely.
    """
    arch_str = (arch_str or arch()).strip().lower().removeprefix("sm_")
    # The family is the arch with any trailing `a` dropped: "107a" -> "107".
    # The `a` suffix selects the architecture-specific (non-portable) ISA, so
    # it is part of the *target* name but never part of the family key.
    family = arch_str.rstrip("a")
    if not family.isdigit():
        raise ValueError(
            f"ICP_KERNEL_ARCH={arch_str!r} is not an arch string like '107a' "
            "or '103'"
        )
    if family in _PLAIN_FORM_FAMILIES:
        return [f"-gencode=arch=compute_{family},code=sm_{family}"]
    return [f"-gencode=arch=compute_{arch_str},code=sm_{arch_str}"]


def cuda_flags(arch_str: str | None = None) -> list[str]:
    # See `_CXX_FLAGS`: no `-std=`, so `cpp_extension.load` supplies the one
    # its own headers need.
    return [
        "-O3",
        "--use_fast_math",
        *(["-lineinfo"] if os.environ.get("ICP_KERNEL_LINEINFO") == "1" else []),
        *gencode_flags(arch_str),
        "--expt-relaxed-constexpr",
    ]


# --------------------------------------------------------------------------
# JIT
# --------------------------------------------------------------------------


def build_dir(name: str, arch_str: str | None = None) -> str:
    # The arch is in the *directory name* on purpose: that is what stops one
    # arch's objects from being served for another, which is the exact failure
    # (a stale `.o` keyed on neither arch nor source hash) this module's JIT
    # approach exists to rule out.
    arch_str = arch_str or arch()
    directory = os.path.join(_cache.component_dir("icp", arch_str),
                             f"{name}_sm{arch_str}")
    os.makedirs(directory, exist_ok=True)
    return directory


def _load_extension(name: str, sources: list[str] | None = None, *,
                    verbose: bool = False):
    """JIT-build and cache one extension. ``sources`` default to :data:`EXTENSIONS`."""
    if name in _loaded:
        return _loaded[name]
    if sources is None:
        try:
            sources = EXTENSIONS[name]
        except KeyError:
            raise KeyError(
                f"unknown extension {name!r}; known: {sorted(EXTENSIONS)}"
            ) from None
    from torch.utils.cpp_extension import load as _load  # noqa: PLC0415

    arch_str = arch()
    # Announced once per arch, not once per extension: which arch was actually
    # built for is the one fact that distinguishes a correct build from one
    # that will die with "no kernel image is available for execution".
    if arch_str not in _arch_reported:
        _arch_reported.add(arch_str)
        print(f"[fmha_sm100.icp] building for sm_{arch_str} "
              f"({' '.join(gencode_flags(arch_str))})", file=sys.stderr)
    directory = build_dir(name, arch_str)
    if not os.path.exists(os.path.join(directory, f"{name}.so")):
        _jit_guard.on_compile("icp", name, arch=_cache.normalize_arch(arch_str),
                              where=directory)
    # --- the header-freshness stamp ----------------------------------------
    #
    # torch's JIT versioner hashes the LISTED sources and the build arguments.
    # Every `.cuh` here -- including `merge_topk.cuh`, the single implementation
    # of the C5 key and of the duplicate max-reduce -- reaches the compiler
    # through `#include` and is listed nowhere, so a header-only edit does not
    # move torch's version. MEASURED on GB300 / torch 2.14: across processes the
    # rebuild happens anyway, because a fresh process has no recorded version
    # and ninja's depfile then catches the header; but WITHIN one process, after
    # the extension has been loaded once, a header edit is invisible and the
    # stale object is served. That is the "a source edit does not prove a
    # rebuild" hole, and an interactive kernel session is exactly where it bites.
    #
    # The fix names the dependency rather than widening a sweep: the digest of
    # the header set goes into a build ARGUMENT, which is the thing torch does
    # hash. Only `.cuh` files are digested -- adding a `.cuh` to `sources` would
    # ask ninja to compile a header, and hashing the rendered object instead
    # would miss a flag change.
    #
    # The macro is never referenced, so it cannot change code generation.
    # Verified rather than assumed: k2_merge_kernel is 22/30/32 registers with
    # zero spills for KPT 1/2/4 with and without the stamp.
    header_define = [f"-DICP_CSRC_HEADER_DIGEST=0x{header_digest()}ULL"]
    module = _load(
        name=name,
        sources=[str(_CSRC / source) for source in sources],
        extra_cuda_cflags=(cuda_flags(arch_str) + DEFINES.get(name, [])
                           + header_define),
        extra_cflags=_CXX_FLAGS + header_define,
        extra_include_paths=[str(_CSRC)],
        build_directory=directory,
        verbose=verbose,
    )
    _loaded[name] = module
    return module


# --------------------------------------------------------------------------
# the extensions. PRIVATE, one accessor each: they return the raw pybind
# module, which carries neither band's rules. The typed modules are the public
# path.
# --------------------------------------------------------------------------


def _k2(verbose: bool = False):
    """The K2 merge extension (host ABI ``refined-icp-v1.k2.2``).

    Exports ``k2_merge(cand, out, head_offset, world, rank, forced,
    n_ordinary, status)``; ``canonical_key(scores, ids, out)``, a probe over
    ``icp::canonical_key`` so a test can compare the shipped C5 key against a
    reference without a second copy of it existing anywhere; and the three
    capability attributes ``k2_abi_version``, ``k2_carrier`` and the
    ``k2_status_*`` bit values.
    """
    return _load_extension("icp_k2", verbose=verbose)


def _k5(verbose: bool = False):
    """The K5 fused symmetric-memory exchange + merge extension.

    Exports ``k5_exchange(...)``, ``k5_plan(T, H_local, max_blocks)``,
    ``k5_ctl_words(slots, nblocks, world)``, ``k5_slot_floats(T, H_group,
    world, opts)`` and ``k5_opts()``.
    """
    return _load_extension("icp_k5", verbose=verbose)


def _k5t(verbose: bool = False):
    """The K5T tiled symmetric-memory push + merge extension (every extent).

    Exports ``k5t_exchange(...)``, ``k5t_plan(T, T_cap, H_local, world,
    max_ctas)``, ``k5t_slot_words(T_cap, H_group)`` and the ``k5t_*`` ABI
    attributes.
    """
    return _load_extension("icp_k5t", verbose=verbose)


def _fused(verbose: bool = False):
    """The D3 fused decode exchange: selector-side publish + poll/merge.

    Exports ``fused_select_publish(...)``, ``fused_merge(...)``,
    ``fused_plan(T, T_cap, H_local, world, max_ctas)``,
    ``fused_slot_words(T_cap, H_group)`` and the ``fused_*`` ABI attributes.
    """
    return _load_extension("icp_fused", verbose=verbose)


def _pack(verbose: bool = False):
    """The native C4 send-carrier pack. Exports ``pack_send_carrier``."""
    return _load_extension("icp_pack", verbose=verbose)


def _select_ext(verbose: bool = False):
    """The local-candidate selector extension.

    PRIVATE: it returns the raw pybind module, which carries neither band's
    rules. ``fmha_sm100.icp.candidates`` is the public path.

    Exports ``select_prefill_candidates(..., live_blocks, arm=-2)`` and
    ``prefill_kernel_attributes(live_blocks, arm=-2)`` for the eager prefill
    band (bounded radix or explicit whole-row selection); and
    ``select_local_candidates(...)`` (the sort arm),
    ``select_local_candidates_radix(...)``,
    ``select_local_candidates_radix_sr(...)`` (the shipping arm),
    ``select_local_candidates_capped(..., cap)`` and
    ``selector_kernel_attributes(cap, blocks)`` for the captured decode band.
    ``set_launch_recording(on)``, ``last_launch_info()`` and the additive
    ``last_launch_threads()`` accessor serve both.
    """
    return _load_extension("icp_select", verbose=verbose)


def _select_control_ext(verbose: bool = False):
    """The selector's negative-control extension. TEST ONLY, and private.

    Built from ``local_candidates_rs_control.cu`` with
    ``ICP_RS_NEGATIVE_CONTROL`` defined, which is what makes the perturbation
    parameter exist at all. Exports
    ``select_local_candidates_control(..., control, cap)``.
    """
    return _load_extension("icp_select_control", verbose=verbose)


# --------------------------------------------------------------------------
# artifact identity: what was built, from which bytes
# --------------------------------------------------------------------------


def header_digest(csrc: pathlib.Path | None = None) -> str:
    """16 hex chars over every ``.cuh`` in ``csrc``, in name order.

    This is the **freshness stamp**, and it exists because torch's JIT versioner
    hashes the listed sources and the build arguments and nothing else. Every
    header here -- including ``merge_topk.cuh``, the single implementation of
    the C5 key and of the duplicate max-reduce -- reaches the compiler through
    ``#include`` and is listed nowhere, so a header-only edit does not move
    torch's version. :func:`_load_extension` folds this digest into the build
    arguments, which torch *does* hash, so it does.

    Two things this deliberately is not. It is not a widened suffix sweep: only
    ``.cuh`` is digested, because those are the files that reach a compiler
    invisibly. And it is not a digest of the produced object: the object is not
    reproducible build to build (measured -- rebuilding from byte-identical
    sources yields a different ``.so``), so hashing the output would answer a
    different and less useful question.

    ``csrc`` is a parameter purely so the property is testable without editing
    the shipped headers.
    """
    root = csrc if csrc is not None else _CSRC
    headers = sorted(p for p in root.iterdir() if p.suffix == ".cuh")
    return hashlib.sha256(
        b"".join(p.read_bytes() for p in headers)
    ).hexdigest()[:16]


def source_digest() -> dict[str, str]:
    """sha256 of every CUDA source and header this package builds from.

    Keyed by filename, sorted. Pure filesystem, no torch and no device, so a
    consumer can record it at install time and compare it later.
    """
    out = {}
    for path in sorted(_CSRC.iterdir()):
        if path.suffix in (".cu", ".cuh") and path.is_file():
            out[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def loaded_artifacts() -> dict[str, dict[str, object]]:
    """What is **actually loaded**, per extension, with its object's sha256.

    A source edit does not prove a rebuild, and a stale ``.o`` keyed on neither
    the arch nor the source hash is the exact failure this module's JIT layout
    exists to prevent -- so this reports the artifact rather than the intent.
    For every extension built in this process it returns the build directory,
    the ``.so`` path, the ``.so``'s sha256 and the arch it was built for; the
    ``sources`` entry is the same digest :func:`source_digest` returns.

    **What the two halves do and do not prove.** The source digests are
    reproducible and are the thing to compare across machines and across time.
    The ``.so`` digest is **not**: measured on GB300 sm_103 with torch 2.14,
    rebuilding from byte-identical sources yields a different ``.so`` hash, so
    the object is not reproducible build-to-build. Use ``so_sha256`` to say
    *which object this process loaded* -- which is what "a source edit does not
    prove a rebuild" needs -- and use ``sources`` to say *which bytes it claims
    to be*. Comparing a ``.so`` hash against one recorded on another machine
    will differ for no reason at all.

    Note also that only the ``.cu`` translation units appear under each
    extension's ``sources``; the ``.cuh`` headers reach the compiler through
    ``#include`` and are covered by :func:`source_digest`'s whole-directory
    digest. ``tests/test_abi.py`` gates that a header-only edit still rebuilds.

    Extensions that have not been loaded yet are absent rather than built: this
    is a report, not a trigger.
    """
    arch_str = arch()
    sources = source_digest()
    report: dict[str, dict[str, object]] = {}
    for name in _loaded:
        directory = build_dir(name, arch_str)
        so_path = os.path.join(directory, f"{name}.so")
        entry: dict[str, object] = {
            "arch": arch_str,
            "build_dir": directory,
            "so": so_path,
            "sources": {s: sources[s] for s in sorted(EXTENSIONS[name])
                        if s in sources},
        }
        if os.path.exists(so_path):
            with open(so_path, "rb") as handle:
                entry["so_sha256"] = hashlib.sha256(handle.read()).hexdigest()
        else:
            entry["so_sha256"] = None
        report[name] = entry
    return report


# --------------------------------------------------------------------------
# the register profile, and the baseline a test asserts it against
# --------------------------------------------------------------------------

#: The K2 merge's register/spill profile, per ``KPT`` (candidates per lane), on
#: ``sm_103``. ``KPT = ceil(W*16 / 32)``, so 1/2/4 is ``W`` 2/4/8 and only 1 and
#: 2 are supported degrees.
#:
#: **This is the current baseline and it is not the historical one.** It was
#: 22/28/32 while the merge read the fp32 gathered tensor. The C4 int32 loader
#: costs +2 at KPT=2, which a compile-only bisect attributes precisely -- S1
#: 22/28/32, S7 22/28/32, M1 22/30/32, on both ``-gencode`` forms and reproduced
#: against M1's unmodified file. The C4 carrier is required by the frozen
#: ``refined-icp-v1`` ABI, so the loader is not optional.
#:
#: **22/28/32 is unreachable, not merely missed.** Two semantics-preserving
#: rewrites of the loader's address arithmetic (hoisting its loop-invariant part,
#: and additionally reading the record as one ``int2``) both measure 24/28/32:
#: they *move* the two registers from ``W = 4`` to ``W = 2`` rather than
#: recovering them. Do not "fix" this back without re-measuring -- doing so
#: silently takes the 24 at ``W = 2``. Zero spills in every variant.
MERGE_REGISTER_BASELINE = {1: 22, 2: 30, 4: 32}


def merge_register_profile(name: str = "icp_k2") -> dict[int, dict[str, int]]:
    """``{KPT: {"regs", "stack", "local"}}`` for the **loaded** K2 merge.

    Read out of ``cuobjdump --dump-resource-usage`` on the ``.so`` that was
    actually loaded, because a register or spill regression is otherwise silent
    and because the source alone does not tell you what ptxas did with it.
    Compare against :data:`MERGE_REGISTER_BASELINE`.

    Returns an empty dict if the extension has not been built or ``cuobjdump``
    is not on ``PATH`` -- it is a measurement, not a requirement.
    """
    entry = loaded_artifacts().get(name)
    if entry is None or not entry.get("so") or not os.path.exists(entry["so"]):
        return {}
    dump = subprocess.run(["cuobjdump", "--dump-resource-usage", entry["so"]],
                          capture_output=True, text=True, check=False)
    if dump.returncode != 0:
        return {}
    profile: dict[int, dict[str, int]] = {}
    kpt = None
    for line in dump.stdout.splitlines():
        match = re.search(r"k2_merge_kernelILi(\d)EE", line)
        if match:
            kpt = int(match.group(1))
            continue
        if kpt is None:
            continue
        fields = dict(re.findall(r"(REG|STACK|LOCAL):(\d+)", line))
        if fields:
            profile[kpt] = {"regs": int(fields["REG"]),
                            "stack": int(fields["STACK"]),
                            "local": int(fields["LOCAL"])}
            kpt = None
    return profile


def ptxas_info(name: str = "icp_k2") -> str:
    """``cuobjdump -res-usage`` for a built extension, for the CSV's provenance.

    A ``__launch_bounds__`` or spill regression is silent otherwise.
    """
    directory = pathlib.Path(build_dir(name))
    artifacts = sorted(directory.glob("*.cubin")) + sorted(directory.glob("*.o"))
    if not artifacts:
        return f"no build artifacts in {directory}"
    dump = subprocess.run(
        ["cuobjdump", "-res-usage", str(artifacts[0])],
        capture_output=True,
        text=True,
        check=False,
    )
    return dump.stdout or dump.stderr


# --------------------------------------------------------------------------
# CPU-only compile gate
# --------------------------------------------------------------------------


def _torch_include_paths() -> list[str]:
    """Torch's header directories, without requiring a CUDA-enabled torch.

    ``include_paths(device_type="cuda")`` insists on ``CUDA_HOME``, which a
    CPU-only wheel does not set even when nvcc is right there on ``PATH``. So
    derive it from nvcc, and if that fails fall back to the CPU include set --
    nvcc supplies its own toolkit headers either way, and the CUDA bits this
    repo's sources need (``c10/cuda/CUDAStream.h``) ship inside the torch
    wheel.
    """
    import shutil  # noqa: PLC0415

    from torch.utils.cpp_extension import include_paths  # noqa: PLC0415

    if not os.environ.get("CUDA_HOME"):
        nvcc = shutil.which("nvcc")
        if nvcc:
            os.environ["CUDA_HOME"] = str(pathlib.Path(nvcc).parent.parent)
    for kwargs in ({"device_type": "cuda"}, {"cuda": True}, {}):
        try:
            return include_paths(**kwargs)
        except (TypeError, OSError, RuntimeError, EnvironmentError):
            continue
    return include_paths()


def compile_check(name: str, *, arch_str: str | None = None,
                  verbose: bool = False) -> str:
    """Compile one extension to an object with ``nvcc -c`` and dump res-usage.

    Compile-only and link-free on purpose: it needs neither a GPU, nor a CUDA
    driver, nor a CUDA-enabled torch build -- only torch's headers and an nvcc
    that knows the target arch. It therefore runs while another lane owns every
    GPU on the node, which is the situation this gate was written for.
    """
    arch_str = arch_str or arch()
    _require_cuda_torch_headers()
    out_dir = pathlib.Path(build_dir(f"{name}_check", arch_str))
    obj = out_dir / f"{name}.o"
    source = _CSRC / EXTENSIONS[name][0]
    includes = [
        f"-I{_CSRC}",
        *(f"-I{path}" for path in _torch_include_paths()),
        # torch/extension.h pulls in pybind11, which needs Python.h.
        f"-I{_sysconfig_include()}",
    ]
    # `cpp_extension.load` picks the C++ standard for us; assembling our own
    # command line means we have to find it ourselves, so try newest first.
    compile_result = None
    for std in _STD_CANDIDATES:
        cmd = [
            "nvcc", "-c", str(source), "-o", str(obj),
            *cuda_flags(arch_str), *DEFINES.get(name, []), f"-std={std}",
            *includes,
            "-DTORCH_EXTENSION_NAME=" + name,
            "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(_cxx11_abi())),
            "--compiler-options", "-fPIC",
        ]
        if verbose:
            print(" ".join(cmd), file=sys.stderr)
        compile_result = subprocess.run(cmd, capture_output=True, text=True,
                                        check=False)
        if compile_result.returncode == 0:
            break
        # torch's headers reject a too-old standard with a specific #error;
        # anything else is a real compile failure and must not be retried.
        if "C++20 or later" not in compile_result.stderr:
            break
    if compile_result is None or compile_result.returncode != 0:
        raise RuntimeError(
            f"nvcc compile of {source.name} for sm_{arch_str} failed:\n"
            f"{compile_result.stderr}"
        )
    dump = subprocess.run(
        ["cuobjdump", "-res-usage", str(obj)],
        capture_output=True, text=True, check=False,
    )
    return (compile_result.stderr or "") + (dump.stdout or dump.stderr)


def _require_cuda_torch_headers() -> None:
    """Fail early, and legibly, on a CPU-only torch wheel.

    ``c10/cuda/impl/cuda_cmake_macros.h`` is generated by torch's *CUDA* build
    and is absent from a ``+cpu`` wheel, so every source here dies deep inside
    ``c10/cuda/CUDAStream.h`` with a bare "No such file or directory" that
    reads like a broken include path rather than the real cause.
    """
    for include_dir in _torch_include_paths():
        if (pathlib.Path(include_dir) / "c10" / "cuda" / "impl"
                / "cuda_cmake_macros.h").exists():
            return
    import torch  # noqa: PLC0415

    raise RuntimeError(
        f"this torch ({torch.__version__}) is a CPU-only build: it ships no "
        "c10/cuda/impl/cuda_cmake_macros.h, so nothing that includes "
        "<c10/cuda/CUDAStream.h> can be compiled against it. The compile gate "
        "needs a CUDA-enabled torch and an nvcc that knows the target arch; "
        "it still needs no GPU, no driver and no CUDA context."
    )


def _sysconfig_include() -> str:
    import sysconfig  # noqa: PLC0415

    return sysconfig.get_paths()["include"]


def _cxx11_abi() -> bool:
    try:
        import torch  # noqa: PLC0415

        return bool(torch._C._GLIBCXX_USE_CXX11_ABI)
    except Exception:
        return True


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    compile_only = "--compile-only" in argv
    verbose = "-v" in argv or "--verbose" in argv
    names = [ALIASES.get(arg, arg) for arg in argv if not arg.startswith("-")] \
        or list(EXTENSIONS)
    arch_str = arch()
    # "unset" is a sentinel that is neither "" nor a device list, so an absent
    # variable takes the same branch as a non-empty one and gets the note.
    if os.environ.get("CUDA_VISIBLE_DEVICES", "unset") != "":
        print(
            "note: CUDA_VISIBLE_DEVICES is not empty. Neither mode creates a "
            "CUDA context, but empty is the safer way to run this while "
            "another lane owns the GPUs.",
            file=sys.stderr,
        )
    print(f"arch: sm_{arch_str}   flags: "
          f"{' '.join(gencode_flags(arch_str))}")
    exit_code = 0
    for name in names:
        if name not in EXTENSIONS:
            print(f"FAIL: unknown extension {name!r}", file=sys.stderr)
            exit_code = 2
            continue
        try:
            if compile_only:
                info = compile_check(name, arch_str=arch_str, verbose=verbose)
                print(f"COMPILE OK: {name}")
            else:
                module = _load_extension(name, verbose=verbose)
                print(f"BUILD OK: {name} -> {module}")
                print("exports: "
                      f"{sorted(e for e in dir(module) if not e.startswith('_'))}")
                info = ptxas_info(name)
            print(f"--- cuobjdump -res-usage ({name}, sm_{arch_str}) ---")
            print(info)
        except Exception as exc:  # noqa: BLE001 - this is the gate's report
            print(f"FAIL: {name}: {exc}", file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
