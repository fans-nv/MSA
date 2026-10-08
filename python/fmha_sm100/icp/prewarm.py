"""Image-build prewarm of every ICP kernel, and the serve-time check of it.

    python -m fmha_sm100.icp.prewarm build  --arch 107a [--commit SHA] [--manifest-out F]
    python -m fmha_sm100.icp.prewarm verify [--arch 107a] [--policy warn|fail]

``build`` populates the serving ICP profiles in the keyed cache. The native
and decode compilers need no CUDA context; NVFP4 attention objects additionally
require a matching export or a CUDA-device production call as described below:

    icp     every extension in ``_build.EXTENSIONS`` except test-only ones
    fmha    every OnlyScoreIcp scorer a device plan can select (W=2 page 64,
            W=4 page 32; unsplit; both single_wg), plus the plan,
            sparse_topk modules
    decode  the CuTe decode scan for every profile in ``AOT_PROFILES``, exported
            as a loadable ``.so`` (the in-process DSL cache does not persist)
    nvfp4   the NVFP4 sparse prefill: the k2q CSR extension, and the CuTe AOT
            objects of one production-form call (needs a CUDA device, or
            ``ICP_NVFP4_AOT_IMPORT`` = the aot dir of such a run for the same
            keyed sources; see ``attention/nvfp4_prefill/_prewarm.py``)

and writes ``<root>/sm_<arch>/PREWARM-MANIFEST.json``: the MSA commit,
per-component source digests and per-file hashes, the toolchain, and every
artifact's path and sha256. Its presence also turns the default runtime-JIT
policy to ``warn`` (:mod:`fmha_sm100.icp._jit_guard`).

``verify`` is the startup assertion: exit 0 only if every required component
and its complete artifact inventory is present, the live source/toolchain
identities match, and every artifact has its recorded sha256. Verification
requires all components by default; ``--components`` selects a developer
subset explicitly. ``--policy warn`` reports and exits 0.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys

from . import _cache

SCHEMA = "fmha_sm100.icp.prewarm.v1"
#: Extensions that only tests load; they are never on a serving path.
TEST_ONLY_EXTENSIONS = frozenset({"icp_select_control"})
#: OnlyScoreIcp page sizes a device plan can select: R = 128 / W for W = 2, 4.
FMHA_PAGE_SIZES = (64, 32)


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _artifact(component: str, name: str, path) -> dict:
    path = pathlib.Path(path)
    if not path.is_file():
        raise RuntimeError(
            f"{component}:{name}: expected artifact {path} was not built")
    return {"component": component, "name": name, "path": str(path),
            "sha256": _sha256(path), "bytes": path.stat().st_size}


def _version(dist: str) -> str | None:
    try:
        from importlib.metadata import version  # noqa: PLC0415

        return version(dist)
    except Exception:  # noqa: BLE001
        return None


def toolchain() -> dict:
    """Everything outside the sources that changes the emitted code."""
    nvcc = None
    try:
        out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True,
                             check=False).stdout
        nvcc = next((ln for ln in out.splitlines() if "release" in ln), out.strip())
    except OSError:
        pass
    facts = {"python": sys.version.split()[0], "nvcc": nvcc}
    for dist in ("torch", "nvidia-cutlass-dsl", "apache-tvm-ffi", "quack-kernels"):
        facts[dist] = _version(dist)
    try:
        import torch  # noqa: PLC0415

        facts["torch.version.cuda"] = torch.version.cuda
    except ImportError:
        facts["torch.version.cuda"] = None
    return facts


def environment() -> dict:
    """Cache routing and build flags that runtime loaders must reproduce."""
    keys = ("TORCH_EXTENSIONS_DIR", "ICP_CACHE_ROOT", "CUDA_HOME", "CC", "CXX",
            "ICP_KERNEL_LINEINFO", "GPU_TRACE", "SM_TIMING", "FMHA_GMEM_CHECK",
            "MM_SPARSE_ATTN_AOT_CACHE", "MM_SPARSE_ATTN_AOT_DISABLE",
            "MINIMAX_KVFP4_FP8_PAIR_DEQUANT")
    if os.environ.get("TORCH_EXTENSIONS_DIR") is None:
        # torch's default build directory depends on these when no path is set.
        keys += ("HOME", "XDG_CACHE_HOME")
    return {key: os.environ.get(key) for key in keys}


def _commit(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    if os.environ.get("MSA_COMMIT"):
        return os.environ["MSA_COMMIT"]
    stamp = _cache.PACKAGE / "COMMIT"
    if stamp.is_file():
        return stamp.read_text().strip()
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=_cache.PACKAGE,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def _build_icp(arch: str) -> list[dict]:
    from . import _build  # noqa: PLC0415

    out = []
    for name in _build.EXTENSIONS:
        if name in TEST_ONLY_EXTENSIONS:
            continue
        _build._load_extension(name)
        out.append(_artifact("icp", name,
                             pathlib.Path(_build.build_dir(name, arch)) / f"{name}.so"))
    return out


def _build_fmha() -> list[dict]:
    from .scorer.prefill import api, jit  # noqa: PLC0415

    variants = api.icp_scorer_variants(page_sizes=FMHA_PAGE_SIZES)
    out = [_artifact("fmha", name,
                     jit._variant_manager.compile_locked(
                         name, jit._variant_key_from_runtime(*args)[1]))
           for name, args in variants]
    for name in ("plan", "sparse_topk"):
        out.append(_artifact("fmha", name, jit.build_module(name)))
    entries = [_artifact("fmha", item["name"] + ".entry.json",
                         pathlib.Path(item["path"]).parent / "entry.json")
               for item in out]
    out.extend(entries)
    out.append(_artifact("fmha", "namespace.json",
                         jit._namespace().path / "manifest.json"))
    return out


def _build_decode(arch: str) -> list[dict]:
    from .scorer.decode import icp_decode_score as d  # noqa: PLC0415

    out = []
    for profile in d.AOT_PROFILES:
        path = d.export_aot_scan(*profile, arch)
        out.append(_artifact("decode", d.aot_function_name(*profile), path))
    return out


def _build_nvfp4(arch: str) -> list[dict]:
    from .attention.nvfp4_prefill import _prewarm  # noqa: PLC0415

    return [_artifact("nvfp4", name, path) for name, path in _prewarm.build(arch)]


BUILDERS = {"icp": _build_icp, "fmha": _build_fmha,
            "decode": _build_decode, "nvfp4": _build_nvfp4}


def build(arch: str, *, components=_cache.COMPONENTS, commit: str | None = None,
          manifest_out: str | None = None) -> dict:
    arch = _cache.normalize_arch(arch)
    legacy = {k: os.environ[k] for k in _cache.LEGACY_ENV.values() if os.environ.get(k)}
    if legacy:
        raise SystemExit(f"refusing to prewarm into unkeyed legacy cache dirs "
                         f"{legacy}; unset them and use ICP_CACHE_ROOT")
    components = _validated_components(components)
    # Compiling is the point here; the arch is fixed for every compiler we drive.
    os.environ["ICP_RUNTIME_JIT"] = "allow"
    os.environ["ICP_KERNEL_ARCH"] = arch
    if "cutlass" in sys.modules and os.environ.get("CUTE_DSL_ARCH") != f"sm_{arch}":
        raise SystemExit("CUTLASS DSL was imported before CUTE_DSL_ARCH was set")
    os.environ["CUTE_DSL_ARCH"] = f"sm_{arch}"

    artifacts = []
    inventories = {}
    for component in components:
        print(f"[fmha_sm100.icp.prewarm] {component} for sm_{arch} ...", flush=True)
        if component in ("icp", "decode", "nvfp4"):
            built = BUILDERS[component](arch)
        else:
            built = BUILDERS[component]()
        names = [item["name"] for item in built]
        if (not names or len(set(names)) != len(names)
                or any(item["component"] != component for item in built)):
            raise RuntimeError(f"{component}: builder returned an invalid inventory")
        artifacts += built
        inventories[component] = sorted(names)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds"),
        "arch": arch,
        "msa_commit": _commit(commit),
        "msa_version": _version("fmha_sm100"),
        "package_dir": str(_cache.PACKAGE),
        "cache_root": str(_cache.root()),
        "components": {c: {"digest": _cache.component_digest(c),
                           "dir": str(_cache.component_dir(c, arch)),
                           "files": _cache.file_hashes(c),
                           "artifacts": inventories[c]}
                       for c in components},
        "toolchain": toolchain(),
        "environment": environment(),
        "artifacts": artifacts,
    }
    path = _cache.manifest_path(arch)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    path.write_text(text)
    if manifest_out:
        pathlib.Path(manifest_out).write_text(text)
    print(f"[fmha_sm100.icp.prewarm] {len(artifacts)} artifacts; manifest {path}")
    return manifest


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------


def _validated_components(components) -> tuple[str, ...]:
    components = tuple(dict.fromkeys(components))
    unknown = set(components) - set(_cache.COMPONENTS)
    if unknown or not components:
        raise ValueError(f"select known prewarm components, got {components!r}")
    return components


def _required_artifact_names(component: str) -> set[str] | None:
    """Enumerate fixed profiles without importing/compiling native extensions."""
    if component == "icp":
        from . import _build  # noqa: PLC0415

        return set(_build.EXTENSIONS) - TEST_ONLY_EXTENSIONS
    if component == "fmha":
        from .scorer.prefill.api import icp_scorer_variants  # noqa: PLC0415

        variants = icp_scorer_variants(page_sizes=FMHA_PAGE_SIZES)
        names = {name for name, _ in variants} | {"plan", "sparse_topk"}
        return names | {name + ".entry.json" for name in names} | {"namespace.json"}
    if component == "decode":
        from .scorer.decode import icp_decode_score as d  # noqa: PLC0415

        return {d.aot_function_name(*profile) for profile in d.AOT_PROFILES}
    # Writer names contain a torch/flag fingerprint and NVFP4 AOT names depend
    # on runtime compile arguments. Their complete build inventory is recorded.
    return None


def _native_extension_path(item: dict, arch: str) -> pathlib.Path | None:
    """Resolve host-side cache paths without compiling or loading native code."""
    if item["component"] == "fmha":
        from .scorer.prefill import jit  # noqa: PLC0415

        name = item["name"]
        if name == "namespace.json":
            return jit._namespace().path / "manifest.json"
        library_name = name.removesuffix(".entry.json")
        if library_name in ("plan", "sparse_topk"):
            recipe = jit._module_recipe(library_name)[0]
        else:
            params = jit._variant_params_from_name(library_name)
            recipe = jit._variant_manager._recipe(library_name, params)[0]
        if name.endswith(".entry.json"):
            return recipe.dir / "entry.json"
        path = recipe.lookup()
        if path is None:
            raise RuntimeError(f"fmha:{name}: no current dev JIT source record; it will JIT")
        return path
    if item["component"] == "nvfp4" and item["name"] == "k2q_ext":
        from .attention.nvfp4_prefill import k2q_extension_path  # noqa: PLC0415

        return k2q_extension_path(arch)

    if item["component"] == "nvfp4":
        from .attention.nvfp4_prefill import aot_dir  # noqa: PLC0415

        name = item["name"]
        filename = ("manifest.json" if name == "aot-manifest.json"
                    else name if name.endswith(".json") else name + ".o")
        return aot_dir(arch) / filename

    return None


def check(arch: str | None = None, *, check_toolchain: bool = True,
          components=_cache.COMPONENTS) -> list[str]:
    """Problems that would make a serving process JIT-compile; empty if none."""
    required = _validated_components(components)
    arch = _cache.normalize_arch(arch or _cache.key_arch())
    path = _cache.manifest_path(arch)
    if not path.is_file():
        return [f"no prewarm manifest at {path}: every kernel would JIT at runtime"]
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA:
        return [f"prewarm schema {manifest.get('schema')!r} != {SCHEMA}; rebuild "
                "the cache with complete component/artifact inventories"]
    problems = []
    if manifest.get("arch") != arch:
        problems.append(f"manifest arch {manifest.get('arch')} != sm_{arch}")
    if pathlib.Path(manifest["cache_root"]).resolve() != _cache.root().resolve():
        problems.append("effective cache root differs from the prewarm manifest")
    if "decode" in required and os.environ.get("ICP_DECODE_AOT", "1") == "0":
        problems.append("ICP_DECODE_AOT=0 disables prewarmed decode artifacts")
    if "nvfp4" in required and os.environ.get("MM_SPARSE_ATTN_AOT_DISABLE", "0") == "1":
        problems.append("MM_SPARSE_ATTN_AOT_DISABLE=1 disables prewarmed NVFP4 artifacts")
    elif "nvfp4" in required:
        from ..cute.src.common import aot_cache  # noqa: PLC0415

        if aot_cache._AOT_DISABLE:
            problems.append("NVFP4 AOT loader was imported with caching disabled")
    for name in _cache.LEGACY_ENV.values():
        if os.environ.get(name):
            problems.append(f"unkeyed cache override {name} is set; unset it")
    saved_environment = manifest.get("environment", {})
    live_environment = environment()
    for key in sorted(saved_environment.keys() | live_environment.keys()):
        if (key not in saved_environment
                or saved_environment.get(key) != live_environment.get(key)):
            problems.append(f"environment {key}: prewarmed with "
                            f"{saved_environment.get(key)!r}, now "
                            f"{live_environment.get(key)!r}; cache paths/flags differ")
    records = manifest["components"]
    missing = set(required) - records.keys()
    if missing:
        problems.append(f"missing required prewarm components: {sorted(missing)}")
    artifacts = manifest["artifacts"]
    for component in required:
        if component not in records:
            continue
        record = records[component]
        live_dir = _cache.component_dir(component, arch).resolve()
        if pathlib.Path(record["dir"]).resolve() != live_dir:
            problems.append(f"{component}: runtime cache directory {live_dir} differs "
                            f"from prewarm directory {record['dir']}")
        declared = record.get("artifacts", [])
        present = [item["name"] for item in artifacts
                   if item["component"] == component]
        if not declared or len(set(declared)) != len(declared):
            problems.append(f"{component}: missing/invalid artifact inventory")
        try:
            required_names = _required_artifact_names(component)
        except ImportError as exc:
            problems.append(f"{component}: cannot enumerate required artifacts: {exc}")
            required_names = None
        if required_names is not None and set(declared) != required_names:
            problems.append(f"{component}: inventory does not cover required profiles "
                            f"{sorted(required_names)}")
        if component == "nvfp4":
            from .attention.nvfp4_prefill import AOT_FAMILIES, aot_dir  # noqa: PLC0415
            from .attention.nvfp4_prefill._prewarm import (  # noqa: PLC0415
                _aot_objects, required_aot_entries,
            )

            missing_families = [family for family in AOT_FAMILIES
                               if not any(name.startswith(family + "_")
                                          and not name.endswith(".json")
                                          for name in declared)]
            if "k2q_ext" not in declared or missing_families:
                problems.append("nvfp4: inventory must contain k2q_ext and AOT "
                                f"families {AOT_FAMILIES}")
            objects = {name for name in declared
                       if name != "k2q_ext" and not name.endswith(".json")}
            if ("aot-manifest.json" not in declared
                    or any(name + ".json" not in declared for name in objects)):
                problems.append("nvfp4: inventory must include AOT source and toolchain metadata")
            expected_objects = set(required_aot_entries(arch))
            expected_entries = expected_objects | {name + ".json" for name in expected_objects}
            if not expected_entries.issubset(declared):
                problems.append("nvfp4: inventory omits required production AOT profiles: "
                                f"{sorted(expected_entries - set(declared))}")
            try:
                _aot_objects(aot_dir(arch), arch)
            except (OSError, ValueError, SystemExit) as exc:
                problems.append(f"nvfp4: production AOT admission failed: {exc}")
        if set(declared) != set(present) or len(set(present)) != len(present):
            problems.append(
                f"{component}: artifact inventory differs "
                f"(missing {sorted(set(declared) - set(present))}, "
                f"unexpected {sorted(set(present) - set(declared))}, "
                f"duplicate records {len(present) != len(set(present))})")
        if _cache.component_digest(component) != record["digest"]:
            live_files = _cache.file_hashes(component)
            changed = sorted(rel for rel in {*record["files"], *live_files}
                             if record["files"].get(rel) != live_files.get(rel))
            problems.append(f"{component}: sources differ from the prewarmed ones "
                            f"({', '.join(changed) or 'toolchain key'}); it will JIT")
    for item in artifacts:
        p = pathlib.Path(item["path"])
        try:
            expected = _native_extension_path(item, arch)
        except (OSError, ValueError, RuntimeError) as exc:
            problems.append(str(exc))
            expected = None
        if expected is not None and p.resolve() != expected.resolve():
            problems.append(f"{item['component']}:{item['name']}: runtime artifact "
                            f"path {expected} differs from manifest path {p}")
        if not p.is_file():
            problems.append(f"{item['component']}:{item['name']}: missing {p}")
        elif _sha256(p) != item["sha256"]:
            problems.append(f"{item['component']}:{item['name']}: {p} sha256 differs "
                            "from the manifest")
    if check_toolchain:
        live = toolchain()
        for key, want in manifest.get("toolchain", {}).items():
            if key == "python" and str(live.get(key)).split(".")[:2] == str(
                    want).split(".")[:2]:
                continue
            if live.get(key) != want:
                problems.append(f"toolchain {key}: prewarmed with {want!r}, "
                                f"now {live.get(key)!r}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--arch", required=True)
    b.add_argument("--components", default=",".join(_cache.COMPONENTS))
    b.add_argument("--commit")
    b.add_argument("--manifest-out")
    v = sub.add_parser("verify")
    v.add_argument("--arch")
    v.add_argument("--components", default=",".join(_cache.COMPONENTS))
    v.add_argument("--policy", choices=("warn", "fail"),
                   default=os.environ.get("ICP_PREWARM_VERIFY_POLICY", "fail"))
    v.add_argument("--no-toolchain", action="store_true")
    args = parser.parse_args(argv)
    try:
        components = _validated_components(c for c in args.components.split(",") if c)
    except ValueError as exc:
        parser.error(str(exc))
    if args.command == "build":
        build(args.arch, components=components, commit=args.commit,
              manifest_out=args.manifest_out)
        return 0
    problems = check(args.arch, check_toolchain=not args.no_toolchain,
                     components=components)
    arch = _cache.normalize_arch(args.arch or _cache.key_arch())
    if not problems:
        print(f"ICP_PREWARM_VERIFIED sm_{arch} {_cache.manifest_path(arch)}")
        return 0
    for p in problems:
        print(f"ICP_PREWARM_PROBLEM {p}", file=sys.stderr)
    return 1 if args.policy == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
