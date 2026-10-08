"""Build/load only the Q8KV4 host API under two Python package namespaces.

This check never creates a CUDA tensor, invokes a decoder, or builds a device
kernel. CUDA lazy initialization is forbidden before either package is imported.
Run once with --build and then without it to exercise reversed cached imports.
The caller supplies an isolated TORCH_EXTENSIONS_DIR and a valid CUDA_HOME.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-site-packages", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    sys.path.insert(0, str(source / "python"))
    if args.extra_site_packages is not None:
        # Preserve the interpreter's CUDA-enabled Torch; supply only missing
        # dependencies from an existing environment, without mutating either.
        sys.path.append(str(args.extra_site_packages.resolve()))
    os.environ["MSA_REGISTER_TVM_FFI"] = "0"
    os.environ["ICP_RUNTIME_JIT"] = "allow" if args.build else "fail"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MAX_JOBS"] = "1"

    import torch
    import torch.utils.cpp_extension
    import tvm_ffi

    def forbidden(*_args, **_kwargs):
        raise AssertionError("CUDA initialization or an unexpected build was attempted")

    torch.cuda._lazy_init = forbidden
    assert not torch.cuda.is_initialized()
    if not args.build:
        torch.utils.cpp_extension.load = forbidden

    root = source / "python/fmha_sm100"
    source_paths = sorted(path for path in (root / "decode_q8kv4").rglob("*")
                          if path.is_file() and "__pycache__" not in path.parts)
    before = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source_paths}
    vendor_name = "validation_vendor_msa"
    names = ["fmha_sm100", vendor_name]
    if not args.build:
        names.reverse()
    modules = []
    records = []
    for name in names:
        if name == vendor_name:
            spec = importlib.util.spec_from_file_location(
                name, root / "__init__.py", submodule_search_locations=[str(root)])
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        interface = importlib.import_module(name + ".decode_q8kv4.interface")
        module = interface._get_cpp()
        artifact = interface.cpp_extension_path()
        assert Path(module.__file__).resolve() == artifact.resolve()
        assert artifact.is_file()
        assert (name + ".decode_q8kv4.native_v2.jit_get_plan").encode() in artifact.read_bytes()
        for callback in ("jit_get_plan", "jit_get_reduction", "jit_get_fmha_fwd_sparse_variant"):
            assert tvm_ffi.get_global_func(name + ".decode_q8kv4.native_v2." + callback)
        modules.append(module)
        records.append({"package": name, "extension_name": module.__name__,
                        "path": str(artifact),
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        "compile_flags": interface._cpp_compile_flags(),
                        "toolchain_identity": interface._cpp_toolchain_identity()})
    assert modules[0] is not modules[1]
    assert modules[0]._PlanHandle is not modules[1]._PlanHandle
    assert not torch.cuda.is_initialized()
    after = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in source_paths}
    assert before == after
    result = {"status": "passed", "mode": "build" if args.build else "reversed-cache-load",
              "torch": torch.__version__, "torch_cuda": torch.version.cuda,
              "torch_path": torch.__file__, "tvm_ffi_path": tvm_ffi.__file__,
              "cuda_initialized": False, "device_kernels_built_or_run": False,
              "source_hashes": before, "libraries": records}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("status", "mode", "cuda_initialized", "device_kernels_built_or_run")}))


if __name__ == "__main__":
    main()
