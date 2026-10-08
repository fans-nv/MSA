"""Copy a pinned CUTLASS header tree into a source distribution without symlinks."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

HEADER_DIRS = ("include", "tools/util/include")
SENTINELS = ("include/cutlass/cutlass.h",
             "tools/util/include/cutlass/util/host_tensor.h")


def stage(source: Path, destination: Path, *, source_id: str,
          license_file: Path | None = None) -> dict:
    """Stage exact bytes, refusing incomplete inputs or a different existing tree."""
    source = source.resolve()
    if not source_id.strip():
        raise ValueError("source_id must identify the selected commit or parent image")
    for name in SENTINELS:
        if not (source / name).is_file():
            raise ValueError(f"CUTLASS source is missing {name}: {source}")
    notices = sorted({*source.glob("LICENSE*"), *source.glob("NOTICE*")})
    if license_file is not None and not license_file.is_file():
        raise ValueError(f"license file does not exist: {license_file}")
    if (not any(p.name.startswith("LICENSE") and p.is_file() for p in notices)
            and license_file is None):
        raise ValueError(f"CUTLASS source has no LICENSE file: {source}")
    files = sorted({*(p for name in HEADER_DIRS
                      for p in (source / name).rglob("*") if p.is_file()),
                    *(p for p in notices if p.is_file())})
    symlinks = [p for name in HEADER_DIRS
                for p in (source / name).rglob("*") if p.is_symlink()]
    if symlinks or any(p.is_symlink() for p in notices):
        raise ValueError("CUTLASS inputs must be materialized, without symlinks")
    inputs = {p.relative_to(source).as_posix(): p for p in files}
    if license_file is not None:
        inputs["LICENSE.provided.txt"] = license_file
    hashes = {rel: hashlib.sha256(p.read_bytes()).hexdigest()
              for rel, p in sorted(inputs.items())}
    manifest = {"schema": "fmha_sm100.cutlass-sources.v1", "source_id": source_id,
                "files": hashes}
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if destination.exists() and any(destination.iterdir()):
        existing = {p.relative_to(destination).as_posix():
                    hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in destination.rglob("*")
                    if p.is_file() and p.name != "SOURCE.json"}
        stamp = destination / "SOURCE.json"
        if existing != hashes or not stamp.is_file() or stamp.read_text() != encoded:
            raise ValueError(
                f"refusing to replace a different CUTLASS tree: {destination}")
        return manifest
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".cutlass-stage-", dir=destination.parent))
    try:
        for rel in hashes:
            target = temporary / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(inputs[rel], target)
        (temporary / "SOURCE.json").write_text(encoded)
        if destination.exists():
            # A fresh, uninitialized git submodule leaves an empty directory.
            destination.rmdir()
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--source-id", required=True,
                        help="immutable parent image SHA or selected CUTLASS commit")
    parser.add_argument("--license-file", type=Path,
                        help="existing license text when the archive omits LICENSE")
    parser.add_argument("--destination", type=Path, default=(
        Path(__file__).resolve().parents[1] / "python/fmha_sm100/cutlass"))
    args = parser.parse_args()
    manifest = stage(args.source, args.destination, source_id=args.source_id,
                     license_file=args.license_file)
    print(f"Staged {len(manifest['files'])} files in {args.destination}")


if __name__ == "__main__":
    main()
