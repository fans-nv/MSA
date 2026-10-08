"""Copy selected, inventoried text evidence into a separate Git review branch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

packet = Path(__file__).resolve().parent
workspace = packet.parents[1]
destination = workspace / "review/gitlab-human-review-20261008"
assert not destination.exists(), "Refusing to replace an existing review checkout"
destination.mkdir()
manifest = json.loads((packet / "manifest.json").read_text())
validation = json.loads((packet / "final-validation.json").read_text())
assert validation["manifest_sha256"] == hashlib.sha256((packet / "manifest.json").read_bytes()).hexdigest()

docs = [
    "human-review-checklist.md", "vllm-human-review-mr.md", "msa-mr-draft.md",
    "model-design.md", "writer-design.md", "runtime-review.md",
    "runtime-independent-review.md", "public-nvfp4-review.md",
    "msa-public-compatibility.md", "target-validation-review.md",
]
sources = {packet / name for name in docs}
for pattern in ("*.json", "*.py", "*.sh", "*-files.txt"):
    sources.update(packet.glob(pattern))
text_suffixes = {".json", ".log", ".txt", ".xml", ".md", ".sha256", ".patch", ".snapshot"}
sources.update(path for path in (packet / "logs").rglob("*")
               if path.is_file() and path.suffix in text_suffixes)
sources.update(path for path in (packet / "target-validation").rglob("*")
               if path.is_file() and path.suffix in text_suffixes | {".py"})
for name in ("vllm-01-offload.patch", "vllm-02-runtime.patch", "vllm-03-writer.patch",
             "vllm-04-model-integration.patch", "msa-public-compat.patch"):
    sources.add(packet / "artifacts" / name)

links = {
    "../../worktrees/vllm-icp-public/":
        "https://gitlab-master.nvidia.com/fans/vllm/-/blob/516bd9a9d9962aff25582f774a49640e022bf348/",
    "../../worktrees/msa-dev-public/":
        "https://gitlab-master.nvidia.com/fans/MSA/-/blob/0f658079c9ea9ab63c76b975b7ee4dd54a198334/",
}
inventory = {}

def copy(source: Path, relative: str) -> None:
    assert not source.is_symlink(), source
    content = source.read_bytes()
    content.decode("utf-8")  # No compiled payload belongs in this notes branch.
    original = hashlib.sha256(content).hexdigest()
    if source.suffix == ".md":
        body = content.decode()
        for old, new in links.items():
            body = body.replace("](" + old, "](" + new)
        content = body.encode()
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    inventory[relative] = {
        "source": source.relative_to(packet).as_posix(),
        "source_sha256": original,
        "published_sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }

for source in sorted(sources):
    copy(source, source.relative_to(packet).as_posix())
copy(packet / "HUMAN-REVIEW.md", "README.md")
missing = []
for name in ["README.md", *docs]:
    for target in re.findall(r"\]\(([^)]+)\)", (destination / name).read_text()):
        if not target.startswith(("http:", "https:", "#")):
            if not (destination / target.split("#", 1)[0]).exists():
                missing.append((name, target))
assert not missing, missing
(destination / "REVIEW_FILES.json").write_text(json.dumps({
    "sources": manifest["sources"],
    "source_manifest_sha256": validation["manifest_sha256"],
    "scope": "Selected text evidence and source patches; no compiled/cache/distribution artifacts.",
    "files": inventory,
}, indent=2) + "\n")
print(json.dumps({"path": str(destination), "files": len(inventory),
                  "bytes": sum(record["bytes"] for record in inventory.values()),
                  "overview_links": "passed"}, indent=2))
