"""Source preservation proof; deliberately makes no GPU-equivalence claim."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
tree = root / "worktrees/vllm-icp-public"
reference = root / "worktrees/vllm-msa-upstream"
base = "242e4213fc9845ff6fe607af1aee626fd8acc990"
report = {"public_base": base, "reference": "53e229bc28df277f0906b94a082293c60a2bc2e7",
          "scope": "AST/source preservation only; no GPU numerical/performance claim."}

def digest(data):
    return hashlib.sha256(data).hexdigest()

def upstream(path):
    return subprocess.check_output(["git", "show", f"{base}:{path}"], cwd=tree)

class DropDocstrings(ast.NodeTransformer):
    def generic_visit(self, node):
        self.generic_visit_children(node)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
        return node

    def generic_visit_children(self, node):
        return super().generic_visit(node)

def normalized(data):
    return ast.dump(DropDocstrings().visit(ast.parse(data)), include_attributes=False)

unchanged = [
    "vllm/models/minimax_m3/common/indexer.py",
    "vllm/models/minimax_m3/common/sparse_attention.py",
    "vllm/models/minimax_m3/nvidia/indexer_msa.py",
    "vllm/models/minimax_m3/nvidia/sparse_attention_msa.py",
    "vllm/models/minimax_m3/nvidia/msa_cutlass_sparse_decode.py",
]
report["ordinary_files_identical_to_public"] = []
for path in unchanged:
    actual = (tree / path).read_bytes()
    assert actual == upstream(path), path
    report["ordinary_files_identical_to_public"].append({"path": path, "sha256": digest(actual)})

report["indexer_and_exchange_ast_preserved"] = []
for path in (
    "vllm/models/minimax_m3/common/indexer_icp.py",
    "vllm/models/minimax_m3/nvidia/indexer_icp.py",
    "vllm/models/minimax_m3/nvidia/ops/icp_dispatch.py",
):
    actual, previous = (tree / path).read_bytes(), (reference / path).read_bytes()
    assert normalized(actual) == normalized(previous), path
    report["indexer_and_exchange_ast_preserved"].append({
        "path": path, "reference_sha256": digest(previous), "candidate_sha256": digest(actual),
        "normalized_ast_sha256": digest(normalized(actual).encode()),
    })

model = "vllm/models/minimax_m3/nvidia/model.py"
def definitions(data):
    return {n.name: n for n in ast.parse(data).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
before, after = definitions(upstream(model)), definitions((tree / model).read_bytes())
retained = []
for name, node in before.items():
    if name in ("MiniMaxM3DecoderLayer", "MiniMaxM3Model"):
        continue
    assert ast.dump(node, include_attributes=False) == ast.dump(after[name], include_attributes=False), name
    retained.append(name)
report["model_unchanged_definitions"] = retained
report["model_intended_changes"] = ["MiniMaxM3DecoderLayer.__init__: opt-in sparse layer selection",
    "MiniMaxM3Model.__init__: opt-in exchange binding", "MiniMaxM3Model.shutdown_model_resources: exchange lifetime"]
report["candidate_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tree, text=True).strip()
output = Path(__file__).with_name("model-parity.json")
output.write_text(json.dumps(report, indent=2) + "\n")
print(f"PASS: {len(unchanged)} unchanged ordinary files, three preserved ICP ASTs, {len(retained)} unchanged model definitions.")
print(output)
