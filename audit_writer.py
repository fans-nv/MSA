"""Reproducible source and native-code comparisons for the public writer port."""

from pathlib import Path
import ast
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path(__file__).parent
TREE = ROOT / "worktrees/vllm-icp-public"
REF = ROOT / "worktrees/vllm-msa-upstream"
BASE = "242e4213fc9845ff6fe607af1aee626fd8acc990"
PATH = "csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu"


def normalized(text):
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S))


def body(text, name):
    start = text.index(name + "(")
    masked = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"', lambda m: " " * len(m[0]), text, flags=re.S)
    opening = masked.index("{", start)
    end, depth = opening + 1, 1
    while depth:
        depth += (masked[end] == "{") - (masked[end] == "}")
        end += 1
    return text[opening:end]


def original(path):
    return subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=TREE, text=True)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


public = original(PATH)
candidate = (TREE / PATH).read_text()
qualified = (REF / PATH).read_text()
public_helpers = ["warpReduceSum", "loadElems", "storeElems", "storeCacheElems",
                  "storeIndexQElemsFp8", "e2m1Code", "rcpApproxFtz", "divRnByRcp",
                  "nvfp4GroupScale", "nvfp4UnitGroupScale", "nvfp4Quotient", "clearNegZeroNibbles"]
qualified_helpers = ["normAndRope", "storeElemsFp8", "storeScaledQElemsFp8",
                     "nvfp4Reciprocal", "quantizeNvfp4Quad"]
checks = {}
for origin, source, names in [("public", public, public_helpers), ("qualified_icp", qualified, qualified_helpers)]:
    for name in names:
        left, right = normalized(body(source, name)), normalized(body(candidate, name))
        checks[f"{origin}:{name}"] = {"equal": left == right, "before": sha(left), "after": sha(right)}
store_left, store_right = body(public, "storeNvfp4CacheElems"), body(candidate, "storeNvfp4CacheElems")
store_left, store_right = [normalized(v[v.index("uint8_t* const slot ="):]) for v in (store_left, store_right)]
checks["public:per_head_slot_stores"] = {"equal": store_left == store_right, "before": sha(store_left), "after": sha(store_right)}
header = "csrc/libtorch_stable/fused_minimax_m3_icp.cuh"
left, right = normalized((REF / header).read_text()), normalized((TREE / header).read_text())
checks["qualified_icp:complete_metadata_header"] = {"equal": left == right, "before": sha(left), "after": sha(right)}
public_test = "tests/kernels/test_fused_minimax_m3_qknorm_rope_kv_insert.py"
checks["public:independent_cuda_oracles"] = {"equal": original(public_test) == (TREE / public_test).read_text()}


def schema(text):
    part = text.split('"fused_minimax_m3_qknorm_rope_kv_insert("', 1)[1].split(");", 1)[0]
    return "fused_minimax_m3_qknorm_rope_kv_insert(" + "".join(ast.literal_eval(p) for p in re.findall(r'"[^"\n]*"', part))


binding = "csrc/libtorch_stable/torch_bindings.cpp"
old_schema, new_schema = schema(original(binding)), schema((TREE / binding).read_text())
checks["public:verbatim_schema_prefix"] = {"equal": new_schema.startswith(old_schema.removesuffix(") -> ()") + ", ")}
result = {"scope": "Exact normalized function/header/store blocks, schema prefix and untouched independent public oracles. Not a whole-source equivalence or GPU numerical/performance proof.",
          "normalization": "Remove whitespace and comments; no numerical token substitutions.",
          "sources": {"public": BASE, "qualified_icp": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REF, text=True).strip(), "candidate_sha256": sha(candidate)},
          "checks": checks, "all_equal": all(x["equal"] for x in checks.values())}
(AUDIT / "writer-source-parity.json").write_text(json.dumps(result, indent=2) + "\n")
print("source comparisons", len(checks), "all_equal", result["all_equal"])
assert result["all_equal"]

proof = AUDIT / "logs/writer-codegen-r2"
proof.mkdir(exist_ok=False)
results = {}
tool = "/tmp/cu134/nvidia/cu13/bin/cuobjdump"
for arch in ("107a", "107f"):
    functions = {}
    for label, folder in [("baseline", "writer-baseline-native-r3"), ("candidate", "writer-native-r2")]:
        obj = AUDIT / "logs" / folder / f"writer-{arch}.o"
        sass = subprocess.check_output([tool, "--dump-sass", str(obj)], text=True)
        (proof / f"{label}-{arch}.sass").write_text(sass)
        resources = subprocess.check_output([tool, "--dump-resource-usage", str(obj)], text=True)
        (proof / f"{label}-{arch}.resources").write_text(resources)
        parts = re.split(r"Function : (\S+)\n", sass)
        functions[label] = {parts[i]: re.findall(r"/\*[0-9a-f]+\*/\s+(.+?;)", parts[i + 1]) for i in range(1, len(parts), 2) if "fusedMiniMaxM3QNormRopeKVInsertKernel" in parts[i]}
    old, new = functions["baseline"], functions["candidate"]
    common = set(old) & set(new)
    differing = [name for name in common if old[name] != new[name]]
    results[arch] = {"baseline_ordinary_kernels": len(old), "candidate_ordinary_kernels": len(new), "common": len(common), "identical_instruction_text": len(common) - len(differing), "different_symbols": differing}
    assert old and all(old.values()) and old == new
    results[arch]["ordinary_instruction_count_range"] = [min(map(len, old.values())), max(map(len, old.values()))]
(proof / "comparison.json").write_text(json.dumps({"scope": "SASS opcode/operand/ordering equality for ordinary instantiations with identical release-like flags. Instruction addresses/encoded hex omitted; register names remain significant. No GPU execution.", "results": results}, indent=2) + "\n")
print("ordinary SASS comparisons:", {arch: value["identical_instruction_text"] for arch, value in results.items()})
