"""Bind the required formatter-only change to the existing native evidence."""

from pathlib import Path
import hashlib
import json
import os
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path(__file__).parent
TREE = ROOT / "worktrees/vllm-icp-public"
PROOF = AUDIT / "logs/writer-format21-r1"
PATHS = [
    "csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu",
    "csrc/libtorch_stable/fused_minimax_m3_icp.cuh",
    "csrc/libtorch_stable/ops.h",
    "csrc/libtorch_stable/torch_bindings.cpp",
]
OWNED = PATHS + [
    "vllm/_custom_ops.py",
    "tests/kernels/test_fused_minimax_m3_icp_writer_api.py",
    "tests/kernels/test_fused_minimax_m3_icp_writer.py",
    "tests/kernels/test_minimax_m3_icp_device_plan.py",
]
ENVIRONMENT = {
    "UV_CACHE_DIR": "/tmp/icp-public-format-cache",
    "UV_TOOL_DIR": "/tmp/icp-public-format-tools",
    "UV_TOOL_BIN_DIR": "/tmp/icp-public-format-bin",
}
FORMAT = ["uv", "tool", "run", "--offline", "--from", "clang-format==21.1.2", "clang-format"]
COMMANDS = {
    "version": FORMAT + ["--version"],
    "applied": FORMAT + ["--style=file", "-i", *PATHS],
    "check": FORMAT + ["--style=file", "--dry-run", "--Werror", *PATHS],
}
(PROOF / "commands.json").write_text(json.dumps({"cwd": str(TREE), "environment": ENVIRONMENT, "commands": COMMANDS}, indent=2) + "\n")

# This conservative lexer retains every comment and string/character literal
# verbatim. C++ line splicing is applied first; multi-character operators are
# distinct tokens. Logical preprocessor directive lines are compared separately.
LEXEMES = re.compile(
    r'//[^\n]*|/\*.*?\*/|(?:u8|u|U|L)?R"(?P<delimiter>[^ ()\\\t\r\n]*)\(.*?\)(?P=delimiter)"'
    r'|(?:u8|u|U|L)?"(?:\\.|[^"\\])*"|(?:u8|u|U|L)?\'(?:\\.|[^\'\\])*\''
    r'|[A-Za-z_$][A-Za-z0-9_$]*|[0-9]+(?:\.[0-9A-Za-z_]*)?'
    r'|>>=|<<=|<=>|->\*|\.\.\.|##|::|->|\.\*|\+\+|--|&&|\|\||<<|>>|<=|>=|==|!='
    r'|\+=|-=|\*=|/=|%=|&=|\|=|\^=|\S',
    re.S,
)


def lexical_view(text):
    logical = re.sub(r"\\\r?\n", "", text)
    tokens = [m[0] for m in LEXEMES.finditer(logical)]
    directives = [[m[0] for m in LEXEMES.finditer(line)] for line in logical.splitlines() if line.lstrip().startswith("#")]
    return tokens, directives


def sha(data):
    return hashlib.sha256(data).hexdigest()


native = json.loads((AUDIT / "logs/writer-native-r2/result.json").read_text())
cpu = json.loads((AUDIT / "logs/writer-cpu-r3/result.json").read_text())
comparisons = {}
for path in PATHS:
    before = (PROOF / (Path(path).name + ".before")).read_bytes()
    after = (TREE / path).read_bytes()
    before_tokens, before_directives = lexical_view(before.decode())
    after_tokens, after_directives = lexical_view(after.decode())
    comparisons[path] = {
        "before_sha256": sha(before), "after_sha256": sha(after),
        "before_matches_native_proof": sha(before) == native["sources_after"][path],
        "before_matches_cpu_proof": sha(before) == cpu["sources_after"][path],
        "tokens_equal": before_tokens == after_tokens,
        "preprocessor_directives_equal": before_directives == after_directives,
        "token_count": len(after_tokens),
    }
    assert all(comparisons[path][key] for key in ("before_matches_native_proof", "before_matches_cpu_proof", "tokens_equal", "preprocessor_directives_equal")), path

results = {}
for name in ("version", "check"):
    with (PROOF / f"{name}.log").open("w") as log:
        result = subprocess.run(COMMANDS[name], cwd=TREE, env=os.environ | ENVIRONMENT, stdout=log, stderr=subprocess.STDOUT)
    results[name] = result.returncode
    assert result.returncode == 0, name
assert (PROOF / "version.log").read_text().strip() == "clang-format version 21.1.2"

final_hashes = {path: sha((TREE / path).read_bytes()) for path in OWNED}
for path in OWNED[4:]:
    assert final_hashes[path] == cpu["sources_after"][path], path
result = {
    "scope": "Formatting only. Strings, characters, comments, operator/identifier tokens and logical preprocessor directives are equal; unchanged Python/tests match the final CPU proof. No native rebuild or GPU execution needed for these whitespace changes.",
    "formatter_results": results, "comparisons": comparisons,
    "final_source_sha256": final_hashes,
    "reused_native_proof": "logs/writer-native-r2/result.json",
    "reused_ordinary_instruction_proof": "logs/writer-codegen-r2/comparison.json",
    "reused_binding_proof": "logs/writer-binding-syntax-r1/result.json",
    "reused_boxing_proof": "logs/writer-boxing-r1/boxing-result.json",
    "reused_cpu_proof": "logs/writer-cpu-r3/result.json",
    "format_configuration_sha256": {
        path: sha((TREE / path).read_bytes())
        for path in (".clang-format", ".pre-commit-config.yaml")
    },
}
(PROOF / "result.json").write_text(json.dumps(result, indent=2) + "\n")
(AUDIT / "writer-final-source-sha256.json").write_text(json.dumps(final_hashes, indent=2) + "\n")
print("clang-format21.1.2 check passed; four C++ files lexically identical; eight final hashes recorded")
