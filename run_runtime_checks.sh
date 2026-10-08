#!/usr/bin/env bash
set -uo pipefail
audit_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
workspace_root=$(cd "$audit_dir/../.." && pwd)
tag=${1:?usage: run_runtime_checks.sh TAG [pytest arguments...]}
shift
[[ "$tag" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || exit 2
tree="$workspace_root/worktrees/vllm-icp-public"
interpreter="$workspace_root/review/v13-formal/.venv/bin/python"
export PATH="$(dirname "$interpreter"):$PATH"
export PYTHONPATH="$tree:$workspace_root/worktrees/msa-dev-upstream/python"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export CUDA_VISIBLE_DEVICES= VLLM_TARGET_DEVICE=cpu
cache_root="$audit_dir/.cache/$tag/runtime"
export XDG_CACHE_HOME="$cache_root"
export UV_CACHE_DIR="$cache_root/uv" PRE_COMMIT_HOME="$cache_root/pre-commit"
export MYPY_CACHE_DIR="$cache_root/mypy" VLLM_CACHE_ROOT="$cache_root/vllm"
export VLLM_CONFIG_ROOT="$cache_root/vllm-config" HF_HOME="$cache_root/huggingface"
export TORCH_EXTENSIONS_DIR="$cache_root/torch-extensions"
log_dir="$audit_dir/logs/$tag/runtime"
if [[ -e "$log_dir" ]]; then
  echo "Refusing to replace existing evidence: $log_dir" >&2
  exit 2
fi
mkdir -p "$log_dir"
command=("$interpreter" -m pytest -q -rs -p no:cacheprovider --noconftest
  tests/v1/test_kv_cache_spec_registry.py
  tests/v1/simple_kv_offload/test_shutdown.py
  tests/v1/cudagraph/test_cudagraph_manager.py
  tests/v1/worker/test_gpu_model_runner_v2_cudagraph_profiling.py
  tests/v1/worker/test_attn_utils.py
  tests/v1/worker/test_gpu_batch_ordering.py
  tests/v1/worker/test_gpu_ubatch_slicing.py
  --junitxml="$log_dir/pytest.xml" "$@")
cd "$tree" || exit
git rev-parse HEAD > "$log_dir/head.txt"
git diff --binary HEAD -- vllm/v1 tests/v1 \
  vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py \
  > "$log_dir/runtime-before.patch"
printf '%s\n' "$PYTHONPATH" > "$log_dir/pythonpath.txt"
printf '%q ' "${command[@]}" > "$log_dir/command.txt"
printf '\n' >> "$log_dir/command.txt"
"${command[@]}" > "$log_dir/output.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$log_dir/exit-code.txt"
git diff --binary HEAD -- vllm/v1 tests/v1 \
  vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py \
  > "$log_dir/runtime-after.patch"
unchanged=0
cmp -s "$log_dir/runtime-before.patch" "$log_dir/runtime-after.patch" && unchanged=1
printf '%s\n' "$unchanged" > "$log_dir/runtime-unchanged.txt"
tail -n 18 "$log_dir/output.log"
printf 'runtime rc=%s runtime_unchanged=%s log=%s\n' "$rc" "$unchanged" "$log_dir"
exit "$rc"
