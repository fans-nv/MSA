#!/usr/bin/env bash
set -uo pipefail
audit_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
workspace_root=$(cd "$audit_dir/../.." && pwd)
tag=${1:?usage: run_checks.sh TAG integration|msa|writer|types [type targets...]}
gate=${2:?usage: run_checks.sh TAG integration|msa|writer|types [type targets...]}
shift 2
if [[ ! "$tag" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
  echo "Invalid evidence tag: $tag" >&2
  exit 2
fi
vllm_tree="$workspace_root/worktrees/vllm-icp-public"
msa_tree="$workspace_root/worktrees/msa-dev-public"
interpreter="$workspace_root/review/v13-formal/.venv/bin/python"
export PATH="$(dirname "$interpreter"):$PATH"
export PYTHONPATH="$vllm_tree:$msa_tree/python"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export CUDA_VISIBLE_DEVICES= VLLM_TARGET_DEVICE=cpu ICP_KERNEL_ARCH=107a
cache_root="$audit_dir/.cache/$tag/$gate"
export XDG_CACHE_HOME="$cache_root"
export UV_CACHE_DIR="$cache_root/uv"
export MYPY_CACHE_DIR="$cache_root/mypy"
export VLLM_CACHE_ROOT="$cache_root/vllm"
export VLLM_CONFIG_ROOT="$cache_root/vllm-config"
export HF_HOME="$cache_root/huggingface"
export TORCH_EXTENSIONS_DIR="$cache_root/torch-extensions"
export ICP_CACHE_ROOT="$cache_root/icp"
log_dir="$audit_dir/logs/$tag/$gate"
if [[ -e "$log_dir" ]]; then
  echo "Refusing to replace existing evidence: $log_dir" >&2
  exit 2
fi
mkdir -p "$log_dir"
run_dir="$vllm_tree"
case "$gate" in
  integration)
    command=("$interpreter" -m pytest -q -rs -p no:cacheprovider --noconftest
      tests/models/minimax_m3
      tests/v1/test_kv_cache_spec_registry.py
      tests/v1/simple_kv_offload/test_shutdown.py
      tests/v1/cudagraph/test_cudagraph_manager.py
      tests/v1/worker/test_gpu_model_runner_v2_cudagraph_profiling.py
      tests/v1/worker/test_attn_utils.py
      tests/v1/worker/test_gpu_batch_ordering.py
      tests/v1/worker/test_gpu_ubatch_slicing.py
      tests/kernels/test_fused_minimax_m3_icp_writer_api.py
      tests/kernels/test_minimax_m3_icp_device_plan.py
      tests/kernels/test_fused_minimax_m3_icp_writer.py
      tests/kernels/test_fused_minimax_m3_qknorm_rope_kv_insert.py
      --junitxml="$log_dir/pytest.xml")
    ;;
  msa)
    run_dir="$msa_tree"
    command=("$interpreter" -m pytest -q -rs -p no:cacheprovider
      -m 'not gpu and not distributed' tests/icp
      tests/regression/test_icp_consolidation.py
      tests/regression/test_nvfp4_scale_compatibility.py
      tests/regression/test_icp_dev_compatibility.py
      tests/jit_cache tests/warmup
      --deselect='tests/warmup/test_msa_warmup.py::test_plan_is_split_by_dtype[nvfp4_kv4_spec]'
      --junitxml="$log_dir/pytest.xml")
    ;;
  nvfp4)
    run_dir="$msa_tree"
    command=("$interpreter" -m pytest -q -rs -p no:cacheprovider
      tests/icp/test_prewarm_cache.py -k 'nvfp4 or production or import_'
      --junitxml="$log_dir/pytest.xml")
    ;;
  writer)
    command=("$interpreter" -m pytest -q -rs -p no:cacheprovider --noconftest
      tests/kernels/test_fused_minimax_m3_icp_writer_api.py
      --junitxml="$log_dir/pytest.xml")
    ;;
  types)
    if (($# == 0)); then
      echo "types requires explicit repository-relative file targets" >&2
      exit 2
    fi
    command=("$interpreter" tools/pre_commit/mypy.py
      "${ICP_MYPY_PYTHON_VERSION:-3.11}" "$@")
    ;;
  *) echo "Unknown gate: $gate" >&2; exit 2 ;;
esac
record_sources() {
  local stage=$1 component tree
  for component in vllm msa; do
    if [[ "$component" == vllm ]]; then tree="$vllm_tree"; else tree="$msa_tree"; fi
    git -C "$tree" rev-parse HEAD > "$log_dir/$component-$stage-head.txt"
    git -C "$tree" status --short > "$log_dir/$component-$stage-status.txt"
    git -C "$tree" diff --binary HEAD > "$log_dir/$component-$stage-working.patch"
    (
      cd "$tree" || exit
      {
        git diff --name-only --diff-filter=ACMRT -z HEAD
        git ls-files --others --exclude-standard -z
      } | sort -zu | xargs -0 -r sha256sum --
    ) > "$log_dir/$component-$stage-working-files.sha256"
  done
}
record_sources before
printf '%s\n' "$PYTHONPATH" > "$log_dir/pythonpath.txt"
printf '%s\n' "$run_dir" > "$log_dir/cwd.txt"
printf '%q ' "${command[@]}" > "$log_dir/command.txt"
printf '\n' >> "$log_dir/command.txt"
cd "$run_dir" || exit
"${command[@]}" > "$log_dir/output.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$log_dir/exit-code.txt"
record_sources after
sources_unchanged=1
for component in vllm msa; do
  for suffix in head.txt status.txt working.patch working-files.sha256; do
    if ! cmp -s "$log_dir/$component-before-$suffix" "$log_dir/$component-after-$suffix"; then
      printf '%s changed during this gate\n' "$component/$suffix" >> "$log_dir/source-changes.txt"
      sources_unchanged=0
    fi
  done
done
printf '%s\n' "$sources_unchanged" > "$log_dir/sources-unchanged.txt"
tail -n 16 "$log_dir/output.log"
printf '%s rc=%s sources_unchanged=%s log=%s\n' "$gate" "$rc" "$sources_unchanged" "$log_dir"
if ((rc == 0 && sources_unchanged == 0)); then exit 3; fi
exit "$rc"
