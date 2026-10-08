Public CPU test sanity without companion MSA: **144 passed, 30 skipped**, zero failures/errors, 8.30 seconds. The before/after hashes of all MiniMax M3 Python source/test files match.

`probe.json` establishes that the unchanged uv-managed interpreter cannot find `fmha_sm100`; `PYTHONPATH` contains only the public vLLM candidate. The older `icp_kernels` distribution remains installed in this shared environment, but it does not provide the canonical `fmha_sm100.icp` import used by the tests. No dependency was installed or removed.

The passing command uses `--noconftest` and disables plugin autoload. Normal collection was attempted first and failed because the sealed environment lacks `tblib`, imported by repository-wide `tests/conftest.py:7`. The full CI conftest environment is therefore not qualified by this local check. The separate pytest process in the CPU job also prevents the tests' scoped import fixtures from affecting unrelated model suites.

Passing contracts include compound page/spec/runtime handling, optional dependency and writer capability errors, offload admission, layer metadata binding, projection replication, phase/transport policies that do not need MSA, and final decode-plan ownership. Package-owned helpers, actual canonical main-attention views/dispatch and indexer-window tests skip when MSA is absent. Module-level skips collapse multiple parametrized tests into one skip record, so the skipped count is not the difference from the full companion-enabled suite.

Exact commands, effective environment, collection error, pytest output/XML, per-file coverage and source hashes are retained beside this file. Reproducer: `review/vllm-public-consolidation-20261008/run_model_no_companion.py` (uses a fresh fixed log directory).
