from pathlib import Path
import ast
import json
import re
import subprocess
import torch

tree = Path('/home/fans/SA-work/MiniMax-M3/rubin-kernel-perf/indexer-dcp-perf/worktrees/vllm-icp-public')
proof = Path('/home/fans/SA-work/MiniMax-M3/rubin-kernel-perf/indexer-dcp-perf/review/vllm-public-consolidation-20261008/logs/writer-boxing-r1')
proof.mkdir(parents=True, exist_ok=True)
headers = Path(torch.__file__).parent / 'include'
libraries = Path(torch.__file__).parent / 'lib'
bindings = (tree / 'csrc/libtorch_stable/torch_bindings.cpp').read_text()
definition = bindings.split('"fused_minimax_m3_qknorm_rope_kv_insert("', 1)[1].split(');', 1)[0]
schema = 'fused_minimax_m3_qknorm_rope_kv_insert(' + ''.join(
    ast.literal_eval(part) for part in re.findall(r'"[^"\n]*"', definition)
)
decl = (tree / 'csrc/libtorch_stable/ops.h').read_text().split('void fused_minimax_m3_qknorm_rope_kv_insert(', 1)[1].split(';', 1)[0]
source = '''#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>
#include <optional>
#include <string>
void writer_probe('''+decl+''' {
  STD_TORCH_CHECK(icp_plan_abi == 3, "ABI default did not unbox as 3");
  if (index_world_size == 0) {
    STD_TORCH_CHECK(enable_pdl && !write_icp_metadata, "legacy defaults changed");
    STD_TORCH_CHECK(!icp_positions && !icp_plan_ranges, "optional defaults changed");
    *qkv.mutable_data_ptr<int32_t>() = 25;
  } else {
    STD_TORCH_CHECK(index_world_size == 2 && index_block_tokens == 128 &&
       index_rows_per_rank == 64 && index_rank == 1 && !enable_pdl &&
       write_icp_metadata && icp_num_reqs == 7 && icp_chunk_width == 1024 &&
       icp_plan_num_ctas == 132 && icp_plan_num_heads == 4 &&
       icp_plan_max_splits == 1 && icp_plan_row_begin == 2 &&
       icp_plan_tile_pages == 4 && icp_plan_min_split_tiles == 8,
       "appended scalar arguments did not unbox in signature order");
    STD_TORCH_CHECK(icp_query_start_loc && icp_seq_lens && icp_positions &&
       icp_active && icp_local_nvalid && icp_local_forced && icp_global_nvalid &&
       icp_forced && icp_n_ordinary && icp_candidates && icp_qo_offsets &&
       icp_plan_segments && icp_plan_work && icp_plan_header && icp_plan_ranges,
       "appended tensor arguments did not unbox");
    *icp_positions->mutable_data_ptr<int32_t>() = 35;
    *icp_plan_ranges->mutable_data_ptr<int32_t>() = 64;
    *qkv.mutable_data_ptr<int32_t>() = 64;
  }
}
STABLE_TORCH_LIBRARY(_msa_writer_boxing_probe, m) {
  m.def(R"SCHEMA('''+schema+''')SCHEMA");
}
STABLE_TORCH_LIBRARY_IMPL(_msa_writer_boxing_probe, CPU, m) {
  m.impl("fused_minimax_m3_qknorm_rope_kv_insert", TORCH_BOX(&writer_probe));
}
'''
cpp = proof / 'boxing-probe.cpp'
lib = proof / 'boxing-probe.so'
cpp.write_text(source)
cmd = ['/usr/bin/g++', '-std=c++17', '-shared', '-fPIC', '-O0',
       '-DTORCH_TARGET_VERSION=0x020B000000000000ULL', f'-I{headers}',
       str(cpp), '-o', str(lib), f'-L{libraries}', '-ltorch_cpu', '-lc10',
       f'-Wl,-rpath,{libraries}']
(proof / 'boxing-command.json').write_text(json.dumps(cmd, indent=2)+'\n')
with (proof / 'boxing-build.log').open('w') as log:
    subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
torch.ops.load_library(str(lib))
op = torch.ops._msa_writer_boxing_probe.fused_minimax_m3_qknorm_rope_kv_insert
t = torch.zeros(1, dtype=torch.int32)
inputs = dict(qkv=t, q_norm_weight=t, k_norm_weight=t, cos_sin_cache=t,
    positions=t, num_heads=32, num_kv_heads=2, rotary_dim=64, eps=1e-6,
    index_q_norm_weight=None, index_k_norm_weight=None, num_index_heads=0,
    slot_mapping=None, index_slot_mapping=None, kv_cache=None, index_cache=None,
    block_size=0, q_out=None, index_q_out=None, kv_cache_dtype='auto')
op(**inputs)
assert t.item() == 25
meta = {name: torch.zeros(1, dtype=torch.int32) for name in (
    'icp_query_start_loc', 'icp_seq_lens', 'icp_positions', 'icp_active',
    'icp_local_nvalid', 'icp_local_forced', 'icp_global_nvalid', 'icp_forced',
    'icp_n_ordinary', 'icp_candidates', 'icp_qo_offsets', 'icp_plan_segments',
    'icp_plan_work', 'icp_plan_header', 'icp_plan_ranges')}
op(**inputs, **meta, index_world_size=2, index_block_tokens=128,
    index_rows_per_rank=64, index_rank=1, enable_pdl=False, write_icp_metadata=True,
    icp_num_reqs=7, icp_chunk_width=1024, icp_plan_num_ctas=132,
    icp_plan_num_heads=4, icp_plan_max_splits=1, icp_plan_row_begin=2,
    icp_plan_tile_pages=4, icp_plan_min_split_tiles=8)
assert t.item() == 64
assert meta['icp_positions'].item() == 35
assert meta['icp_plan_ranges'].item() == 64
result = {'schema_argument_count':len(op.default._schema.arguments),
          'legacy_defaults': 'passed', 'appended_scalars_and_optional_tensors':'passed',
          'mutable_tensor_aliases':'passed', 'device':'CPU',
          'scope':'actual schema + actual C++ declaration with CPU probe body, not CUDA writer execution'}
(proof/'boxing-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
