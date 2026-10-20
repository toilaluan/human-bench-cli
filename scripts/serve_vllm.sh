#!/usr/bin/env bash
# Run inside a separate vLLM environment on the GPU host, not the benchmark venv.
set -euo pipefail

args=(
  "${VLM_MODEL:-Qwen/Qwen3.8-27B-FP8}"
  --host "${VLM_HOST:-127.0.0.1}"
  --port "${VLM_PORT:-8000}"
  --tensor-parallel-size "${VLM_TP:-1}"
  --max-model-len "${VLM_CONTEXT:-32768}"
  --max-num-seqs "${VLM_MAX_SEQS:-4}"
  --gpu-memory-utilization "${VLM_MEMORY:-0.85}"
  --limit-mm-per-prompt '{"image":1}'
  --generation-config vllm
  --reasoning-parser qwen3
)
if [[ -n "${VLM_REVISION:-}" ]]; then
  args+=(--revision "$VLM_REVISION")
fi
exec vllm serve "${args[@]}" "$@"
