#!/usr/bin/env bash
# Custom vLLM entrypoint: serve a model unpacked from a KitOps ModelKit that the
# kitunpacker initContainer placed on a shared volume. This deliberately serves
# from a LOCAL directory instead of the default `vllm serve <hf-repo>` so there
# is no Hugging Face download at runtime.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models/qwen3}"
SERVED_NAME="${SERVED_NAME:-qwen3-4b-instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
PORT="${PORT:-8000}"

echo "[vllm-jozu] serving KitOps model from ${MODEL_DIR} (source: Jozu Hub, no HF download)"
if [ ! -f "${MODEL_DIR}/config.json" ]; then
  echo "[vllm-jozu] ERROR: ${MODEL_DIR}/config.json not found -- did the kitunpacker init run?" >&2
  exit 1
fi

exec vllm serve "${MODEL_DIR}" \
  --served-model-name "${SERVED_NAME}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --host 0.0.0.0 \
  --port "${PORT}"
