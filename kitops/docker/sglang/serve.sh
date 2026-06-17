#!/usr/bin/env bash
# Custom SGLang entrypoint: serve a model unpacked from a KitOps ModelKit that
# the kitunpacker initContainer placed on a shared volume. Serves from a LOCAL
# directory (--model-path) so there is no Hugging Face download at runtime.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models/qwen3}"
SERVED_NAME="${SERVED_NAME:-qwen3-4b-instruct}"
CONTEXT_LEN="${CONTEXT_LEN:-8192}"
MEM_FRACTION="${MEM_FRACTION:-0.8}"
PORT="${PORT:-30000}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"

echo "[sglang-jozu] serving KitOps model from ${MODEL_DIR} (source: Jozu Hub, no HF download)"
if [ ! -f "${MODEL_DIR}/config.json" ]; then
  echo "[sglang-jozu] ERROR: ${MODEL_DIR}/config.json not found -- did the kitunpacker init run?" >&2
  exit 1
fi

exec python3 -m sglang.launch_server \
  --model-path "${MODEL_DIR}" \
  --served-model-name "${SERVED_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --mem-fraction-static "${MEM_FRACTION}" \
  --context-length "${CONTEXT_LEN}" \
  --attention-backend "${ATTENTION_BACKEND}"
