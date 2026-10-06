#!/usr/bin/env bash
# Start Clef on vLLM in a detached container (3-4 min to healthy; check GET /health).
#   MODEL_ROOT=/data/clef PORT=8000 ./start.sh [extra clef-systemone flags]
# Stop: docker stop clef-vllm    Restart the same container: docker start clef-vllm
# No restart policy is set; add --restart to the docker run below if you want one.
set -euo pipefail
cd "$(dirname "$0")"
IMAGE=${IMAGE:-clef-vllm:0.28.0}
NAME=${NAME:-clef-vllm}
PORT=${PORT:-8000}
MODEL_ROOT=$(realpath "${MODEL_ROOT:-./models}")

docker rm "$NAME" >/dev/null 2>&1 || true   # only removes a stopped container
mkdir -p "$HOME/.cache/flashinfer" "$HOME/.cache/vllm"
exec docker run -d --name "$NAME" --gpus all --network host --ipc host \
  -v "$MODEL_ROOT":/models:ro \
  -v "$HOME/.cache/flashinfer":/root/.cache/flashinfer \
  -v "$HOME/.cache/vllm":/root/.cache/vllm \
  "$IMAGE" \
  --model /models/clef-vllm-view --served-model-name clef --port "$PORT" \
  --max-model-len 65536 --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" "$@"
