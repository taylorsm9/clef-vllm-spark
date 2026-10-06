#!/usr/bin/env bash
# Build the image and download the original BF16 Clef release at a pinned commit.
#   ./setup.sh                       # models go to ./models
#   MODEL_ROOT=/data/clef ./setup.sh
# Set HF_TOKEN if your Hugging Face account needs one. Needs ~55 GB of free disk.
set -euo pipefail
cd "$(dirname "$0")"
IMAGE=${IMAGE:-clef-vllm:0.28.0}
MODEL_ROOT=$(realpath -m "${MODEL_ROOT:-./models}")
COMMIT=2f3de3dd85f379784083b0814d997ab627200f0c
RELEASE=clef-${COMMIT:0:7}

docker build -t "$IMAGE" -f Dockerfile vllm_plugin

mkdir -p "$MODEL_ROOT"
docker run --rm --entrypoint python3 -e HF_TOKEN -v "$MODEL_ROOT":/models "$IMAGE" -c "
from huggingface_hub import snapshot_download
snapshot_download('Cloudflare/clef', revision='$COMMIT', local_dir='/models/$RELEASE')
"

# The plugin expects the joint head under clef_head/. Build that layout with relative
# symlinks so the weights are not copied.
VIEW="$MODEL_ROOT/clef-vllm-view"
mkdir -p "$VIEW/clef_head"
for f in "$MODEL_ROOT/$RELEASE"/*; do
  ln -sfn "../$RELEASE/$(basename "$f")" "$VIEW/$(basename "$f")"
done
rm -f "$VIEW/__pycache__"
ln -sfn "../../$RELEASE/joint_head.safetensors" "$VIEW/clef_head/joint_head.safetensors"
ln -sfn "../../$RELEASE/joint_head_config.json" "$VIEW/clef_head/joint_head_config.json"
echo "Ready: MODEL_ROOT=$MODEL_ROOT ./start.sh"
