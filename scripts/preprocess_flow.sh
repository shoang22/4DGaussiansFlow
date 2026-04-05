#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

python "$WORKSPACE/preprocess_flow.py" \
    --image_dir "$WORKSPACE/data/multipleview/background1" \
    --flow_dir  "$WORKSPACE/output/flows" \
    --num_cameras 12 \
    --num_frames  30 \
    --model   gmflow \
    --weights "$WORKSPACE/gmflow/checkpoints/gmflow_sintel-0c07dcb3.pth" \
    --device  cuda
