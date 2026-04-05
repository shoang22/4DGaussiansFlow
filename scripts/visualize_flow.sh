#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

python "$WORKSPACE/scripts/visualize_flow.py" \
    --image_dir "$WORKSPACE/data/multipleview/background1" \
    --flow_dir  "$WORKSPACE/output/flows" \
    --out_dir   "$WORKSPACE/output/flow_vis" \
    --cam_ids   1 2 3 \
    --frames    1 5 10
