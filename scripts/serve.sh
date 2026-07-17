#!/usr/bin/env bash
# Serve Butterfly on http://localhost:7860 (one GPU, >=24 GB VRAM).
set -euo pipefail
cd "$(dirname "$0")/.."

MG2_ROOT=${SAVEPOINT_MG2_ROOT:-$PWD/third_party/Matrix-Game/Matrix-Game-2}
export SAVEPOINT_MG2_ROOT=$MG2_ROOT
export SAVEPOINT_CKPT_DIR=${SAVEPOINT_CKPT_DIR:-$PWD/ckpts/Matrix-Game-2.0}
export SAVEPOINT_IMAGE_DIR=${SAVEPOINT_IMAGE_DIR:-$MG2_ROOT/demo_images/universal}
export SAVEPOINT_SAVE_DIR=${SAVEPOINT_SAVE_DIR:-$PWD/saves}
export PYTHONPATH=$PWD/src:$MG2_ROOT${PYTHONPATH:+:$PYTHONPATH}

cd "$MG2_ROOT"   # MG2 configs use repo-relative paths
exec python -m uvicorn server.app:app --host 0.0.0.0 --port "${PORT:-7860}"
