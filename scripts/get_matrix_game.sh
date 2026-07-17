#!/usr/bin/env bash
# Fetch the Matrix-Game-2 source + weights that Butterfly drives.
#   source  -> third_party/Matrix-Game (SkyworkAI/Matrix-Game, MIT)
#   weights -> ckpts/Matrix-Game-2.0   (~28 GB from Hugging Face)
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p third_party ckpts
if [ ! -d third_party/Matrix-Game ]; then
  git clone --depth 1 https://github.com/SkyworkAI/Matrix-Game.git third_party/Matrix-Game
fi

python -m pip show huggingface_hub >/dev/null 2>&1 || python -m pip install huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Skywork/Matrix-Game-2.0", local_dir="ckpts/Matrix-Game-2.0")
print("weights ready: ckpts/Matrix-Game-2.0")
PY
echo "done. next: bash scripts/serve.sh"
