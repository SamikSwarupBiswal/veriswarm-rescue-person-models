#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[local,test]'

for asset in yolov8n-640-best.pt yolov8n-960-best.pt; do
  if [[ -f "models/$asset" ]]; then
    echo "Keeping existing checkpoint for hash verification: models/$asset"
  else
    gh release download candidate-2026-08-23 \
      --repo SamikSwarupBiswal/veriswarm-rescue-person-models \
      --pattern "$asset" --dir models
  fi
done

.venv/bin/python -m veriswarm_rescue_models.cli verify
.venv/bin/python -m pytest
