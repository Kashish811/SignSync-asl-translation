#!/usr/bin/env bash
set -euo pipefail; source "$(dirname "$0")/config.env"; cd "$SS_HOME"
echo "== machine =="; (nvidia-smi 2>&1 | head -12) || echo "no nvidia-smi"; nproc; free -h | sed -n 1,2p; df -h . | tail -1
[ -d Uni-Sign ] || git clone https://github.com/ZechengLi19/Uni-Sign
if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c 'import sys;assert sys.version_info[:2] in ((3,9),(3,10),(3,11))' 2>/dev/null; then
  rm -rf "$VENV"
  if command -v conda >/dev/null; then conda create -y -q -p "$VENV" python=3.10 pip
  else
    PY=""; for c in python3.10 python3.11 python3.9; do command -v $c >/dev/null && PY=$c && break; done
    [ -n "$PY" ] || { echo "Need python 3.9-3.11 or conda."; exit 1; }
    $PY -m venv "$VENV"
  fi
fi
export PATH="$VENV/bin:$PATH"; python --version
python -m pip install -q --upgrade pip
python -m pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -q einops sentencepiece sacrebleu rouge huggingface_hub timm "transformers>=4.40,<4.50" "numpy<2" pillow tqdm
python - <<'PY'
import torch, transformers, sys
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda available:", torch.cuda.is_available(), "| transformers", transformers.__version__)
if torch.cuda.is_available(): print("gpu:", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
PY
echo "OK. Next: bash 01_download.sh"
