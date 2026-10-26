#!/usr/bin/env bash
# Run the web app on the college PC (GPU machine). Uses the model assets already in ~/Downloads/SignSync_PC.
#   bash run_local.sh            then open http://localhost:8000
set -e
SS="${SS_HOME:-$HOME/Downloads/SignSync_PC}"
export UNISIGN_DIR="$SS/Uni-Sign"
export CKPT="$SS/hf_unisign/how2sign_pose_only_slt.pth"
export PYTHONNOUSERSITE=1; unset LD_LIBRARY_PATH || true
PY="$SS/.venv/bin/python"
"$PY" -m pip install -q fastapi "uvicorn[standard]" python-multipart
exec "$PY" -m uvicorn main:app --app-dir app --host 127.0.0.1 --port 8000
