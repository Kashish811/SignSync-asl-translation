#!/usr/bin/env bash
# usage: bash 06_demo.sh path/to/clip.mp4   (needs rtmlib; installs it on first use)
source "$(dirname "$0")/_env.sh"
python -m pip install -q onnxruntime-gpu 2>/dev/null || python -m pip install -q onnxruntime
(cd Uni-Sign/demo/rtmlib-main && python -m pip install -q -e .)
python signsync_demo.py --ckpt "$CKPT" --video "$1"
