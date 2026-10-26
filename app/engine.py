"""Translation engine: wraps Uni-Sign's own demo pipeline (RTMPose pose extraction -> part-aware GNN -> mT5).

The model code and weights are NOT part of this repository. They are Uni-Sign's
(Li et al., CC BY-NC 4.0) and are fetched by `scripts/fetch_model.py`.
"""
import os
import re
import subprocess
import sys
import time

UNISIGN_DIR = os.path.abspath(os.environ.get("UNISIGN_DIR", "Uni-Sign"))
CKPT = os.path.abspath(os.environ.get("CKPT", "hf_unisign/how2sign_pose_only_slt.pth"))
TIMEOUT_S = int(os.environ.get("TRANSLATE_TIMEOUT", "600"))


def status():
    """Return (live, reason). Live translation needs the Uni-Sign code, the checkpoint and the demo script."""
    if not os.path.isdir(UNISIGN_DIR):
        return False, "Uni-Sign code not found (run scripts/fetch_model.py)"
    if not os.path.exists(CKPT):
        return False, "model checkpoint not found (run scripts/fetch_model.py)"
    if not os.path.exists(os.path.join(UNISIGN_DIR, "demo", "online_inference.py")):
        return False, "Uni-Sign demo script not found"
    return True, "ready"


def translate(video_path):
    """Run the pipeline on one video file. Returns (sentence, seconds)."""
    started = time.time()
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_demo.py")   # adds placeholder modules, then runs the demo
    proc = subprocess.run(
        [sys.executable, runner, os.path.abspath(video_path), CKPT],
        cwd=UNISIGN_DIR, capture_output=True, text=True, timeout=TIMEOUT_S,
        env={**os.environ, "PYTHONPATH": UNISIGN_DIR},
    )
    match = re.search(r"Prediction result is:\s*(.*)", proc.stdout)
    if not match:
        raise RuntimeError("no prediction produced: " + (proc.stderr or proc.stdout)[-400:])
    return match.group(1).strip(), round(time.time() - started, 1)
