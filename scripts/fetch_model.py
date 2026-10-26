"""Fetch the Uni-Sign code, the pose-only How2Sign checkpoint and mT5-base (all Uni-Sign / Google assets, not part of this repo).

  python scripts/fetch_model.py
Uni-Sign code and weights: Li et al., CC BY-NC 4.0 (non-commercial use only; attribution required).
"""
import os
import subprocess

from huggingface_hub import hf_hub_download, snapshot_download

REPO = "ZechengLi19/Uni-Sign"
CKPT_FILE = "how2sign_pose_only_slt.pth"

if not os.path.isdir("Uni-Sign"):
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ZechengLi19/Uni-Sign"], check=True)

os.makedirs("hf_unisign", exist_ok=True)
hf_hub_download(REPO, CKPT_FILE, local_dir="hf_unisign")                      # ~1.2 GB

mt5_dir = "Uni-Sign/pretrained_weight/mt5-base"                                 # where the Uni-Sign code looks for mT5
if not os.path.isdir(mt5_dir):
    snapshot_download("google/mt5-base", local_dir=mt5_dir,
                      ignore_patterns=["*.h5", "flax_model*", "*.msgpack", "tf_model*", "*.ot"])   # ~2.3 GB

rtm = "Uni-Sign/demo/rtmlib-main"
if os.path.isdir(rtm):
    subprocess.run(["python", "-m", "pip", "install", "-q", "-e", rtm], check=True)   # pose extractor used by the demo
print("model assets ready")
