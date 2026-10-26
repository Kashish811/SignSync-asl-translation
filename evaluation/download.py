"""Download Uni-Sign How2Sign pose files + checkpoints + mT5-base.  Run inside the activated env, from the project root (parent of Uni-Sign).
  python download.py --list                      # prints every file in ZechengLi19/Uni-Sign with size (SEND THIS TO CLAUDE if unsure)
  python download.py --include how2sign --yes    # downloads files whose path matches regex 'how2sign' (case-insens.)
  python download.py --mt5 --yes                 # google/mt5-base -> Uni-Sign/pretrained_weight/mt5-base
"""
import argparse, os, re, shutil, sys
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
REPO = "ZechengLi19/Uni-Sign"
ap = argparse.ArgumentParser()
ap.add_argument("--list", action="store_true"); ap.add_argument("--include", default=None)
ap.add_argument("--exclude", default=None); ap.add_argument("--mt5", action="store_true")
ap.add_argument("--yes", action="store_true"); ap.add_argument("--dest", default="hf_unisign")
a = ap.parse_args(); api = HfApi()
files = [(f.path, getattr(f, "size", None) or 0) for f in api.list_repo_tree(REPO, recursive=True) if hasattr(f, "size")]
gb = lambda b: f"{b/1e9:.2f} GB"
if a.list or (a.include is None and not a.mt5):
    for p, s in sorted(files): print(f"{gb(s):>10}  {p}")
    print("total", gb(sum(s for _, s in files))); sys.exit(0)
if a.include:
    sel = [(p, s) for p, s in files if re.search(a.include, p, re.I) and not (a.exclude and re.search(a.exclude, p, re.I))]
    tot = sum(s for _, s in sel); free = shutil.disk_usage(".").free
    print(f"selected {len(sel)} files, {gb(tot)}; free disk {gb(free)} (zip merge + unzip needs ~3x the zip size)")
    for p, s in sel: print(f"  {gb(s):>10}  {p}")
    if tot * 3 > free: print("NOT ENOUGH DISK for download+merge+unzip -> free space first (or use an external drive)"); sys.exit(2)
    if not a.yes: print("re-run with --yes to download"); sys.exit(0)
    for p, _ in sel: print("downloading", p); hf_hub_download(REPO, p, local_dir=a.dest)   # resumable
if a.mt5:
    d = "Uni-Sign/pretrained_weight/mt5-base"
    if not a.yes: print("would download google/mt5-base ->", d, "(~2.3 GB); add --yes"); sys.exit(0)
    snapshot_download("google/mt5-base", local_dir=d, ignore_patterns=["*.h5", "flax_model*", "*.msgpack", "tf_model*", "*.ot"])
    print(sorted(os.listdir(d)))
