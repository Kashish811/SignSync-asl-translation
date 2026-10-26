#!/usr/bin/env bash
# mT5-base (2.3GB) + How2Sign pose zips (~8.5GB) + How2Sign pose-only checkpoint (1.2GB). Resumable.
source "$(dirname "$0")/_env.sh"
python download.py --list | tee hf_files.txt | grep -i -E "how2sign_pose|total"
python download.py --mt5 --yes
python download.py --include "how2sign_pose" --yes
ls -la hf_unisign; echo "Next: bash 02_prepare.sh"
