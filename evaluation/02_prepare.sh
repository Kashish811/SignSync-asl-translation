#!/usr/bin/env bash
# unzip poses, build labels from official CSVs, patch config, verify names.  Need ~25GB free for unzip.
source "$(dirname "$0")/_env.sh"
ls "$CSV_DIR"/how2sign_realigned_*.csv || { echo "CSV_DIR wrong: edit config.env"; exit 1; }
df -h . | tail -1
python prepare.py --hf_dir hf_unisign --csv_dir "$CSV_DIR" --repo Uni-Sign
cat prepare_report.json; echo "All three splits must say PASS. Next: bash 03_smoke.sh"
