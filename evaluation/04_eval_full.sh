#!/usr/bin/env bash
# Headline result on the FULL official test set. Run once; do not tune on test.
source "$(dirname "$0")/_env.sh"
python signsync_eval.py --ckpt "$CKPT" --split test --tag main_test --batch_size 8 --num_workers 4
