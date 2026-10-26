#!/usr/bin/env bash
# Inference-only GNN insight runs on an evenly spaced test subset (same clips each time).
source "$(dirname "$0")/_env.sh"
N=$TEST_SUBSET; run(){ tag=$1; shift; python signsync_eval.py --ckpt "$CKPT" --split test --limit $N --batch_size 8 --num_workers 4 --tag "$tag" "$@"; }
run sub_full
run sub_shuffled --control shuffle
for P in body left right face_all; do run sub_drop_$P --drop $P; done
run sub_hands_off --drop left,right
python make_tables.py results
