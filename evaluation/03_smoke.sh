#!/usr/bin/env bash
# 20-clip test on GPU: checks checkpoint+data+metrics end-to-end.
source "$(dirname "$0")/_env.sh"
python signsync_eval.py --ckpt "$CKPT" --split test --limit 20 --batch_size 4 --num_workers 2 --tag smoke
head -6 results/smoke_predictions.csv; python -c "import json;print(json.load(open('results/smoke_metrics.json'))['meta'])"
echo "Look at sec_per_clip above. Sentences should be English related to the references. Next: bash 04_eval_full.sh"
