"""Build app/data/examples.json and stats.json from your evaluation results (real numbers, nothing typed by hand).

Run from the SignSync_PC project folder (where results/ lives), pointing at this app:
  .venv/bin/python /path/to/SignSync_App/scripts/make_examples.py --results results --out /path/to/SignSync_App/app/data
Examples are chosen by a fixed seeded rule: 2 exact matches, 2 typical, 2 failures (+ 2 random) - failures are shown on purpose.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
from sacrebleu.metrics import BLEU

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="app/data")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

# ---- per-clip predictions from the full test run
rows = list(csv.DictReader(open(os.path.join(args.results, "main_test_predictions.csv"), encoding="utf-8")))
sb = np.array([float(r["sentence_bleu"]) for r in rows])
rl = np.array([len(r["reference"].split()) for r in rows])
rng = np.random.default_rng(7)


def pick(pool, k):
    return [int(i) for i in rng.choice(pool, min(k, len(pool)), replace=False)] if len(pool) else []


best = [i for i in np.argsort(-sb) if rl[i] >= 6 and sb[i] >= 99][:60]
mid = [i for i in range(len(rows)) if 15 < sb[i] < 60]
worst = [i for i in np.argsort(sb) if rl[i] >= 10][:60]
chosen = [("Exact match", i) for i in pick(best, 2)] + [("Partly right", i) for i in pick(mid, 2)] + [("Failure", i) for i in pick(worst, 2)]
examples = [{"kind": k, "reference": rows[i]["reference"], "prediction": rows[i]["prediction"], "score": round(float(sb[i]), 1)} for k, i in chosen]
json.dump(examples, open(os.path.join(args.out, "examples.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

# ---- headline numbers with a bootstrap interval
main = json.load(open(os.path.join(args.results, "main_test_metrics.json")))
refs = [r["reference"] for r in rows]
hyps = [r["prediction"] for r in rows]
boot = []
for _ in range(300):
    idx = rng.integers(0, len(rows), len(rows))
    boot.append(BLEU().corpus_score([hyps[i] for i in idx], [[refs[i] for i in idx]]).score)
lo, hi = np.percentile(boot, [2.5, 97.5])

controls = {}
for f in glob.glob(os.path.join(args.results, "sub_*_metrics.json")):
    d = json.load(open(f))
    controls[d["meta"]["tag"].replace("sub_", "")] = round(d["metrics"]["BLEU-4"], 2)

stats = {
    "n_clips": main["meta"]["n_scored"],
    "bleu1": round(main["metrics"]["BLEU-1"], 1), "bleu2": round(main["metrics"]["BLEU-2"], 1),
    "bleu3": round(main["metrics"]["BLEU-3"], 1), "bleu4": round(main["metrics"]["BLEU-4"], 2),
    "bleu4_ci": [round(float(lo), 2), round(float(hi), 2)],
    "rouge_l": round(main["metrics"]["ROUGE-L"], 1),
    "exact_pct": round(float((sb >= 99.99).mean() * 100), 1),
    "controls": controls,
}
json.dump(stats, open(os.path.join(args.out, "stats.json"), "w"), indent=1)
print("wrote", os.path.join(args.out, "examples.json"), "and stats.json")
print(json.dumps(stats, indent=1))
