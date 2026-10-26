import csv, gzip, pickle, re, sys, numpy as np
from sacrebleu.metrics import BLEU
norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
train = pickle.load(gzip.open("Uni-Sign/data/How2Sign/labels.train", "rb")); tset = {norm(v["text"]) for v in train.values()}
refs = [r["reference"] for r in rows]; hyps = [r["prediction"] for r in rows]
seen = np.array([norm(r) in tset for r in refs]); em = np.array([norm(h) == norm(r) for h, r in zip(hyps, refs)])
b4 = lambda m: BLEU().corpus_score([h for h, k in zip(hyps, m) if k], [[r for r, k in zip(refs, m) if k]]).score
print(f"test clips: {len(rows)} | reference sentence also appears verbatim in train: {seen.sum()} ({100*seen.mean():.1f}%)")
print(f"exact-match predictions: {em.sum()} | of those, sentence also in train: {(em & seen).sum()} | NOT in train: {(em & ~seen).sum()}")
print(f"BLEU-4 all {b4(np.ones(len(rows),bool)):.2f} | sentences seen in train {b4(seen):.2f} (n={seen.sum()}) | sentences NOT in train {b4(~seen):.2f} (n={(~seen).sum()})")
print("\nexact matches whose sentence is NOT in train:")
for r in [r for r, k in zip(refs, em & ~seen) if k][:15]: print(" -", r)
