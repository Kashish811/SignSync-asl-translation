import argparse, csv, re, os, numpy as np
from sacrebleu.metrics import BLEU
ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--out", default="results/ANALYSIS.md")
a = ap.parse_args(); rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
refs = [r["reference"] for r in rows]; hyps = [r["prediction"] for r in rows]; sb = np.array([float(r["sentence_bleu"]) for r in rows]); n = len(rows)
bleu4 = lambda h, r: BLEU().corpus_score(h, [r]).score
rng = np.random.default_rng(0); bs = []
for _ in range(a.boot):
    idx = rng.integers(0, n, n); bs.append(bleu4([hyps[i] for i in idx], [refs[i] for i in idx]))
lo, hi = np.percentile(bs, [2.5, 97.5])
norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
exact = sum(norm(h) == norm(r) for h, r in zip(hyps, refs))
def has_rep(h, k=3):
    w = h.split(); g = [tuple(w[i:i + k]) for i in range(len(w) - k + 1)]; return len(g) != len(set(g))
rep = sum(has_rep(h) for h in hyps)
rl = np.array([len(r.split()) for r in refs]); out = [f"# Error analysis ({a.csv}, n={n})\n",
    f"- Corpus BLEU-4 = {bleu4(hyps, refs):.2f}; bootstrap 95% CI ({a.boot} resamples of sentences) = [{lo:.2f}, {hi:.2f}]",
    f"- Exact matches (case/punct-insensitive): {exact} ({100*exact/n:.1f}%)",
    f"- Hypotheses with a repeated 3-gram: {rep} ({100*rep/n:.1f}%)",
    f"- Avg words: reference {rl.mean():.1f}, hypothesis {np.mean([len(h.split()) for h in hyps]):.1f}; distinct hypotheses {len(set(hyps))/n:.2f}\n",
    "## BLEU-4 by reference length\n| ref words | clips | corpus BLEU-4 |\n|---|---|---|"]
for lo_, hi_ in ((1, 5), (6, 10), (11, 20), (21, 40), (41, 999)):
    m = (rl >= lo_) & (rl <= hi_)
    if m.sum() >= 20: out.append(f"| {lo_}-{hi_ if hi_<999 else '+'} | {int(m.sum())} | {bleu4([h for h,k in zip(hyps,m) if k],[r for r,k in zip(refs,m) if k]):.2f} |")
order = np.argsort(-sb)
def show(title, ids):
    out.append(f"\n## {title}\n| name | reference | prediction | sent-BLEU |\n|---|---|---|---|")
    for i in ids: out.append(f"| {rows[i]['name'][:28]} | {refs[i]} | {hyps[i]} | {sb[i]:.1f} |")
show("Best 12 (highest sentence BLEU, ref >= 6 words)", [i for i in order if rl[i] >= 6][:12])
rng2 = np.random.default_rng(1); show("12 random predictions (unselected, for an honest picture)", sorted(rng2.choice(n, 12, replace=False)))
show("Worst-looking 12 among long references (ref >= 10 words, lowest sentence BLEU)", [i for i in order[::-1] if rl[i] >= 10][:12])
os.makedirs(os.path.dirname(a.out), exist_ok=True); open(a.out, "w", encoding="utf-8").write("\n".join(out)); print("\n".join(out[:12])); print("wrote", a.out)
