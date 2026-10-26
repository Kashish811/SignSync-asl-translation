"""Collect results/*_metrics.json -> results/RESULTS.md (measured numbers only) and print the literature table.
Published numbers below were gathered from papers while planning; VERIFY each against its paper before putting it in slides."""
import glob, json, os, sys
R = sys.argv[1] if len(sys.argv) > 1 else "results"
rows = []
for f in sorted(glob.glob(os.path.join(R, "*_metrics.json"))):
    d = json.load(open(f)); m, t = d["metrics"], d["meta"]
    rows.append(f"| {t['tag']} | {t['split']} | {t['n_scored']} | {t['control']}/{','.join(t['drop']) or '-'} | " +
                " | ".join(f"{m[k]:.2f}" for k in ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L")) + f" | {m.get('BLEURT-20', float('nan')):.1f} |")
hdr = "| run | split | clips | control/dropped parts | B1 | B2 | B3 | B4 | ROUGE-L | BLEURT |\n|---|---|---|---|---|---|---|---|---|---|\n"
lit = """
## Published How2Sign results (BLEU-4; to verify against each paper before citing)
| method | BLEU-4 | note |
|---|---|---|
| H2S-only baselines | ~0.9-2.3 | no large-scale pretraining |
| Tarres et al. | 8.03 | |
| SignFormer-GCN | 8.53 | ST-GCN keypoints + RGB features |
| YouTube-ASL | 12.39 | pretrain + finetune |
| Uni-Sign (RGB-pose) | 14.9 | ST-GCN pose encoder + mT5 |
| Uni-Sign (pose-only) | ~14.5 | read from table; verify |
| SSVP-SLT | 15.5 | |
"""
open(os.path.join(R, "RESULTS.md"), "w").write("# Measured results (this project)\n\n" + hdr + "\n".join(rows) + "\n" + lit)
print(open(os.path.join(R, "RESULTS.md")).read())
