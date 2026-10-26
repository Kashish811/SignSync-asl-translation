"""SignSync evaluation (inference only, no deepspeed).  Run from anywhere:
  python signsync_eval.py --repo Uni-Sign --ckpt <ckpt.pth> --split test --limit 50 --tag smoke
Controls (cheap, no training):  --control shuffle   (pose of sample i fed with sample i+1's mask/pose within a batch)
                                --drop body,left,right,face_all  (zero the chosen GNN part inputs)
Writes results/<tag>_metrics.json and results/<tag>_predictions.csv.  Only measured numbers are written."""
import argparse, csv, json, os, sys, time, hashlib
import numpy as np

PARTS = ["body", "left", "right", "face_all"]

# ---------- metrics (pure python / sacrebleu; unit-tested without a GPU) ----------
def _lcs(a, b):
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]; dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1]); prev = cur
    return dp[-1]

def rouge_l(ref, hyp):
    r, h = ref.split(), hyp.split()
    if not r or not h: return 0.0
    l = _lcs(r, h)
    if l == 0: return 0.0
    p, rc = l / len(h), l / len(r); return 2 * p * rc / (p + rc)

def corpus_metrics(refs, hyps):
    from sacrebleu.metrics import BLEU
    out = {}
    for n in (1, 2, 3, 4):
        out[f"BLEU-{n}"] = BLEU(max_ngram_order=n).corpus_score(hyps, [refs]).score
    out["ROUGE-L"] = 100 * float(np.mean([rouge_l(r, h) for r, h in zip(refs, hyps)]))
    out["avg_hyp_words"] = float(np.mean([len(h.split()) for h in hyps])); out["avg_ref_words"] = float(np.mean([len(r.split()) for r in refs]))
    out["distinct_hyp_ratio"] = len(set(hyps)) / max(len(hyps), 1)      # low = model collapsing to a few sentences
    return out

def apply_controls(src, control="none", drop=()):
    """src: dict of tensors (parts + attention_mask).  Returns False if the batch must be skipped (shuffle needs B>=2)."""
    import torch
    for p in drop: src[p] = torch.zeros_like(src[p])
    if control == "shuffle":
        B = src["attention_mask"].shape[0]
        if B < 2: return False
        idx = torch.roll(torch.arange(B), 1)
        for k in PARTS + ["attention_mask"]: src[k] = src[k][idx.to(src[k].device)]
    return True

def sha(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f: h.update(f.read(n))
    return h.hexdigest()[:12]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Uni-Sign"); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"]); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8); ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=256); ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"]); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--control", default="none", choices=["none", "shuffle"]); ap.add_argument("--drop", default="")
    ap.add_argument("--tag", default="run"); ap.add_argument("--out", default="results"); ap.add_argument("--non_strict", action="store_true")
    ap.add_argument("--bleurt", default="", help="path to BLEURT-20 dir (optional)")
    a = ap.parse_args()
    drop = [p for p in a.drop.split(",") if p]; assert all(p in PARTS for p in drop), PARTS
    out_abs = os.path.abspath(a.out); ckpt = os.path.abspath(a.ckpt)
    os.chdir(a.repo); sys.path.insert(0, os.getcwd()); os.makedirs(out_abs, exist_ok=True)
    # inference never uses deepspeed/decord, but Uni-Sign imports them at module level -> stub if absent (e.g. Colab CPU)
    from unittest import mock
    import types, importlib.machinery as _im
    class _Stub(types.ModuleType):
        def __getattr__(self, n):
            if n.startswith("__"): raise AttributeError(n)
            return mock.MagicMock()
    for _m in ("deepspeed", "deepspeed.comm", "deepspeed.accelerator", "decord"):
        try: __import__(_m)
        except Exception:
            st = _Stub(_m); st.__spec__ = _im.ModuleSpec(_m, None); st.__path__ = []; sys.modules[_m] = st
            if "." in _m: setattr(sys.modules[_m.rsplit(".", 1)[0]], _m.rsplit(".", 1)[1], st)
    import torch, utils
    from torch.utils.data import DataLoader
    from models import Uni_Sign
    from datasets import S2T_Dataset
    import config as C
    args = utils.get_args_parser().parse_args([])
    args.dataset, args.task, args.rgb_support = "How2Sign", "SLT", False
    args.batch_size, args.max_length, args.seed, args.num_workers = a.batch_size, a.max_length, a.seed, a.num_workers
    utils.set_seed(a.seed); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = (a.dtype == "bf16") or (a.dtype == "auto" and dev.type == "cuda" and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    label = {"train": C.train_label_paths, "val": C.dev_label_paths, "test": C.test_label_paths}[a.split]["How2Sign"]
    ds = S2T_Dataset(path=label, args=args, phase=a.split)
    keys = sorted(ds.list)
    have = [k for k in keys if os.path.exists(os.path.join(ds.pose_dir, ds.raw_data[k]["video_path"].replace(".mp4", ".pkl")))]
    n_no_pose = len(keys) - len(have); keys = have
    if n_no_pose: print(f"WARNING: {n_no_pose} clips have no pose file and are excluded from scoring")
    if a.limit and a.limit < len(keys): keys = [keys[i] for i in np.linspace(0, len(keys) - 1, a.limit).astype(int)]
    ds.list = keys; print(f"split={a.split} clips={len(ds)} dtype={dtype} device={dev} control={a.control} drop={drop}")
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers, collate_fn=ds.collate_fn)
    model = Uni_Sign(args=args)
    sd = torch.load(ckpt, map_location="cpu")["model"]
    ret = model.load_state_dict(sd, strict=not a.non_strict)
    if a.non_strict: print("missing:", len(ret.missing_keys), ret.missing_keys[:5], "unexpected:", len(ret.unexpected_keys), ret.unexpected_keys[:5])
    model.to(dev).to(dtype).eval(); tok = model.mt5_tokenizer
    names, refs, hyps, skipped = [], [], [], 0; t0 = time.time()
    with torch.no_grad():
        for src, tgt in dl:
            for k, v in list(src.items()):
                if isinstance(v, torch.Tensor): src[k] = v.to(dev).to(dtype) if v.is_floating_point() else v.to(dev)
            if not apply_controls(src, a.control, drop): skipped += len(tgt["gt_sentence"]); continue
            out = model(src, tgt); gen = model.generate(out, max_new_tokens=100, num_beams=a.beams)
            hyps += [h.strip() for h in tok.batch_decode(gen, skip_special_tokens=True)]
            refs += list(tgt["gt_sentence"]); names += list(src["name_batch"])
            print(f"{len(hyps)}/{len(ds)}  {time.time()-t0:.0f}s", end="\r")
    secs = time.time() - t0; m = corpus_metrics(refs, hyps)
    try:
        from SLRT_metrics import translation_performance
        bd, rg = translation_performance(refs, hyps); m["UniSign_metric_script"] = {"bleu": bd, "rouge": rg}
    except Exception as e: m["UniSign_metric_script"] = f"unavailable: {type(e).__name__}"
    if a.bleurt:
        from bleurt import score
        s = score.BleurtScorer(a.bleurt).score(references=refs, candidates=hyps); m["BLEURT-20"] = float(np.mean(s))
    from sacrebleu.metrics import BLEU
    meta = dict(n_no_pose_excluded=n_no_pose, tag=a.tag, split=a.split, n_scored=len(hyps), n_skipped=skipped, control=a.control, drop=drop, beams=a.beams,
                dtype=str(dtype), seed=a.seed, max_length=a.max_length, ckpt=os.path.basename(ckpt), ckpt_sha_head=sha(ckpt),
                seconds=round(secs, 1), sec_per_clip=round(secs / max(len(hyps), 1), 3), sacrebleu_note="BLEU-n = corpus BLEU with max n-gram order n, 13a tokenization, case-sensitive")
    json.dump({"meta": meta, "metrics": m}, open(os.path.join(out_abs, f"{a.tag}_metrics.json"), "w"), indent=1)
    with open(os.path.join(out_abs, f"{a.tag}_predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["name", "reference", "prediction", "sentence_bleu"])
        for n, r, h in zip(names, refs, hyps): w.writerow([n, r, h, round(BLEU(effective_order=True).sentence_score(h, [r]).score, 2)])
    print("\n", json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in m.items() if k != "UniSign_metric_script"}, indent=1)); print(meta)

if __name__ == "__main__": main()
