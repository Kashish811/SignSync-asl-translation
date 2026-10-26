"""Unpack How2Sign poses, build label files from the official CSVs, fix Uni-Sign config, verify name matching.
  python prepare.py --hf_dir hf_unisign --csv_dir <folder with how2sign_realigned_{train,val,test}.csv> [--repo Uni-Sign]
Idempotent.  Prints a PASS/FAIL verdict per split.  Writes prepare_report.json."""
import argparse, csv, glob, gzip, json, os, pickle, re, shutil, zipfile
from collections import defaultdict

def merge_and_unzip(hf_dir, out_dir, keep=False):
    parts = sorted(glob.glob(os.path.join(hf_dir, "**", "how2sign_pose_format.zip*"), recursive=True))
    if not parts: print("no how2sign_pose_format.zip* under", hf_dir); return False
    if os.path.isdir(out_dir) and sum(1 for _ in glob.iglob(out_dir + "/**/*.pkl", recursive=True)) > 100:
        print("poses already unpacked in", out_dir); return True
    os.makedirs(out_dir, exist_ok=True); merged = os.path.join(hf_dir, "how2sign_pose_format.merged.zip")
    if len(parts) == 1 and parts[0].endswith(".zip"): merged = parts[0]
    else:
        with open(merged, "wb") as w:
            for p in parts:
                with open(p, "rb") as r: shutil.copyfileobj(r, w, 1 << 24)
    with zipfile.ZipFile(merged) as z: z.extractall(out_dir)
    if not keep and merged.endswith(".merged.zip"): os.remove(merged)
    return True

def find_pose_dirs(root):
    dirs = defaultdict(int)
    for p in glob.iglob(root + "/**/*.pkl", recursive=True): dirs[os.path.dirname(p)] += 1
    return {d: n for d, n in dirs.items() if n >= 50}

def link_splits(root, target):
    """create target/{train,val,test} pointing at the unpacked folders (by folder name), or all -> one flat folder"""
    found = find_pose_dirs(root); os.makedirs(target, exist_ok=True); res = {}
    by_split = {}
    for d in found:
        for s, pat in (("train", r"train"), ("val", r"(^|[^a-z])(val|dev)"), ("test", r"test")):
            if re.search(pat, os.path.basename(d), re.I) or re.search(pat, d.replace(root, ""), re.I): by_split.setdefault(s, d)
    if len(found) == 1 and not by_split: by_split = {s: next(iter(found)) for s in ("train", "val", "test")}
    for s, d in by_split.items():
        link = os.path.join(target, s)
        if os.path.lexists(link): os.remove(link) if os.path.islink(link) else None
        if not os.path.exists(link): os.symlink(os.path.abspath(d), link)
        res[s] = (link, found[d])
    return res, found

def build_labels(csv_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True); out = {}
    for split, fname in (("train", "labels.train"), ("val", "labels.dev"), ("test", "labels.test")):
        c = [p for p in glob.glob(os.path.join(csv_dir, "**", "*.csv"), recursive=True) if "realign" in os.path.basename(p).lower() and re.search(split, os.path.basename(p), re.I)]
        if not c: print("no CSV for", split); continue
        d = {}
        with open(c[0], newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                n, t = r["SENTENCE_NAME"].strip(), r["SENTENCE"].strip()
                if n and t: d[n] = {"name": n, "video_path": n + ".mp4", "text": t}
        with gzip.open(os.path.join(out_dir, fname), "wb") as f: pickle.dump(d, f)
        out[split] = len(d); print(f"{split}: {len(d)} sentences from {os.path.basename(c[0])}")
    return out

def patch_config(repo):
    p = os.path.join(repo, "config.py"); s = open(p).read()
    new = '"How2Sign": "./dataset/How2Sign/pose_format/{}"'
    s2 = s.replace('"How2Sign": "./dataset/WLASL/pose_format"', new)
    s2 = s2.replace('"How2Sign": "",', '"How2Sign": "./data/How2Sign/labels.dev",')
    if s2 != s: open(p, "w").write(s2); print("config.py patched")
    return new in s2

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--hf_dir", default="hf_unisign"); ap.add_argument("--csv_dir", required=True)
    ap.add_argument("--repo", default="Uni-Sign"); ap.add_argument("--keep_zip", action="store_true"); a = ap.parse_args()
    raw = os.path.join(a.repo, "dataset", "How2Sign", "pose_raw"); tgt = os.path.join(a.repo, "dataset", "How2Sign", "pose_format")
    ok = merge_and_unzip(a.hf_dir, raw, a.keep_zip)
    links, found = link_splits(raw, tgt) if ok else ({}, {})
    print("pose folders:", {d: n for d, n in found.items()}); print("split links:", {k: v for k, v in links.items()})
    nl = build_labels(a.csv_dir, os.path.join(a.repo, "data", "How2Sign")); cfg = patch_config(a.repo)
    rep = {"labels": nl, "config_patched": cfg, "splits": {}}
    for split, lf in (("train", "labels.train"), ("val", "labels.dev"), ("test", "labels.test")):
        lp = os.path.join(a.repo, "data", "How2Sign", lf)
        if not os.path.exists(lp) or split not in links: rep["splits"][split] = "SKIPPED"; continue
        d = pickle.load(gzip.open(lp, "rb")); pd = links[split][0]
        miss = [k for k, s in d.items() if not os.path.exists(os.path.join(pd, s["video_path"].replace(".mp4", ".pkl")))]
        frac = len(miss) / max(len(d), 1); verdict = "PASS" if frac < 0.01 else "FAIL"
        rep["splits"][split] = {"labels": len(d), "missing_pose": len(miss), "verdict": verdict, "example_missing": miss[:3],
                                "example_pose_files": sorted(os.listdir(pd))[:3]}
        print(f"[{verdict}] {split}: {len(d)} labels, {len(miss)} without pose")
    json.dump(rep, open("prepare_report.json", "w"), indent=1); print("wrote prepare_report.json  (if any FAIL: send it to Claude)")
