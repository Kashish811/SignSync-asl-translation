"""
signsync_pipeline_template.py
=============================


What it is:
     body/hand/face keypoints ->
    four skeleton graphs -> ST-GCN encoders -> mT5-base decoder -> English text.
    It shows every stage a student would have to build:

        STAGE 1  use the ready-made RTMPose poses (no extraction step)
        STAGE 2  data set + loader               (Dataset, collate)
        STAGE 3  model: ST-GCN encoder + mT5     (PoseEncoder, SignToText)
        STAGE 4  training loop                   (train)
        STAGE 5  small-scale hyper-parameter search   (tune)
        STAGE 6  decoding-only tuning (beams, repetition penalty)   (decode-tune)
        STAGE 7  final evaluation on the test split  (eval)


Quick start
    pip install torch transformers sentencepiece sacrebleu numpy
    python signsync_pipeline_template.py smoke            # tiny end-to-end test, CPU

    # real data layout (you create it). Poses are NOT extracted here: download the
    # already-extracted RTMPose keypoints that come with the dataset release, and save
    # one array per clip (convert them if they are stored in another format):
    #   data/manifest.csv            columns: clip_id,text,split   (split = train|val|test)
    #   data/poses/<clip_id>.npy     float32 array (frames, 133, 3) = x, y, confidence
    python signsync_pipeline_template.py tune --data_dir data --out_dir runs/exp1
    python signsync_pipeline_template.py train --data_dir data --out_dir runs/exp1
    python signsync_pipeline_template.py decode-tune --data_dir data --out_dir runs/exp1
    python signsync_pipeline_template.py eval --data_dir data --out_dir runs/exp1

Paper recipe for reference (Li et al., Table 2): AdamW, base learning rate 3e-4 with
cosine decay, weight decay 1e-4, 20 epochs for downstream fine-tuning. The defaults
below follow that, but they are a starting point, not a result.
"""

# Libraries: numpy/torch build the model, sacrebleu computes the BLEU score.
import argparse
import csv
import itertools
import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np
import sacrebleu
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset


# =============================================================================
# 0. CONFIGURATION
# =============================================================================
# All settings in one place (change them here or from the command line).
@dataclass
class Cfg:
    data_dir: str = "data"
    out_dir: str = "runs/exp1"
    model_name: str = "google/mt5-base"      # the text decoder (582 M parameters)
    tiny: bool = False                       # True = tiny random model for smoke tests
    prompt: str = "Translate sign language video to English:"
    max_frames: int = 256                    # longer clips are subsampled
    gcn_channels: tuple = (64, 128, 256)     # ST-GCN widths; last one = per-part feature size
    temporal_kernel: int = 5                 # frames each temporal convolution looks across
    epochs: int = 20
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    warmup_frac: float = 0.05
    max_new_tokens: int = 64
    seed: int = 0
    num_workers: int = 0


# Fix the random numbers so a run can be repeated.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 1. POSE LAYOUT  (COCO-WholeBody, 133 keypoints per frame, as given by RTMPose)
#    0-16 body | 17-22 feet | 23-90 face (68) | 91-111 left hand | 112-132 right hand
#    We keep 69 of them: 9 body + 21 + 21 hand + 18 face. WHICH body and face points
#    to keep is a design choice; the choice below is illustrative. The original authors
#    define their own subset.
# =============================================================================
# Which of the 133 keypoints we keep for each part.
BODY_IDX = [0, 5, 6, 7, 8, 9, 10, 11, 12]            # nose, shoulders, elbows, wrists, hips
LHAND_IDX = list(range(91, 112))
RHAND_IDX = list(range(112, 133))
FACE_IDX = [23 + i for i in list(range(17, 27)) + list(range(48, 56))]   # 10 brow + 8 mouth


# Says which hand joints are joined by a bone.
def hand_edges():
    """Wrist (0) -> each finger base, then along each finger: the usual 21-joint hand."""
    edges = []
    for finger in range(5):
        base = 1 + 4 * finger
        edges.append((0, base))
        edges += [(base + k, base + k + 1) for k in range(3)]
    return edges


# Bones of the body and face graphs (pairs of joint numbers).
BODY_EDGES = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 8)]
FACE_EDGES = ([(i, i + 1) for i in range(0, 4)] + [(i, i + 1) for i in range(5, 9)] + [(4, 5)]
              + [(i, i + 1) for i in range(10, 17)])


# Turns the bones into a table of which joints are neighbours.
def build_adjacency(num_joints, edges):
    """Normalised adjacency D^-1/2 (A + I) D^-1/2: how much each joint listens to its neighbours."""
    a = np.eye(num_joints, dtype=np.float32)
    for i, j in edges:
        a[i, j] = a[j, i] = 1.0
    d = np.diag(1.0 / np.sqrt(a.sum(1)))
    return torch.tensor(d @ a @ d, dtype=torch.float32)


# Centre and rescale joints so every clip has the same size and position.
def _norm(xy, center, scale):
    return (xy - center[:, None, :]) / (scale[:, None, None] + 1e-6)


# Cuts each pose sequence into four parts: body, left hand, right hand, face.
def split_parts(kp):
    """kp: (T, 133, 3) -> dict of four arrays (T, V, 3), each centred and scaled by its own part."""
    out = {}
    body = kp[:, BODY_IDX]
    center = (body[:, 1, :2] + body[:, 2, :2]) / 2                      # between the shoulders
    scale = np.linalg.norm(body[:, 1, :2] - body[:, 2, :2], axis=-1)    # shoulder width
    out["body"] = np.concatenate([_norm(body[..., :2], center, scale), body[..., 2:]], -1)
    for name, idx, root in (("lhand", LHAND_IDX, 0), ("rhand", RHAND_IDX, 0), ("face", FACE_IDX, None)):
        p = kp[:, idx]
        center = p[:, root, :2] if root is not None else p[..., :2].mean(1)
        scale = np.abs(p[..., :2] - center[:, None, :]).max(axis=(1, 2))
        out[name] = np.concatenate([_norm(p[..., :2], center, scale), p[..., 2:]], -1)
    return {k: v.astype(np.float32) for k, v in out.items()}


# =============================================================================
# STAGE 1. POSES: use the already-extracted RTMPose keypoints (no extraction code needed)
#   Put one (frames, 133, 3) .npy file per clip in <data_dir>/poses/<clip_id>.npy
# =============================================================================


# =============================================================================
# STAGE 2. DATASET
# =============================================================================
# Reads the list of clips (id, sentence, split) from manifest.csv.
def read_manifest(data_dir):
    with open(os.path.join(data_dir, "manifest.csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# Loads one clip's poses and its English sentence.
class PoseTextDataset(Dataset):
    def __init__(self, data_dir, split, max_frames, train):
        self.dir, self.max_frames, self.train = data_dir, max_frames, train
        self.rows = [r for r in read_manifest(data_dir) if r["split"] == split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        kp = np.load(os.path.join(self.dir, "poses", row["clip_id"] + ".npy")).astype(np.float32)
        t = len(kp)
        if t > self.max_frames:                       # train: random frames; eval: evenly spaced
            if self.train:
                keep = np.sort(np.random.choice(t, self.max_frames, replace=False))
            else:
                keep = np.linspace(0, t - 1, self.max_frames).astype(int)
            kp = kp[keep]
        return split_parts(kp), row["text"]


# Pads clips to the same length so they can be stacked into a batch.
def collate(batch):
    """Pad every clip in the batch to the same number of frames and build a frame mask."""
    t_max = max(parts["body"].shape[0] for parts, _ in batch)
    out, mask = {}, torch.zeros(len(batch), t_max, dtype=torch.long)
    for name in batch[0][0]:
        v = batch[0][0][name].shape[1]
        arr = np.zeros((len(batch), t_max, v, 3), np.float32)
        for b, (parts, _) in enumerate(batch):
            arr[b, : len(parts[name])] = parts[name]
        out[name] = torch.from_numpy(arr)
    for b, (parts, _) in enumerate(batch):
        mask[b, : len(parts["body"])] = 1
    return out, mask, [text for _, text in batch]


# Makes tiny fake data so the code can be tested in seconds.
def make_synthetic(data_dir, n_train=48, n_val=12, n_test=12, seed=0):
    """Tiny fake dataset so the pipeline can be tested without any real data.
    Each of 4 sentences has its own pose pattern, so a working model can learn it."""
    rng = np.random.RandomState(seed)
    sentences = ["open the box", "close the door", "turn the page", "wash your hands"]
    patterns = [rng.randn(133, 3).astype(np.float32) for _ in sentences]
    os.makedirs(os.path.join(data_dir, "poses"), exist_ok=True)
    rows, n = [], 0
    for split, count in (("train", n_train), ("val", n_val), ("test", n_test)):
        for _ in range(count):
            k = rng.randint(len(sentences))
            frames = rng.randint(12, 24)
            kp = patterns[k][None] * 0.5 + 0.1 * rng.randn(frames, 133, 3).astype(np.float32)
            kp[..., 2] = 1.0
            clip = f"clip{n:04d}"
            np.save(os.path.join(data_dir, "poses", clip + ".npy"), kp)
            rows.append({"clip_id": clip, "text": sentences[k], "split": split})
            n += 1
    with open(os.path.join(data_dir, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "text", "split"])
        w.writeheader()
        w.writerows(rows)


# =============================================================================
# TOKENIZERS: a real mT5 tokenizer, or a tiny word-level one for smoke tests
# =============================================================================
# Real mT5 tokenizer: turns text into numbers and back.
class HFTokenizer:
    def __init__(self, name):
        from transformers import AutoTokenizer
        self.t = AutoTokenizer.from_pretrained(name)
        self.pad_id, self.eos_id, self.vocab_size = self.t.pad_token_id, self.t.eos_token_id, len(self.t)

    def encode(self, text, add_eos=True):
        return self.t.encode(text, add_special_tokens=add_eos)

    def batch_encode(self, texts, max_len=64):
        return self.t(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").input_ids

    def decode_batch(self, ids):
        return self.t.batch_decode(ids, skip_special_tokens=True)


# Tiny word-level tokenizer, used only for the quick smoke test.
class WordTokenizer:
    """ids: 0 pad, 1 eos, 2 unk, then one id per word."""

    def __init__(self, vocab):
        self.vocab = vocab
        self.inv = {i: w for w, i in vocab.items()}
        self.pad_id, self.eos_id, self.vocab_size = 0, 1, len(vocab) + 3

    @staticmethod
    def build(texts):
        words = sorted({w for t in texts for w in t.lower().split()})
        return WordTokenizer({w: i + 3 for i, w in enumerate(words)})

    def encode(self, text, add_eos=True):
        ids = [self.vocab.get(w, 2) for w in text.lower().split()]
        return ids + [self.eos_id] if add_eos else ids

    def batch_encode(self, texts, max_len=64):
        seqs = [self.encode(t)[:max_len] for t in texts]
        out = torch.zeros(len(seqs), max(len(s) for s in seqs), dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = torch.tensor(s)
        return out

    def decode_batch(self, ids):
        res = []
        for row in ids.tolist():
            words = []
            for i in row:
                if i == self.eos_id:
                    break
                if i > 2:
                    words.append(self.inv.get(i, ""))
            res.append(" ".join(words))
        return res


# Picks the real tokenizer, or the tiny one in smoke-test mode.
def get_tokenizer(cfg):
    if not cfg.tiny:
        return HFTokenizer(cfg.model_name)
    path = os.path.join(cfg.out_dir, "vocab.json")
    os.makedirs(cfg.out_dir, exist_ok=True)
    if os.path.exists(path):
        return WordTokenizer(json.load(open(path)))
    tok = WordTokenizer.build([r["text"] for r in read_manifest(cfg.data_dir)] + [cfg.prompt])
    json.dump(tok.vocab, open(path, "w"))
    return tok


# =============================================================================
# STAGE 3. MODEL
# =============================================================================
# One graph layer: mixes neighbouring joints, then neighbouring frames.
class STGCNBlock(nn.Module):
    """One spatial-temporal graph convolution block.
    spatial : every joint mixes its features with its neighbours' (through adjacency A)
    temporal: every joint mixes its features across neighbouring frames (1-D conv in time)
    A residual connection keeps training stable."""

    def __init__(self, in_c, out_c, t_kernel, dropout):
        super().__init__()
        self.gcn = nn.Conv2d(in_c, out_c, 1)                              # feature transform
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_c), nn.ReLU(),
            nn.Conv2d(out_c, out_c, (t_kernel, 1), padding=(t_kernel // 2, 0)),
            nn.BatchNorm2d(out_c), nn.Dropout(dropout),
        )
        self.res = nn.Identity() if in_c == out_c else nn.Conv2d(in_c, out_c, 1)
        self.act = nn.ReLU()

    # Spatial step, then temporal step, plus a shortcut connection.
    def forward(self, x, adjacency):                                      # x: (B, C, T, V)
        y = torch.einsum("bctv,vw->bctw", self.gcn(x), adjacency)         # spatial mixing
        return self.act(self.tcn(y) + self.res(x))                        # temporal mixing


# Stacks graph layers to turn one body part's poses into features.
class PartEncoder(nn.Module):
    """A stack of ST-GCN blocks for ONE body part. Output: (B, T, channels[-1])."""

    def __init__(self, adjacency, channels, t_kernel, dropout, in_c=3):
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.bn = nn.BatchNorm2d(in_c)
        sizes = [in_c, *channels]
        self.blocks = nn.ModuleList(
            STGCNBlock(sizes[i], sizes[i + 1], t_kernel, dropout) for i in range(len(channels)))

    def forward(self, x):                                                 # x: (B, T, V, 3)
        x = self.bn(x.permute(0, 3, 1, 2))                                # -> (B, 3, T, V)
        for block in self.blocks:
            x = block(x, self.adjacency)
        return x.mean(-1).transpose(1, 2)                                 # average over joints


# Runs all four parts and combines them into one feature per frame.
class PoseEncoder(nn.Module):
    """Four part graphs -> one feature vector per frame, sized for the text model.
    * the two hands share ONE encoder (same weights)
    * the body feature is added into the hand and face branches
    * the four 256-d vectors are joined (4 x 256 = 1024) and projected to d_model."""

    def __init__(self, d_model, channels, t_kernel, dropout):
        super().__init__()
        self.body = PartEncoder(build_adjacency(len(BODY_IDX), BODY_EDGES), channels, t_kernel, dropout)
        self.hand = PartEncoder(build_adjacency(21, hand_edges()), channels, t_kernel, dropout)
        self.face = PartEncoder(build_adjacency(len(FACE_IDX), FACE_EDGES), channels, t_kernel, dropout)
        c = channels[-1]
        self.body_to_hand, self.body_to_face = nn.Linear(c, c), nn.Linear(c, c)
        self.project = nn.Linear(4 * c, d_model)

    # Encode each part; hands and face also receive the body features.
    def forward(self, parts):
        body = self.body(parts["body"])
        left = self.hand(parts["lhand"]) + self.body_to_hand(body)
        right = self.hand(parts["rhand"]) + self.body_to_hand(body)
        face = self.face(parts["face"]) + self.body_to_face(body)
        return self.project(torch.cat([body, left, right, face], dim=-1))  # (B, T, d_model)


# Full model: pose features + a text prompt go into mT5, which writes English.
class SignToText(nn.Module):
    """Pose encoder + mT5. The text prompt's embeddings and the pose features are
    concatenated and fed to the mT5 encoder; the mT5 decoder writes the sentence."""

    def __init__(self, cfg, tok):
        super().__init__()
        from transformers import MT5Config, MT5ForConditionalGeneration
        if cfg.tiny:
            conf = MT5Config(vocab_size=tok.vocab_size, d_model=64, d_kv=16, d_ff=128, num_layers=2,
                             num_decoder_layers=2, num_heads=4, dropout_rate=cfg.dropout,
                             pad_token_id=tok.pad_id, eos_token_id=tok.eos_id, decoder_start_token_id=tok.pad_id)
            self.t5 = MT5ForConditionalGeneration(conf)
        else:
            self.t5 = MT5ForConditionalGeneration.from_pretrained(cfg.model_name, dropout_rate=cfg.dropout)
        self.tok = tok
        self.encoder = PoseEncoder(self.t5.config.d_model, cfg.gcn_channels, cfg.temporal_kernel, cfg.dropout)
        self.register_buffer("prompt_ids", torch.tensor(tok.encode(cfg.prompt, add_eos=False)))

    # Puts the prompt and the pose features together as the encoder input.
    def _inputs(self, parts, mask):
        pose = self.encoder(parts)
        prompt = self.t5.get_input_embeddings()(self.prompt_ids)[None].expand(pose.size(0), -1, -1)
        ones = torch.ones(pose.size(0), prompt.size(1), dtype=mask.dtype, device=mask.device)
        return torch.cat([prompt, pose], 1), torch.cat([ones, mask], 1)

    # Training pass: returns the loss for the correct English sentences.
    def forward(self, parts, mask, texts, max_len=64):
        embeds, attn = self._inputs(parts, mask)
        labels = self.tok.batch_encode(texts, max_len).to(embeds.device)
        labels[labels == self.tok.pad_id] = -100                          # ignore padding in the loss
        return self.t5(inputs_embeds=embeds, attention_mask=attn, labels=labels).loss

    @torch.no_grad()
    # Writes an English sentence for each clip.
    def generate(self, parts, mask, **gen_kwargs):
        embeds, attn = self._inputs(parts, mask)
        return self.t5.generate(inputs_embeds=embeds, attention_mask=attn, **gen_kwargs)


# Counts the model's learnable numbers (parameters).
def count_parameters(model):
    enc = sum(p.numel() for p in model.encoder.parameters())
    total = sum(p.numel() for p in model.parameters())
    return {"graph_encoder": enc, "text_model": total - enc, "total": total}


# =============================================================================
# METRICS
# =============================================================================
# ROUGE-L score: how well the word order matches the reference.
def rouge_l(hyp, ref):
    """ROUGE-L F1: longest common subsequence of words, in order, gaps allowed."""
    h, r = hyp.lower().split(), ref.lower().split()
    if not h or not r:
        return 0.0
    dp = [[0] * (len(r) + 1) for _ in range(len(h) + 1)]
    for i in range(len(h)):
        for j in range(len(r)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if h[i] == r[j] else max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    p, rec = lcs / len(h), lcs / len(r)
    return 2 * p * rec / (p + rec)


# Share of outputs stuck repeating the same phrase.
def loop_rate(hyps, n=3, times=3):
    """Share of outputs where some n-gram of words repeats `times` or more: a repetition loop."""
    def loops(text):
        w = text.lower().split()
        grams = Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
        return bool(grams) and max(grams.values()) >= times
    return sum(loops(h) for h in hyps) / max(len(hyps), 1)


# Moves the data to the GPU (or the CPU).
def to_device(parts, device):
    return {k: v.to(device) for k, v in parts.items()}


# Translates a dataset and scores it with BLEU-4 and ROUGE-L.
@torch.no_grad()
def evaluate(model, dataset, device, gen_kwargs, batch_size=8):
    model.eval()
    hyps, refs = [], []
    for parts, mask, texts in DataLoader(dataset, batch_size, collate_fn=collate):
        ids = model.generate(to_device(parts, device), mask.to(device), **gen_kwargs)
        hyps += model.tok.decode_batch(ids.cpu())
        refs += texts
    metrics = {
        "bleu4": sacrebleu.corpus_bleu(hyps, [refs]).score,
        "rougeL": 100 * float(np.mean([rouge_l(h, r) for h, r in zip(hyps, refs)])),
        "loop_rate": loop_rate(hyps),
    }
    return metrics, hyps, refs


# =============================================================================
# STAGE 4. TRAINING
# =============================================================================
# Training loop: learn, check on validation each epoch, save the best model.
def train_model(cfg, tok, train_ds, val_ds, epochs, device, save_path=None, log=print):
    """Fine-tune the whole model. Validate each epoch with fast greedy decoding.
    Returns (best validation BLEU-4, history, model)."""
    set_seed(cfg.seed)
    model = SignToText(cfg, tok).to(device)
    log(f"parameters: {count_parameters(model)}")
    loader = DataLoader(train_ds, cfg.batch_size, shuffle=True, collate_fn=collate, num_workers=cfg.num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = max(1, epochs * math.ceil(len(loader) / cfg.grad_accum))
    warmup = max(1, int(cfg.warmup_frac * total))
    schedule = torch.optim.lr_scheduler.LambdaLR(                         # warm-up, then cosine decay
        optimizer, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))
    use_bf16 = device.type == "cuda"                                      # bf16 is safer than fp16 for mT5
    best, history, step = -1.0, [], 0
    for epoch in range(epochs):
        model.train()
        t0, losses = time.time(), []
        for i, (parts, mask, texts) in enumerate(loader):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(to_device(parts, device), mask.to(device), texts)
            (loss / cfg.grad_accum).backward()
            losses.append(loss.item())
            if (i + 1) % cfg.grad_accum == 0 or i + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); schedule.step(); optimizer.zero_grad(); step += 1
        metrics, _, _ = evaluate(model, val_ds, device, dict(num_beams=1, max_new_tokens=cfg.max_new_tokens))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **metrics})
        log(f"epoch {epoch + 1}/{epochs}  loss {np.mean(losses):.3f}  val BLEU-4 {metrics['bleu4']:.2f}  ({time.time() - t0:.0f}s)")
        if metrics["bleu4"] > best:
            best = metrics["bleu4"]
            if save_path:
                torch.save(model.state_dict(), save_path)
    return best, history, model


# =============================================================================
# STAGE 5. SMALL-SCALE HYPER-PARAMETER SEARCH (random search on a data subset)
# =============================================================================
# The values the small search is allowed to try.
SEARCH_SPACE = {
    "lr": [1e-4, 3e-4, 1e-3],
    "weight_decay": [0.0, 1e-4, 1e-2],
    "dropout": [0.0, 0.1, 0.3],
    "batch_size": [4, 8],
}


# Picks a small random part of the data for fast trials.
def subset(ds, n, seed):
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return Subset(ds, idx[: min(n, len(ds))])


# Tries random settings briefly and keeps the best-scoring ones.
def random_search(cfg, tok, train_ds, val_ds, device, trials=6, epochs=3, n_train=500, n_val=100):
    """Few trials, few epochs, small subsets: cheap, and only a rough guide.
    Pick the settings with the best validation BLEU-4, then retrain on all the data."""
    rng, results = random.Random(cfg.seed), []
    tr, va = subset(train_ds, n_train, cfg.seed), subset(val_ds, n_val, cfg.seed)
    for t in range(trials):
        hp = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
        print(f"trial {t + 1}/{trials}: {hp}")
        best, _, _ = train_model(replace(cfg, **hp), tok, tr, va, epochs, device, log=lambda *_: None)
        results.append({"trial": t + 1, **hp, "val_bleu4": best})
        print(f"   val BLEU-4 {best:.2f}")
    return max(results, key=lambda r: r["val_bleu4"]), results


# =============================================================================
# STAGE 6. DECODING-ONLY TUNING (no retraining: only how the sentence is generated)
#   num_beams            candidate sentences kept at every step
#   repetition_penalty   >1 makes words that were already produced less likely
#   no_repeat_ngram_size forbids repeating any n-gram of that length (0 = off)
# =============================================================================
# Tries different ways of generating text with the trained model (no retraining).
def tune_decoding(cfg, model, val_ds, device, beams=(1, 4, 8), penalties=(1.0, 1.2, 1.5), no_repeat=(0, 3)):
    rows = []
    for nb, rp, nr in itertools.product(beams, penalties, no_repeat):
        kw = dict(num_beams=nb, repetition_penalty=rp, no_repeat_ngram_size=nr, max_new_tokens=cfg.max_new_tokens)
        metrics, _, _ = evaluate(model, val_ds, device, kw)
        rows.append({"num_beams": nb, "repetition_penalty": rp, "no_repeat_ngram_size": nr, **metrics})
        print(f"beams {nb}  rep.penalty {rp}  no-repeat {nr}  ->  BLEU-4 {metrics['bleu4']:.2f}  loops {100 * metrics['loop_rate']:.0f}%")
    return max(rows, key=lambda r: r["bleu4"]), rows


# =============================================================================
# COMMAND LINE
# =============================================================================
# Builds the train, validation and test sets.
def load_data(cfg):
    train = PoseTextDataset(cfg.data_dir, "train", cfg.max_frames, train=True)
    val = PoseTextDataset(cfg.data_dir, "val", cfg.max_frames, train=False)
    test = PoseTextDataset(cfg.data_dir, "test", cfg.max_frames, train=False)
    return train, val, test


# Uses the best settings found by the search, if there are any.
def load_hparams(cfg):
    path = os.path.join(cfg.out_dir, "best_hparams.json")
    if os.path.exists(path):
        hp = json.load(open(path))
        print("using tuned hyper-parameters:", hp)
        return replace(cfg, **{k: hp[k] for k in SEARCH_SPACE})
    return cfg


# Loads the best saved model.
def load_trained(cfg, tok, device):
    model = SignToText(cfg, tok).to(device)
    model.load_state_dict(torch.load(os.path.join(cfg.out_dir, "best.pt"), map_location=device))
    return model


# Reads the command (smoke, tune, train, decode-tune, eval) and runs that stage.
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["smoke", "tune", "train", "decode-tune", "eval"])
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="runs/exp1")
    ap.add_argument("--tiny", action="store_true", help="tiny random model and word tokenizer (no downloads)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--trial_epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    cfg = Cfg(data_dir=args.data_dir, out_dir=args.out_dir, tiny=args.tiny or args.command == "smoke",
              epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    # smoke: runs every stage on tiny fake data to prove the code works.
    if args.command == "smoke":                                            # whole pipeline, tiny and fast
        cfg = replace(cfg, data_dir=os.path.join(cfg.out_dir, "synthetic"), epochs=6, batch_size=8, lr=1e-3,
                      gcn_channels=(16, 32), max_frames=32)
        make_synthetic(cfg.data_dir)
        tok = get_tokenizer(cfg)
        train, val, test = load_data(cfg)
        print("\n--- STAGE 5: hyper-parameter search (tiny) ---")
        best_hp, rows = random_search(cfg, tok, train, val, device, trials=2, epochs=2, n_train=24, n_val=8)
        json.dump({"best": best_hp, "all": rows}, open(os.path.join(cfg.out_dir, "tuning_results.json"), "w"), indent=2)
        cfg = replace(cfg, **{k: best_hp[k] for k in SEARCH_SPACE})
        print("\n--- STAGE 4: training ---")
        _, history, model = train_model(cfg, tok, train, val, cfg.epochs, device, os.path.join(cfg.out_dir, "best.pt"))
        print("\n--- STAGE 6: decoding tuning ---")
        best_dec, _ = tune_decoding(cfg, model, val, device, beams=(1, 2), penalties=(1.0, 1.3), no_repeat=(0, 3))
        print("\n--- STAGE 7: test evaluation ---")
        kw = {k: best_dec[k] for k in ("num_beams", "repetition_penalty", "no_repeat_ngram_size")}
        metrics, hyps, refs = evaluate(model, test, device, dict(kw, max_new_tokens=cfg.max_new_tokens))
        print("test:", {k: round(v, 3) for k, v in metrics.items()})
        print("example:", refs[0], "->", hyps[0])
        print("\nsmoke test finished. These numbers say nothing about real accuracy: the data is synthetic.")
        return

    tok = get_tokenizer(cfg)
    train, val, test = load_data(cfg)

    # tune: small random search for good training settings.
    if args.command == "tune":
        best_hp, rows = random_search(cfg, tok, train, val, device, args.trials, args.trial_epochs)
        json.dump(best_hp, open(os.path.join(cfg.out_dir, "best_hparams.json"), "w"), indent=2)
        json.dump(rows, open(os.path.join(cfg.out_dir, "tuning_results.json"), "w"), indent=2)
        print("best:", best_hp)

    # train: fine-tune the model with the best settings.
    elif args.command == "train":
        cfg = load_hparams(cfg)
        best, history, _ = train_model(cfg, tok, train, val, cfg.epochs, device, os.path.join(cfg.out_dir, "best.pt"))
        json.dump(history, open(os.path.join(cfg.out_dir, "train_history.json"), "w"), indent=2)
        print("best validation BLEU-4:", round(best, 2))

    # decode-tune: pick the best beams / repetition penalty on validation data.
    elif args.command == "decode-tune":
        model = load_trained(cfg, tok, device)
        best_dec, rows = tune_decoding(cfg, model, val, device)
        json.dump(best_dec, open(os.path.join(cfg.out_dir, "best_decoding.json"), "w"), indent=2)
        json.dump(rows, open(os.path.join(cfg.out_dir, "decoding_results.json"), "w"), indent=2)
        print("best decoding:", best_dec)

    # eval: final score on the test split.
    elif args.command == "eval":                                           # run ONCE, on the test split
        model = load_trained(cfg, tok, device)
        dec = json.load(open(os.path.join(cfg.out_dir, "best_decoding.json")))
        kw = {k: dec[k] for k in ("num_beams", "repetition_penalty", "no_repeat_ngram_size")}
        metrics, hyps, refs = evaluate(model, test, device, dict(kw, max_new_tokens=cfg.max_new_tokens))
        json.dump({"decoding": kw, "metrics": metrics}, open(os.path.join(cfg.out_dir, "results.json"), "w"), indent=2)
        with open(os.path.join(cfg.out_dir, "test_predictions.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["reference", "prediction"])
            w.writerows(zip(refs, hyps))
        print("test:", {k: round(v, 3) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
