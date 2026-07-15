"""Training loop: python -m coengtada.train [--max-lines N] [--steps N] ..."""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .charset import Vocab
from .data import SiteDataset, load_site_words, sampler_weights
from .lexicon import load_dict_words
from .model import CoengTaDaNet

ROOT = Path(__file__).resolve().parent.parent


def evaluate(model, loader, device, max_batches=None):
    model.eval()
    tp = np.zeros(2)
    fp = np.zeros(2)
    fn = np.zeros(2)
    correct = total = 0
    with torch.inference_mode():
        for b, (x, y, _) in enumerate(loader):
            if max_batches is not None and b >= max_batches:
                break
            pred = model(x.to(device)).argmax(1).cpu()
            for c in (0, 1):
                tp[c] += ((pred == c) & (y == c)).sum().item()
                fp[c] += ((pred == c) & (y != c)).sum().item()
                fn[c] += ((pred != c) & (y == c)).sum().item()
            correct += (pred == y).sum().item()
            total += len(y)
    prec = tp / np.maximum(1, tp + fp)
    rec = tp / np.maximum(1, tp + fn)
    f1 = 2 * prec * rec / np.maximum(1e-9, prec + rec)
    return {
        "accuracy": correct / max(1, total),
        "macro_f1": float(f1.mean()),
        "ta": {"precision": prec[0], "recall": rec[0], "f1": f1[0]},
        "da": {"precision": prec[1], "recall": rec[1], "f1": f1[1]},
        "n": total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/processed"))
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--dict", default=str(ROOT / "khmerdict.txt"))
    ap.add_argument("--max-lines", type=int, default=None, help="cap train lines (quick runs)")
    ap.add_argument("--steps", type=int, default=None, help="cap optimizer steps")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data)
    art_dir = Path(args.artifacts)
    vocab = Vocab.load(art_dir / "vocab.json")
    with open(art_dir / "holdout_words.json", encoding="utf-8") as f:
        holdout_keys = set(json.load(f))
    pseudo_words = load_site_words(load_dict_words(args.dict), exclude_keys=holdout_keys)

    train_ds = SiteDataset([data_dir / "train.jsonl"], vocab, train=True,
                           pseudo_words=pseudo_words, max_lines=args.max_lines)
    val_ds = SiteDataset([data_dir / "val.jsonl"], vocab)
    print(f"train sites: {len(train_ds):,}  val sites: {len(val_ds):,}  vocab: {vocab.size}")

    sampler = WeightedRandomSampler(sampler_weights(train_ds.labels), num_samples=len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=512, num_workers=0)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CoengTaDaNet(vocab.size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device}  params: {n_params:,}")

    total_steps = args.steps or args.epochs * math.ceil(len(train_ds) / args.batch_size)

    def lr_lambda(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        t = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_f1 = -1.0
    bad_evals = 0
    step = 0
    art_dir.mkdir(parents=True, exist_ok=True)
    history = []
    stop = False

    for epoch in range(args.epochs):
        if stop:
            break
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}", unit="step")
        for x, y, w in pbar:
            model.train()
            x, y, w = x.to(device), y.to(device), w.to(device, dtype=torch.float32)
            logits = model(x)
            loss = (F.cross_entropy(logits, y, reduction="none", label_smoothing=0.05) * w).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{sched.get_last_lr()[0]:.1e}")
            if step % args.eval_every == 0 or step == total_steps:
                m = evaluate(model, val_loader, device, max_batches=100)
                history.append({"step": step, **m})
                tqdm.write(f"step {step}: val acc {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}  "
                           f"DA recall {m['da']['recall']:.4f}")
                if m["macro_f1"] > best_f1:
                    best_f1 = m["macro_f1"]
                    bad_evals = 0
                    torch.save(model.state_dict(), art_dir / "model.pt")
                    with open(art_dir / "config.json", "w") as f:
                        json.dump({"vocab_size": vocab.size, "emb_dim": 48, "hidden": 96,
                                   "window": 64, "best_val": m, "step": step}, f, indent=1)
                else:
                    bad_evals += 1
                    if bad_evals >= args.patience:
                        tqdm.write(f"early stop at step {step} (best macro-F1 {best_f1:.4f})")
                        stop = True
                        break
            if step >= total_steps:
                stop = True
                break

    with open(art_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=1)
    print(f"done. best val macro-F1: {best_f1:.4f}  -> {art_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
