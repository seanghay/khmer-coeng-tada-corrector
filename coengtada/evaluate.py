"""Evaluation: corpus test split, dictionary holdout, ambiguous-70, baselines.

python -m coengtada.evaluate
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .charset import DA, TA, Vocab, norm
from .data import LABELS, SEPARATORS, SiteDataset, load_site_words
from .infer import Corrector
from .lexicon import WEIGHT_DICT, Lexicon, load_dict_words
from .model import CoengTaDaNet

ROOT = Path(__file__).resolve().parent.parent


def predict_dataset(model, ds, device):
    loader = DataLoader(ds, batch_size=512)
    preds, labels, weights = [], [], []
    with torch.inference_mode():
        for x, y, w in loader:
            preds.append(model(x.to(device)).argmax(1).cpu().numpy())
            labels.append(y.numpy())
            weights.append(w.numpy())
    if not preds:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    return np.concatenate(preds), np.concatenate(labels), np.concatenate(weights)


def metrics_dict(preds, labels):
    out = {"n": int(len(labels))}
    if not len(labels):
        return out
    out["accuracy"] = float((preds == labels).mean())
    for c, name in ((0, "ta"), (1, "da")):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        out[name] = {"precision": prec, "recall": rec,
                     "f1": 2 * prec * rec / max(1e-9, prec + rec), "support": tp + fn}
    out["macro_f1"] = (out["ta"]["f1"] + out["da"]["f1"]) / 2
    return out


def synthetic_holdout_examples(missing: dict[str, dict], dict_words, rng, per_word=20):
    """Pseudo-sentence contexts for held-out words never seen in the corpus."""
    pool = [w for w, _ in dict_words]
    examples = []  # (text, pos, label)
    for key, info in missing.items():
        for _ in range(per_word):
            others = rng.choices(pool, k=rng.randint(2, 6))
            slot = rng.randint(0, len(others))
            parts = others[:slot] + [key] + others[slot:]
            seps = [rng.choice(SEPARATORS) for _ in range(len(parts) - 1)] + [""]
            text, off = "", 0
            for i, (p, sep) in enumerate(zip(parts, seps)):
                if i == slot:
                    off = len(text)
                text += p + sep
            for idx, cons in info["sites"].items():
                examples.append((norm(text), off + int(idx), LABELS[cons]))
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/processed"))
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--dict", default=str(ROOT / "khmerdict.txt"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_dir, art_dir = Path(args.data), Path(args.artifacts)
    vocab = Vocab.load(art_dir / "vocab.json")
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    model = CoengTaDaNet(vocab.size).to(device)
    model.load_state_dict(torch.load(art_dir / "model.pt", map_location=device, weights_only=True))
    model.eval()

    words = load_dict_words(args.dict)
    lex = Lexicon(words)
    with open(art_dir / "holdout_words.json", encoding="utf-8") as f:
        holdout_words = json.load(f)
    report = {}

    # ---- 1. Corpus test split, stratified by weight bucket + ambiguity ----
    test_ds = SiteDataset([data_dir / "test.jsonl"], vocab)
    preds, labels, weights = predict_dataset(model, test_ds, device)
    report["test"] = metrics_dict(preds, labels)
    strong = weights >= WEIGHT_DICT
    report["test_dict_labeled"] = metrics_dict(preds[strong], labels[strong])
    report["test_weak_labeled"] = metrics_dict(preds[~strong], labels[~strong])

    # Dictionary-lookup + majority baselines on the same split.
    lookup_preds, ambig_rows = [], []
    row = 0
    with open(data_dir / "test.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            for idx, _, _ in d["sites"]:
                cons, _ = lex.match_site(d["text"], idx)
                lookup_preds.append(LABELS[sorted(cons)[0]] if cons and len(cons) == 1 else 0)
                if cons and len(cons) > 1:
                    ambig_rows.append((row, d["text"], idx))
                row += 1
    lookup_preds = np.asarray(lookup_preds)
    report["baseline_majority"] = metrics_dict(np.zeros_like(labels), labels)
    report["baseline_dict_lookup"] = metrics_dict(lookup_preds, labels)

    # ---- 2. Ambiguous-70 sites: correct if prediction is in the valid set ----
    ambig_ok, dist = 0, defaultdict(Counter)
    for row, text, idx in ambig_rows:
        _, keys = lex.match_site(text, idx)
        pred_char = TA if preds[row] == 0 else DA
        ambig_ok += 1  # both spellings valid by definition
        if keys:
            dist[keys[0]][pred_char] += 1
    report["ambiguous"] = {
        "n_sites": len(ambig_rows),
        "valid_rate": 1.0 if ambig_rows else None,
        "prediction_distribution": {k: dict(v) for k, v in sorted(dist.items())},
    }

    # ---- 3. Dictionary holdout: corpus contexts + synthetic for missing ----
    holdout_ds = SiteDataset([data_dir / "holdout.jsonl"], vocab)
    h_preds, h_labels, _ = predict_dataset(model, holdout_ds, device)
    report["dict_holdout_corpus"] = metrics_dict(h_preds, h_labels)

    seen_keys = set()
    with open(data_dir / "holdout.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            for idx, _, _ in d["sites"]:
                _, keys = lex.match_site(d["text"], idx)
                seen_keys.update(k for k in keys if k in holdout_words)
    missing = {k: v for k, v in holdout_words.items() if k not in seen_keys}
    rng = random.Random(args.seed)
    dict_words = load_site_words(words, exclude_keys=set(holdout_words))
    synth = synthetic_holdout_examples(missing, dict_words, rng)
    if synth:
        corr = Corrector(art_dir / "model.pt", art_dir / "vocab.json", device=device)
        s_preds, s_labels = [], []
        for text, pos, label in synth:
            p = dict(corr.predict_sites(text)).get(pos)
            s_preds.append(int(p >= 0.5))
            s_labels.append(label)
        report["dict_holdout_synthetic"] = metrics_dict(np.asarray(s_preds), np.asarray(s_labels))
        report["dict_holdout_synthetic"]["n_words"] = len(missing)

    with open(art_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    def fmt(name, m):
        if m.get("n"):
            print(f"{name:28s} n={m['n']:>8,}  acc={m['accuracy']:.4f}  macroF1={m['macro_f1']:.4f}  "
                  f"DA P/R={m['da']['precision']:.3f}/{m['da']['recall']:.3f}")
        else:
            print(f"{name:28s} (empty)")

    print(f"\n== COENG TA/DA evaluation ({device}) ==")
    for key in ("baseline_majority", "baseline_dict_lookup", "test", "test_dict_labeled",
                "test_weak_labeled", "dict_holdout_corpus", "dict_holdout_synthetic"):
        if key in report:
            fmt(key, report[key])
    print(f"ambiguous-70 sites in test: {report['ambiguous']['n_sites']} (all predictions valid by definition)")
    print(f"full report -> {art_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
