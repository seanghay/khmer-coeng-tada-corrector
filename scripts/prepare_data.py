"""One-time corpus pass: dictionary label correction, splits, holdout, vocab.

Writes data/processed/{train,val,test,holdout}.jsonl with records
  {"text": <normalized line>, "sites": [[idx, "ត"|"ដ", weight], ...]}
plus artifacts/vocab.json, artifacts/holdout_words.json and data/processed/stats.json.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coengtada.charset import TA, Vocab, find_sites, nfc, norm
from coengtada.lexicon import WEIGHT_DICT, Lexicon, load_dict_words

VOCAB_SAMPLE_LINES = 200_000


def line_hash(s: str) -> int:
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:8], "little")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="raw Khmer text corpus, one sentence per line")
    ap.add_argument("--dict", default=str(ROOT / "khmerdict.txt"))
    ap.add_argument("--out", default=str(ROOT / "data/processed"))
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--max-lines", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-frac", type=float, default=0.10)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir = Path(args.artifacts)
    art_dir.mkdir(parents=True, exist_ok=True)

    words = load_dict_words(args.dict)
    lex = Lexicon(words)

    # Held-out dictionary words: 10% of unambiguous DA-keys and TA-keys.
    unambig = lex.unambiguous_keys()
    rng = random.Random(args.seed)
    holdout_keys: set[str] = set()
    for cls in ("da", "ta"):
        keys = sorted(k for k, c in unambig.items() if c == cls)
        holdout_keys |= set(rng.sample(keys, max(1, int(len(keys) * args.holdout_frac))))
    with open(art_dir / "holdout_words.json", "w", encoding="utf-8") as f:
        json.dump(
            {k: {"class": unambig[k], "sites": {str(i): sorted(s)[0] for i, s in lex.entries[k].items()}}
             for k in sorted(holdout_keys)},
            f, ensure_ascii=False, indent=1,
        )

    stats = Counter()
    char_counts: Counter = Counter()
    seen_hashes: set[int] = set()
    writers = {name: open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") for name in
               ("train", "val", "test", "holdout")}

    with open(args.corpus, encoding="utf-8") as f:
        for line_no, raw in enumerate(tqdm(f, total=args.max_lines, unit="lines")):
            if args.max_lines is not None and line_no >= args.max_lines:
                break
            line = nfc(raw.rstrip("\n"))
            sites = find_sites(line)
            if not sites:
                continue
            text = norm(line)
            h = line_hash(text)
            if h in seen_hashes:
                stats["dup_lines"] += 1
                continue
            seen_hashes.add(h)

            records = []
            is_holdout = False
            for i in sites:
                corpus_char = line[i]
                label, weight, keys = lex.decide(text, i, corpus_char)
                if any(k in holdout_keys for k in keys):
                    is_holdout = True
                records.append([i, label, weight])
                stats["sites"] += 1
                stats[f"label_{'ta' if label == TA else 'da'}"] += 1
                if weight == WEIGHT_DICT and keys:
                    if label != corpus_char:
                        stats[f"relabel_{'ta_to_da' if corpus_char == TA else 'da_to_ta'}"] += 1
                    else:
                        stats["match_kept"] += 1
                    stats["matched"] += 1
                else:
                    stats["unmatched"] += 1

            if is_holdout:
                split = "holdout"
            else:
                bucket = h % 100
                split = "train" if bucket < 98 else ("val" if bucket == 98 else "test")
            writers[split].write(json.dumps({"text": text, "sites": records}, ensure_ascii=False) + "\n")
            stats[f"lines_{split}"] += 1
            if stats["lines_train"] <= VOCAB_SAMPLE_LINES and split == "train":
                char_counts.update(text)

    for w in writers.values():
        w.close()

    vocab = Vocab.build(char_counts, extra_chars=set("".join(words)))
    vocab.save(art_dir / "vocab.json")
    stats["vocab_size"] = vocab.size

    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(dict(sorted(stats.items())), f, ensure_ascii=False, indent=1)
    total = stats["sites"] or 1
    print(f"\nsites: {stats['sites']:,}  matched: {stats['matched'] / total:.1%}  "
          f"relabels ta→da: {stats['relabel_ta_to_da']:,}  da→ta: {stats['relabel_da_to_ta']:,}")
    print(f"labels  ta: {stats['label_ta']:,}  da: {stats['label_da']:,}  "
          f"(ratio {stats['label_ta'] / max(1, stats['label_da']):.2f}:1)")
    print(f"lines  train: {stats['lines_train']:,}  val: {stats['lines_val']:,}  "
          f"test: {stats['lines_test']:,}  holdout: {stats['lines_holdout']:,}  dups: {stats['dup_lines']:,}")
    print(f"vocab size: {vocab.size}")


if __name__ == "__main__":
    main()
