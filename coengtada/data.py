"""Windowed site dataset with on-the-fly augmentation."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .charset import DA, PAD, TA, UNK, Vocab, find_sites, norm

LABELS = {TA: 0, DA: 1}
WINDOW = 64
SEPARATORS = ["", " ", "​", "។ ", "៖ "]
ZWSP_ID_CHARS = [" ", "​"]


def load_site_words(dict_words: list[str], exclude_keys: set[str]) -> list[tuple[str, list[tuple[int, int]]]]:
    """Dictionary words usable for pseudo-sentences: (word, [(site idx, label)])."""
    out = []
    for w in dict_words:
        if norm(w) in exclude_keys:
            continue
        sites = [(i, LABELS[w[i]]) for i in find_sites(w)]
        out.append((w, sites))
    return out


class SiteDataset(Dataset):
    """One example per COENG TA/DA site: (window ids, label, weight).

    Windows are ±WINDOW chars around the site, PAD beyond line edges, site
    always at the center index. All stored text is already normalized.
    """

    def __init__(
        self,
        jsonl_paths: list[str | Path],
        vocab: Vocab,
        train: bool = False,
        pseudo_words: list[tuple[str, list[tuple[int, int]]]] | None = None,
        pseudo_p: float = 0.2,
        noise_p: float = 0.1,
        max_lines: int | None = None,
    ):
        self.vocab = vocab
        self.train = train
        self.pseudo_p = pseudo_p
        self.noise_p = noise_p
        chunks: list[np.ndarray] = []
        records = []  # (line_start, line_end, site_abs_pos, label, weight)
        offset = 0
        n_lines = 0
        for path in jsonl_paths:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if max_lines is not None and n_lines >= max_lines:
                        break
                    d = json.loads(line)
                    ids = np.asarray(self.vocab.encode(d["text"]), dtype=np.int16)
                    chunks.append(ids)
                    start, end = offset, offset + len(ids)
                    offset = end
                    n_lines += 1
                    for idx, label, weight in d["sites"]:
                        records.append((start, end, start + idx, LABELS[label], weight))
        self.buffer = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        self.records = np.asarray(records, dtype=np.float64) if records else np.zeros((0, 5))
        self.labels = self.records[:, 3].astype(np.int64)
        # Pseudo-sentence sources bucketed by the label they can supply.
        self.pseudo_by_label: dict[int, list[tuple[str, list[int]]]] = {0: [], 1: []}
        self.pseudo_all: list[str] = []
        if train and pseudo_words:
            for w, sites in pseudo_words:
                self.pseudo_all.append(w)
                for lab in (0, 1):
                    idxs = [i for i, site_label in sites if site_label == lab]
                    if idxs:
                        self.pseudo_by_label[lab].append((w, idxs))

    def __len__(self) -> int:
        return len(self.records)

    def _window_from_buffer(self, start: int, end: int, pos: int) -> np.ndarray:
        lo, hi = pos - WINDOW, pos + WINDOW + 1
        arr = np.full(2 * WINDOW + 1, PAD, dtype=np.int64)
        src = self.buffer[max(lo, start):min(hi, end)]
        left = max(0, start - lo)
        arr[left:left + len(src)] = src
        return arr

    def _window_from_text(self, text: str, pos: int) -> np.ndarray:
        ids = np.asarray(self.vocab.encode(text), dtype=np.int64)
        arr = np.full(2 * WINDOW + 1, PAD, dtype=np.int64)
        lo, hi = pos - WINDOW, pos + WINDOW + 1
        src = ids[max(lo, 0):min(hi, len(ids))]
        left = max(0, -lo)
        arr[left:left + len(src)] = src
        return arr

    def _pseudo_example(self, label: int) -> tuple[np.ndarray, int, float]:
        """A dictionary word with a `label` site, embedded among random words."""
        target, site_idxs = random.choice(self.pseudo_by_label[label])
        others = random.choices(self.pseudo_all, k=random.randint(2, 7))
        slot = random.randint(0, len(others))
        parts = others[:slot] + [target] + others[slot:]
        seps = [random.choice(SEPARATORS) for _ in range(len(parts) - 1)] + [""]
        pieces, target_off = [], 0
        cursor = 0
        for i, (p, sep) in enumerate(zip(parts, seps)):
            if i == slot:
                target_off = cursor
            pieces.append(p + sep)
            cursor += len(p) + len(sep)
        text = norm("".join(pieces))
        pos = target_off + random.choice(site_idxs)
        return self._window_from_text(text, pos), label, 1.0

    def _add_noise(self, arr: np.ndarray) -> np.ndarray:
        """One random perturbation, never touching the center COENG+consonant."""
        center = WINDOW
        protected = {center - 1, center}
        j = random.randrange(len(arr))
        if j in protected or arr[j] == PAD:
            return arr
        op = random.randrange(4)
        if op == 0:  # substitute with UNK
            arr[j] = UNK
        elif op == 1:  # delete: shift the outer part of that side inward
            if j < center:
                arr[1:j + 1] = arr[:j].copy()
                arr[0] = PAD
            else:
                arr[j:-1] = arr[j + 1:].copy()
                arr[-1] = PAD
        else:  # duplicate char, or insert separator
            ins = arr[j] if op == 2 else self.vocab.encode_char(random.choice(ZWSP_ID_CHARS))
            if j < center:
                arr[:j] = arr[1:j + 1].copy()
                arr[j] = ins
            else:
                arr[j + 1:] = arr[j:-1].copy()
                arr[j] = ins
        return arr

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, float]:
        start, end, pos, label, weight = self.records[i]
        label = int(label)
        if self.train and self.pseudo_by_label[label] and random.random() < self.pseudo_p:
            arr, label, weight = self._pseudo_example(label)
        else:
            arr = self._window_from_buffer(int(start), int(end), int(pos))
        if self.train and random.random() < self.noise_p:
            arr = self._add_noise(arr)
        return torch.from_numpy(arr), label, float(weight)


def sampler_weights(labels: np.ndarray, target_ratio: float = 3.0) -> torch.Tensor:
    """Per-example sampling weights that oversample DA to ~target_ratio TA:DA."""
    n_ta = int((labels == 0).sum())
    n_da = int((labels == 1).sum())
    da_weight = max(1.0, n_ta / (target_ratio * max(1, n_da)))
    w = np.where(labels == 1, da_weight, 1.0)
    return torch.from_numpy(w)
