"""Character-level utilities: site detection, normalization, and vocabulary."""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

PAD = 0
UNK = 1
LATIN = 2
DIGIT = 3
NUM_SPECIAL = 4

COENG = "្"
TA = "ត"  # ត
DA = "ដ"  # ដ

SITE_RE = re.compile(f"{COENG}[{TA}{DA}]")

# Dependent vowels/signs that cannot begin a word or immediately follow one
# without belonging to it (U+17B6–U+17D1, U+17DD).
DEPENDENT = frozenset(chr(c) for c in range(0x17B6, 0x17D2)) | {"៝"}


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def norm(s: str) -> str:
    """Normalize every COENG DA site to COENG TA so inputs never reveal the label."""
    return s.replace(COENG + DA, COENG + TA)


def find_sites(s: str) -> list[int]:
    """Indices of the TA/DA consonant of every COENG+TA/DA site."""
    return [m.start() + 1 for m in SITE_RE.finditer(s)]


def _is_bucketed(ch: str) -> bool:
    return (ch.isascii() and ch.isalpha()) or ch.isdigit()


class Vocab:
    def __init__(self, stoi: dict[str, int]):
        self.stoi = stoi
        self.size = NUM_SPECIAL + len(stoi)

    def encode_char(self, ch: str) -> int:
        i = self.stoi.get(ch)
        if i is not None:
            return i
        if ch.isascii() and ch.isalpha():
            return LATIN
        if ch.isdigit():
            return DIGIT
        return UNK

    def encode(self, s: str) -> list[int]:
        return [self.encode_char(c) for c in s]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False, indent=0)

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    @classmethod
    def build(cls, char_counts: Counter, extra_chars: set[str], min_freq: int = 50) -> "Vocab":
        chars = {ch for ch, c in char_counts.items() if c >= min_freq and not _is_bucketed(ch)}
        chars |= {ch for ch in extra_chars if not _is_bucketed(ch)}
        chars.discard("\n")
        stoi = {ch: NUM_SPECIAL + i for i, ch in enumerate(sorted(chars))}
        return cls(stoi)
