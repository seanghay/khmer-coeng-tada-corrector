"""Dictionary lexicon and the site-label correction matcher."""

from collections import defaultdict
from pathlib import Path

from .charset import COENG, DEPENDENT, find_sites, nfc, norm

WEIGHT_DICT = 1.0
WEIGHT_WEAK = 0.3


def load_dict_words(path: str | Path) -> list[str]:
    words = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            w = nfc(line.strip())
            if w:
                words.append(w)
    return words


class Lexicon:
    """Maps normalized dictionary words to the true consonant at each site."""

    def __init__(self, words: list[str]):
        # key = norm(word); value = {site index within key: set of true consonants}
        self.entries: dict[str, dict[int, set[str]]] = {}
        for w in words:
            sites = find_sites(w)
            if not sites:
                continue
            entry = self.entries.setdefault(norm(w), {})
            for i in sites:
                entry.setdefault(i, set()).add(w[i])
        self.max_key_len = max((len(k) for k in self.entries), default=0)
        # Prune candidate substrings by first char: most starts fail immediately.
        by_first: dict[str, set[int]] = defaultdict(set)
        for k in self.entries:
            by_first[k[0]].add(len(k))
        self.lengths_by_first = {ch: sorted(ls, reverse=True) for ch, ls in by_first.items()}

    def match_site(self, text: str, i: int) -> tuple[set[str] | None, list[str]]:
        """Match dictionary words covering the site at index ``i`` of normalized ``text``.

        Returns (consonant set or None if no boundary-valid match, matched keys).
        The set has one element when the dictionary is unambiguous, more when
        ambiguous (dual-spelling words or conflicting overlaps).
        """
        n = len(text)
        best_len = 0
        best_sets: list[set[str]] = []
        best_keys: list[str] = []
        for start in range(max(0, i - self.max_key_len + 1), i):
            ch = text[start]
            if ch == COENG or ch in DEPENDENT:
                continue
            if start > 0 and text[start - 1] == COENG:
                continue
            lengths = self.lengths_by_first.get(ch)
            if not lengths:
                continue
            rel = i - start
            for klen in lengths:
                if klen < best_len:
                    break  # sorted descending; nothing longer left
                end = start + klen
                if end <= i or end > n:
                    continue
                if end < n and (text[end] == COENG or text[end] in DEPENDENT):
                    continue
                entry = self.entries.get(text[start:end])
                if entry is None:
                    continue
                consonants = entry.get(rel)
                if consonants is None:
                    continue
                if klen > best_len:
                    best_len, best_sets, best_keys = klen, [consonants], [text[start:end]]
                else:
                    best_sets.append(consonants)
                    best_keys.append(text[start:end])
        if not best_sets:
            return None, []
        if len(best_sets) > 1:
            inter = set.intersection(*best_sets)
            return (inter if len(inter) == 1 else set.union(*best_sets)), best_keys
        return best_sets[0], best_keys

    def decide(self, text: str, i: int, corpus_char: str) -> tuple[str, float, list[str]]:
        """Label decision for the site at ``i``: (label, weight, matched keys)."""
        consonants, keys = self.match_site(text, i)
        if consonants is None:
            return corpus_char, WEIGHT_WEAK, keys
        if len(consonants) == 1:
            return next(iter(consonants)), WEIGHT_DICT, keys
        return corpus_char, WEIGHT_DICT, keys

    def unambiguous_keys(self) -> dict[str, str]:
        """Keys whose every site has a single true consonant -> 'ta' | 'da' class.

        A key counts as 'da' if any of its sites is DA (the rarer, harder class).
        """
        from .charset import DA

        out = {}
        for key, entry in self.entries.items():
            if all(len(s) == 1 for s in entry.values()):
                out[key] = "da" if any(DA in s for s in entry.values()) else "ta"
        return out
