"""Inference: correct COENG TA/DA sites in running text."""

from pathlib import Path

import numpy as np
import torch

from .charset import DA, PAD, TA, Vocab, find_sites, norm
from .model import CoengTaDaNet

ROOT = Path(__file__).resolve().parent.parent
WINDOW = 64
CONSONANTS = [TA, DA]


class Corrector:
    def __init__(
        self,
        model_path: str | Path = ROOT / "artifacts/model.pt",
        vocab_path: str | Path = ROOT / "artifacts/vocab.json",
        device: str = "cpu",
        batch_size: int = 512,
    ):
        self.vocab = Vocab.load(vocab_path)
        self.device = device
        self.batch_size = batch_size
        self.model = CoengTaDaNet(self.vocab.size).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        self.model.eval()

    def _windows(self, ids: np.ndarray, sites: list[int]) -> torch.Tensor:
        out = np.full((len(sites), 2 * WINDOW + 1), PAD, dtype=np.int64)
        n = len(ids)
        for r, pos in enumerate(sites):
            lo, hi = pos - WINDOW, pos + WINDOW + 1
            src = ids[max(lo, 0):min(hi, n)]
            left = max(0, -lo)
            out[r, left:left + len(src)] = src
        return torch.from_numpy(out)

    def predict_sites(self, text: str) -> list[tuple[int, float]]:
        """[(site index, P(DA))] for every COENG TA/DA site in ``text``."""
        sites = find_sites(text)
        if not sites:
            return []
        ids = np.asarray(self.vocab.encode(norm(text)), dtype=np.int64)
        windows = self._windows(ids, sites)
        probs = []
        with torch.inference_mode():
            for i in range(0, len(windows), self.batch_size):
                batch = windows[i:i + self.batch_size].to(self.device)
                p = torch.softmax(self.model(batch), dim=1)[:, 1].cpu()
                probs.extend(p.tolist())
        return list(zip(sites, probs))

    def correct(self, text: str) -> str:
        """Rewrite every COENG TA/DA site with the predicted consonant."""
        preds = self.predict_sites(text)
        if not preds:
            return text
        chars = list(text)
        for pos, p_da in preds:
            chars[pos] = CONSONANTS[int(p_da >= 0.5)]
        return "".join(chars)


_corrector: Corrector | None = None


def correct(text: str) -> str:
    """Module-level convenience wrapper with a lazily created singleton."""
    global _corrector
    if _corrector is None:
        _corrector = Corrector()
    return _corrector.correct(text)
