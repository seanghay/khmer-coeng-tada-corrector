"""Character-level BiLSTM site classifier (~635k params)."""

import torch
import torch.nn as nn

EMB_DIM = 64
HIDDEN = 128


class CoengTaDaNet(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = EMB_DIM, hidden: int = HIDDEN,
                 dropout: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            emb_dim,
            hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) char ids with the site at the center index. Returns (B, 2) logits."""
        out, _ = self.rnn(self.emb(x))
        return self.head(out[:, out.size(1) // 2])
