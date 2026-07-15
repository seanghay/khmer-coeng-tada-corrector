"""Export model.pt + vocab.json to a single binary for the C++/WASM engine.

Format (little-endian):
  magic "CTDA", u32 version=1, u32 vocab_size, u32 emb_dim, u32 hidden,
  u32 window, u32 n_vocab_entries, then n x (u32 codepoint, u32 id),
  then float32 tensors in the fixed order listed in TENSOR_ORDER.
"""

import json
import struct
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VERSION = 2  # v2 = BiLSTM (v1 was BiGRU)

TENSOR_ORDER = [
    "emb.weight",
    "rnn.weight_ih_l0", "rnn.weight_hh_l0", "rnn.bias_ih_l0", "rnn.bias_hh_l0",
    "rnn.weight_ih_l0_reverse", "rnn.weight_hh_l0_reverse",
    "rnn.bias_ih_l0_reverse", "rnn.bias_hh_l0_reverse",
    "rnn.weight_ih_l1", "rnn.weight_hh_l1", "rnn.bias_ih_l1", "rnn.bias_hh_l1",
    "rnn.weight_ih_l1_reverse", "rnn.weight_hh_l1_reverse",
    "rnn.bias_ih_l1_reverse", "rnn.bias_hh_l1_reverse",
    "head.0.weight", "head.0.bias", "head.3.weight", "head.3.bias",
]


def main():
    art = ROOT / "artifacts"
    state = torch.load(art / "model.pt", map_location="cpu", weights_only=True)
    with open(art / "vocab.json", encoding="utf-8") as f:
        stoi = json.load(f)
    with open(art / "config.json") as f:
        cfg = json.load(f)

    out = art / "model.bin"
    with open(out, "wb") as f:
        f.write(b"CTDA")
        f.write(struct.pack("<6I", VERSION, cfg["vocab_size"], cfg["emb_dim"], cfg["hidden"],
                            cfg["window"], len(stoi)))
        for ch, idx in sorted(stoi.items(), key=lambda kv: ord(kv[0])):
            assert len(ch) == 1
            f.write(struct.pack("<2I", ord(ch), idx))
        for name in TENSOR_ORDER:
            t = state[name].contiguous().float()
            f.write(t.numpy().tobytes())
    n_params = sum(state[n].numel() for n in TENSOR_ORDER)
    print(f"wrote {out} ({out.stat().st_size:,} bytes, {n_params:,} params, "
          f"vocab {cfg['vocab_size']})")
    assert set(TENSOR_ORDER) == set(state.keys()), set(state.keys()) ^ set(TENSOR_ORDER)


if __name__ == "__main__":
    main()
