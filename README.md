# khmer-coeng-tada-corrector

Khmer COENG TA (`្ត`) and COENG DA (`្ដ`) render identically, so they get typed
interchangeably. This model predicts the orthographically correct consonant at
every site from character context — a ~635k-param BiLSTM, also available as a
dependency-free C++ engine and a WASM browser demo.

## Python

```python
from coengtada import Corrector

corrector = Corrector()                    # loads artifacts/model.pt
corrector.correct("គ្របដណ្តប់")             # 'គ្របដណ្ដប់'
corrector.predict_sites("ស្តីពី")           # [(2, P(ដ))]
```

## C++ / WASM

`cpp/coengtada.hpp` is a pure C++17 port — hand-written SIMD kernels (NEON /
WASM SIMD128 / scalar), no libtorch or onnxruntime. Weights load zero-copy:
the native CLI `mmap`s `artifacts/model.bin`; the WASM build embeds it in the
module's data segment. Output is identical to PyTorch.

```sh
./cpp/build.sh                 # build/ctda (native) + web/ctda.{js,wasm}
./build/ctda "អត្ថបទខ្មែរ"       # correct text (stdin works too; -p for P(ដ))
cd web && python3 -m http.server 8791   # browser demo at :8791
```

## Training

Labels come from ~15M lines of crawled Khmer text corrected against
`khmerdict.txt`: real-world text overuses `្ត` ~8:1, so a boundary-aware
substring matcher relabels sites to the dictionary spelling when unambiguous,
keeps both labels for the ~70 dual-spelling words (e.g. កណ្តាល/កណ្ដាល), and
downweights out-of-dictionary sites. Inputs are normalized (every site becomes
`្ត`) so the answer is never visible; augmentation adds dictionary
pseudo-sentences and typing noise.

```sh
uv run python scripts/prepare_data.py --corpus your-khmer-corpus.txt
uv run python -m coengtada.train
uv run python -m coengtada.evaluate
uv run python scripts/export_weights.py   # -> artifacts/model.bin for C++/WASM
```

## Results

| | Accuracy | Macro-F1 |
|---|---|---|
| Majority baseline | 73.0% | 0.42 |
| Corpus test split | 95.2% | 0.94 |
| Held-out dictionary words in context | 96.0% | 0.93 |

~6.6 ms/site (C++ native, single thread) · ~8.2 ms/site (WASM SIMD).

## License

MIT
