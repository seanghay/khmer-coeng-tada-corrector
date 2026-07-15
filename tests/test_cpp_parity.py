"""Parity: the C++ engine must produce identical corrections to Python.

Skipped unless build/ctda and artifacts/model.pt exist (run ./cpp/build.sh).
"""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CTDA = ROOT / "build/ctda"

pytestmark = pytest.mark.skipif(
    not (CTDA.exists() and (ROOT / "artifacts/model.pt").exists()),
    reason="build/ctda or model missing",
)

TEXTS = [
    "ខ្យល់កន្រ្តាក់គ្របដណ្តប់លើផ្ទៃដី",
    "ក្រសួងបានចេញព្រឹត្តិបត្រស្តីពីស្ថានភាពធាតុអាកាស",
    "សេចក្តីស្រឡាញ់ និងសន្ដិភាពនៅកម្ពុជា",
    "គាត់រស់នៅកណ្តាលទីក្រុងភ្នំពេញ",
    "hello ខ្មែរ 123",  # latin/digit buckets, no sites
    "បញ្ចូលឆ្នាំ២០២៥ លេខ០១២៣៤៥៦៧៨៩ និង ្ត ចាប់ផ្ដើម",  # khmer digits, leading site
]


def test_cpp_matches_python():
    from coengtada import Corrector

    corr = Corrector()
    expected = [corr.correct(t) for t in TEXTS]
    out = subprocess.run(
        [str(CTDA), "-m", str(ROOT / "artifacts/model.bin"), *TEXTS],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    assert out == expected


def test_cpp_probabilities_close():
    from coengtada import Corrector

    corr = Corrector()
    for text in TEXTS:
        py = corr.predict_sites(text)
        raw = subprocess.run(
            [str(CTDA), "-m", str(ROOT / "artifacts/model.bin"), "-p", text],
            capture_output=True, text=True, check=True,
        ).stdout.splitlines()
        cpp = [(int(a), float(b)) for a, b in (line.split("\t") for line in raw)]
        assert [i for i, _ in cpp] == [i for i, _ in py]
        for (_, p_cpp), (_, p_py) in zip(cpp, py):
            assert abs(p_cpp - p_py) < 1e-3
