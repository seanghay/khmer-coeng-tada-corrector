"""Sanity tests: lexicon matcher always; correct() only once a model exists."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "artifacts/model.pt"

from coengtada.charset import DA, TA, find_sites, norm
from coengtada.lexicon import Lexicon, load_dict_words


@pytest.fixture(scope="module")
def lex():
    return Lexicon(load_dict_words(ROOT / "khmerdict.txt"))


def test_find_sites():
    assert find_sites("ស្តីពី") == [2]
    assert find_sites("ស្ដីពី") == [2]
    assert find_sites("កខគ") == []
    # misordered cluster ្រ្ត still detected
    assert find_sites("កន្រ្តាក់") == [5]


def test_norm_preserves_indices():
    s = "កណ្ដាល និងស្ដីពី"
    assert len(norm(s)) == len(s)
    assert find_sites(norm(s)) == find_sites(s)


def test_matcher_relabels_danop(lex):
    # ដណ្តប់ typed with TA must relabel to DA (dictionary has ដណ្ដប់)
    text = norm("គ្របដណ្តប់លើផ្ទៃដី")
    i = find_sites("គ្របដណ្តប់លើផ្ទៃដី")[0]  # the site inside ដណ្តប់
    label, weight, keys = lex.decide(text, i, TA)
    assert label == DA and weight == 1.0


def test_matcher_keeps_ambiguous(lex):
    # កណ្តាល appears with both spellings in the dictionary -> keep corpus label
    for cons, word in ((TA, "កណ្តាល"), (DA, "កណ្ដាល")):
        text = norm(f"នៅ{word}ទីក្រុង")
        i = find_sites(f"នៅ{word}ទីក្រុង")[0]
        label, weight, _ = lex.decide(text, i, cons)
        assert label == cons and weight == 1.0


def test_matcher_unmatched_is_downweighted(lex):
    text = norm("កន្រ្តាក់")  # misordered cluster, not a dictionary form
    i = find_sites(text)[0]
    label, weight, keys = lex.decide(text, i, TA)
    assert weight == 0.3 and label == TA and not keys


needs_model = pytest.mark.skipif(not MODEL.exists(), reason="train a model first")


@needs_model
class TestCorrect:
    @pytest.fixture(scope="class")
    def corr(self):
        from coengtada.infer import Corrector

        return Corrector()

    def test_relabels_known_da_words(self, corr):
        out = corr.correct("ក្រសួងបានចេញព្រឹត្តិបត្រស្តីពីការគ្របដណ្តប់")
        assert "ស្ដីពី" in out or "ស្តីពី" in out  # ambiguous group: both valid
        assert "ដណ្ដប់" in out

    def test_preserves_correct_ta(self, corr):
        text = "សន្តិភាពនៅកម្ពុជា"
        assert corr.correct(text) == text

    def test_ambiguous_membership(self, corr):
        out = corr.correct("នៅកណ្តាលទីក្រុង")
        assert "កណ្តាល" in out or "កណ្ដាល" in out

    def test_idempotent_and_stable(self, corr):
        text = "ខ្យល់កន្រ្តាក់គ្របដណ្តប់លើផ្ទៃដីនៅកណ្តាលប្រទេស"
        once = corr.correct(text)
        assert corr.correct(once) == once
        assert len(once) == len(text)

    def test_no_sites_unchanged(self, corr):
        assert corr.correct("hello ខ្មែរ 123") == "hello ខ្មែរ 123"

    def test_long_document(self, corr):
        text = "ស្តីពីការគ្របដណ្តប់ " * 2500  # ~50k chars
        out = corr.correct(text)
        assert len(out) == len(text)
