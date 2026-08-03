from collections import Counter

from audio8_tts_mine_tokens import (
    iter_malayalam_candidates,
    malayalam_grapheme_runs,
    rank_candidates,
)


class FakeTokenizer:
    def __init__(self, lengths: dict[str, int], vocab: dict[str, int] | None = None):
        self.lengths = lengths
        self.vocab = vocab or {}

    def get_vocab(self) -> dict[str, int]:
        return self.vocab.copy()

    def encode(self, token: str, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return list(range(self.lengths[token]))


def test_malayalam_grapheme_runs_keep_marks_and_conjuncts_together() -> None:
    assert malayalam_grapheme_runs("English മലയാളം, നമസ്കാരം!") == [
        ["മ", "ല", "യാ", "ളം"],
        ["ന", "മ", "സ്കാ", "രം"],
    ]


def test_iter_malayalam_candidates_stays_inside_words() -> None:
    assert list(iter_malayalam_candidates("മലയാളം കേരളം", max_graphemes=2)) == [
        ("മ", 1),
        ("മല", 2),
        ("ല", 1),
        ("ലയാ", 2),
        ("യാ", 1),
        ("യാളം", 2),
        ("ളം", 1),
        ("കേ", 1),
        ("കേര", 2),
        ("ര", 1),
        ("രളം", 2),
        ("ളം", 1),
    ]


def test_rank_candidates_prioritizes_estimated_token_savings() -> None:
    frequencies = Counter({"മ": 10, "മല": 6, "കേരളം": 4, "existing": 100})
    grapheme_counts = {"മ": 1, "മല": 2, "കേരളം": 3, "existing": 1}
    tokenizer = FakeTokenizer(
        {"മ": 3, "മല": 5, "കേരളം": 8, "existing": 4},
        vocab={"existing": 7},
    )

    ranked = rank_candidates(
        tokenizer,
        frequencies,
        grapheme_counts,
        min_frequency=4,
        min_current_tokens=2,
        max_tokens=2,
    )

    assert [item.token for item in ranked] == ["കേരളം", "മല"]
    assert ranked[0].estimated_tokens_saved == 28
    assert ranked[1].estimated_tokens_saved == 24
