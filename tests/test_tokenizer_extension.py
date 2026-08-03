import json
from types import SimpleNamespace

import pytest

from audio8_tts_tokenizer import (
    extend_tokenizer_embeddings,
    load_additional_tokens,
    normalize_additional_tokens,
)


def test_normalize_additional_tokens_normalizes_and_deduplicates() -> None:
    assert normalize_additional_tokens(["e\u0301", "é", "മലയാളം", "ക്\u200dക"]) == [
        "é",
        "മലയാളം",
        "ക്\u200dക",
    ]


@pytest.mark.parametrize("token", ["", "two words", "line\nbreak", "bad\u200bcontrol"])
def test_normalize_additional_tokens_rejects_unsafe_tokens(token: str) -> None:
    with pytest.raises(ValueError):
        normalize_additional_tokens([token])


def test_load_additional_tokens_accepts_miner_output(tmp_path) -> None:
    path = tmp_path / "tokens.json"
    path.write_text(
        json.dumps({"format": "audio8_tts.additional_tokens.v1", "tokens": ["മ", "മല"]}),
        encoding="utf-8",
    )

    assert load_additional_tokens(path) == ["മ", "മല"]


def test_extend_tokenizer_embeddings_appends_without_reindexing() -> None:
    torch = pytest.importorskip("torch")

    class FakeTokenizer:
        def __init__(self):
            self.vocab = {"a": 0, "b": 1}

        def __len__(self):
            return len(self.vocab)

        def get_vocab(self):
            return self.vocab.copy()

        def encode(self, token, add_special_tokens):
            assert not add_special_tokens
            return [0, 1]

        def add_tokens(self, values):
            added = 0
            for value in values:
                token = value.content
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                    added += 1
            return added

        def convert_tokens_to_ids(self, token):
            return self.vocab[token]

    class FakeModel:
        def __init__(self):
            self.embedding = torch.nn.Embedding(4, 3)
            with torch.no_grad():
                self.embedding.weight[0].fill_(1.0)
                self.embedding.weight[1].fill_(3.0)
            self.config = SimpleNamespace(
                vocab_size=4,
                semantic_begin_id=10,
                semantic_end_id=20,
                pad_token_id=0,
                eos_token_id=1,
            )

        def get_input_embeddings(self):
            return self.embedding

        def resize_token_embeddings(self, size):
            replacement = torch.nn.Embedding(size, self.embedding.embedding_dim)
            with torch.no_grad():
                rows = min(size, self.embedding.num_embeddings)
                replacement.weight[:rows].copy_(self.embedding.weight[:rows])
            self.embedding = replacement
            self.config.vocab_size = size
            return replacement

    tokenizer = FakeTokenizer()
    model = FakeModel()
    result = extend_tokenizer_embeddings(tokenizer, model, ["മ", "ല", "യ"])

    assert tokenizer.get_vocab() == {"a": 0, "b": 1, "മ": 2, "ല": 3, "യ": 4}
    assert result.added_tokens == 3
    assert result.final_embedding_size == 5
    expected = torch.full((3,), 2.0)
    for token_id in (2, 3, 4):
        assert torch.allclose(model.embedding.weight[token_id], expected)
    assert model.config.semantic_begin_id == 10
    assert model.config.semantic_end_id == 20
