"""Safe tokenizer vocabulary extension utilities for audio8_tts SFT."""

from __future__ import annotations

import json
import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
ALLOWED_FORMAT_CHARACTERS = {"\u200c", "\u200d"}


@dataclass(frozen=True)
class TokenizerExtensionResult:
    requested_tokens: int
    added_tokens: int
    original_tokenizer_size: int
    final_tokenizer_size: int
    original_embedding_size: int
    final_embedding_size: int


def normalize_additional_tokens(values: list[Any]) -> list[str]:
    """Validate, NFC-normalize, and deduplicate additional text tokens."""
    tokens: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise TypeError(f"additional token {index} must be a string")
        token = unicodedata.normalize("NFC", value)
        if not token:
            raise ValueError(f"additional token {index} must not be empty")
        if any(character.isspace() for character in token):
            raise ValueError(
                f"additional token {index} contains whitespace; use in-word subwords only"
            )
        if any(
            unicodedata.category(character).startswith("C")
            and character not in ALLOWED_FORMAT_CHARACTERS
            for character in token
        ):
            raise ValueError(f"additional token {index} contains a control character")
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    if not tokens:
        raise ValueError("additional token list must not be empty")
    return tokens


def load_additional_tokens(path: Path) -> list[str]:
    """Load a JSON token list or a miner output object containing ``tokens``."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"additional token file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid additional token JSON: {path}") from exc
    if isinstance(value, dict):
        value = value.get("tokens")
    if not isinstance(value, list):
        raise ValueError(f"additional token JSON must be a list or contain a tokens list: {path}")
    return normalize_additional_tokens(value)


def extend_tokenizer_embeddings(
    tokenizer,
    model,
    tokens: list[str],
    *,
    logger: logging.Logger | None = None,
) -> TokenizerExtensionResult:
    """Append text tokens and initialize their tied embeddings from old tokenizations.

    Audio8 reserves fixed IDs for semantic audio tokens. ``add_tokens`` appends to
    the vocabulary, and the complete original mapping is checked after extension
    so a tokenizer implementation cannot silently reindex those reserved IDs.
    """
    import torch
    from tokenizers import AddedToken

    log = logger or LOGGER
    tokens = normalize_additional_tokens(tokens)
    original_vocab = tokenizer.get_vocab()
    original_tokenizer_size = len(tokenizer)
    embeddings = model.get_input_embeddings()
    if embeddings is None or not hasattr(embeddings, "weight"):
        raise TypeError("model does not expose a resizable input embedding table")
    original_embedding_size = int(embeddings.weight.shape[0])

    candidates = [token for token in tokens if token not in original_vocab]
    if not candidates:
        log.warning("All requested additional tokens already exist in the tokenizer")
        return TokenizerExtensionResult(
            requested_tokens=len(tokens),
            added_tokens=0,
            original_tokenizer_size=original_tokenizer_size,
            final_tokenizer_size=original_tokenizer_size,
            original_embedding_size=original_embedding_size,
            final_embedding_size=original_embedding_size,
        )

    initial_vectors: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for token in candidates:
            old_ids = tokenizer.encode(token, add_special_tokens=False)
            if not old_ids:
                raise ValueError(f"existing tokenizer produced no IDs for token {token!r}")
            if min(old_ids) < 0 or max(old_ids) >= original_embedding_size:
                raise ValueError(f"existing tokenization for {token!r} contains an out-of-range ID")
            indices = torch.tensor(old_ids, dtype=torch.long, device=embeddings.weight.device)
            initial_vectors[token] = (
                embeddings.weight.index_select(0, indices).mean(dim=0).detach().cpu()
            )

    added = tokenizer.add_tokens(
        [
            AddedToken(
                token,
                normalized=True,
                special=False,
                single_word=False,
                lstrip=False,
                rstrip=False,
            )
            for token in candidates
        ]
    )
    if added != len(candidates):
        log.warning("Tokenizer accepted %d of %d new token candidates", added, len(candidates))

    updated_vocab = tokenizer.get_vocab()
    moved = [
        token for token, token_id in original_vocab.items() if updated_vocab.get(token) != token_id
    ]
    if moved:
        preview = ", ".join(repr(token) for token in moved[:5])
        raise RuntimeError(f"tokenizer extension changed existing token IDs: {preview}")

    added_ids: dict[str, int] = {}
    for token in candidates:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= original_tokenizer_size:
            added_ids[token] = token_id
    if len(added_ids) != added:
        raise RuntimeError(
            f"could not resolve all appended token IDs: expected {added}, found {len(added_ids)}"
        )

    special_ids = {
        name: getattr(model.config, name, None)
        for name in ("semantic_begin_id", "semantic_end_id", "pad_token_id", "eos_token_id")
    }
    required_size = max(len(tokenizer), original_embedding_size)
    if required_size != original_embedding_size:
        model.resize_token_embeddings(required_size)
    model.config.vocab_size = required_size

    resized_embeddings = model.get_input_embeddings()
    with torch.no_grad():
        for token, token_id in added_ids.items():
            vector = initial_vectors[token].to(
                device=resized_embeddings.weight.device,
                dtype=resized_embeddings.weight.dtype,
            )
            resized_embeddings.weight[token_id].copy_(vector)

    for name, expected in special_ids.items():
        if getattr(model.config, name, None) != expected:
            raise RuntimeError(f"tokenizer extension changed model.config.{name}")

    final_embedding_size = int(resized_embeddings.weight.shape[0])
    log.info(
        "Extended tokenizer by %d tokens (%d -> %d); embeddings %d -> %d",
        added,
        original_tokenizer_size,
        len(tokenizer),
        original_embedding_size,
        final_embedding_size,
    )
    return TokenizerExtensionResult(
        requested_tokens=len(tokens),
        added_tokens=added,
        original_tokenizer_size=original_tokenizer_size,
        final_tokenizer_size=len(tokenizer),
        original_embedding_size=original_embedding_size,
        final_embedding_size=final_embedding_size,
    )


__all__ = [
    "TokenizerExtensionResult",
    "extend_tokenizer_embeddings",
    "load_additional_tokens",
    "normalize_additional_tokens",
]
