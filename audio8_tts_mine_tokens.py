#!/usr/bin/env python3
"""Mine Malayalam tokenizer-extension candidates from SFT transcripts."""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Audio8/Audio8-TTS-Preview-0.6b"
MALAYALAM_START = 0x0D00
MALAYALAM_END = 0x0D7F
MALAYALAM_VIRAMA = "\u0d4d"
JOINERS = {"\u200c", "\u200d"}


@dataclass(frozen=True)
class TokenCandidate:
    token: str
    frequency: int
    current_token_count: int
    estimated_tokens_saved: int
    grapheme_count: int


def is_malayalam_character(character: str) -> bool:
    return len(character) == 1 and MALAYALAM_START <= ord(character) <= MALAYALAM_END


def _finish_cluster(
    run: list[str],
    cluster: list[str],
) -> None:
    if cluster:
        run.append("".join(cluster))
        cluster.clear()


def malayalam_grapheme_runs(text: str) -> list[list[str]]:
    """Return approximate extended-grapheme runs for Malayalam script text.

    Malayalam marks stay attached to their base, and consonants following a
    virama or joiner stay in the same cluster. Non-Malayalam characters split
    runs, which keeps mined candidates inside words.
    """
    runs: list[list[str]] = []
    run: list[str] = []
    cluster: list[str] = []

    def finish_run() -> None:
        _finish_cluster(run, cluster)
        if run:
            runs.append(run.copy())
            run.clear()

    for character in unicodedata.normalize("NFC", str(text)):
        if is_malayalam_character(character):
            category = unicodedata.category(character)
            continues_cluster = bool(cluster) and (
                category.startswith("M")
                or cluster[-1] == MALAYALAM_VIRAMA
                or cluster[-1] in JOINERS
            )
            if not continues_cluster:
                _finish_cluster(run, cluster)
            cluster.append(character)
        elif character in JOINERS and cluster:
            cluster.append(character)
        else:
            finish_run()
    finish_run()
    return runs


def iter_malayalam_candidates(text: str, max_graphemes: int) -> Iterator[tuple[str, int]]:
    for run in malayalam_grapheme_runs(text):
        for start in range(len(run)):
            limit = min(len(run), start + max_graphemes)
            for end in range(start + 1, limit + 1):
                yield "".join(run[start:end]), end - start


def read_manifest_texts(paths: Iterable[Path], fields: tuple[str, ...]) -> Iterator[str]:
    for path in paths:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"manifest does not exist: {path}")
        with path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: row must be a JSON object")
                for field in fields:
                    value = row.get(field)
                    if value:
                        yield str(value)


def count_candidates(
    texts: Iterable[str],
    *,
    max_graphemes: int,
) -> tuple[Counter[str], dict[str, int], int]:
    frequencies: Counter[str] = Counter()
    grapheme_counts: dict[str, int] = {}
    text_count = 0
    for text in texts:
        text_count += 1
        for token, grapheme_count in iter_malayalam_candidates(text, max_graphemes):
            frequencies[token] += 1
            grapheme_counts[token] = grapheme_count
    return frequencies, grapheme_counts, text_count


def rank_candidates(
    tokenizer,
    frequencies: Counter[str],
    grapheme_counts: dict[str, int],
    *,
    min_frequency: int,
    min_current_tokens: int,
    max_tokens: int,
) -> list[TokenCandidate]:
    existing_vocab = tokenizer.get_vocab()
    candidates: list[TokenCandidate] = []
    for token, frequency in frequencies.items():
        if frequency < min_frequency or token in existing_vocab:
            continue
        current_ids = tokenizer.encode(token, add_special_tokens=False)
        current_token_count = len(current_ids)
        if current_token_count < min_current_tokens:
            continue
        candidates.append(
            TokenCandidate(
                token=token,
                frequency=frequency,
                current_token_count=current_token_count,
                estimated_tokens_saved=frequency * (current_token_count - 1),
                grapheme_count=grapheme_counts[token],
            )
        )
    candidates.sort(
        key=lambda item: (
            -item.estimated_tokens_saved,
            -item.frequency,
            -item.grapheme_count,
            item.token,
        )
    )
    return candidates[:max_tokens]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine frequent Malayalam grapheme/subword tokens for Audio8 TTS SFT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-graphemes", type=int, default=6)
    parser.add_argument("--min-frequency", type=int, default=5)
    parser.add_argument("--min-current-tokens", type=int, default=2)
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["text", "reference_text"],
        help="JSONL text fields included when mining candidates.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("max_tokens", "max_graphemes", "min_frequency", "min_current_tokens"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not args.fields:
        raise ValueError("--fields must not be empty")


def write_output(
    path: Path,
    *,
    model: str,
    manifests: list[Path],
    fields: list[str],
    text_count: int,
    raw_candidate_count: int,
    candidates: list[TokenCandidate],
    tokenizer_size: int,
) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": "audio8_tts.additional_tokens.v1",
        "language": "Malayalam",
        "model": model,
        "source_manifests": [str(item.expanduser().resolve()) for item in manifests],
        "fields": fields,
        "text_count": text_count,
        "raw_candidate_count": raw_candidate_count,
        "original_tokenizer_size": tokenizer_size,
        "tokens": [item.token for item in candidates],
        "candidates": [asdict(item) for item in candidates],
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    frequencies, grapheme_counts, text_count = count_candidates(
        read_manifest_texts(args.input_jsonl, tuple(args.fields)),
        max_graphemes=args.max_graphemes,
    )
    candidates = rank_candidates(
        tokenizer,
        frequencies,
        grapheme_counts,
        min_frequency=args.min_frequency,
        min_current_tokens=args.min_current_tokens,
        max_tokens=args.max_tokens,
    )
    if not candidates:
        raise ValueError(
            "no Malayalam token candidates survived filtering; check the transcripts or "
            "lower --min-frequency"
        )
    write_output(
        args.output_json,
        model=args.model,
        manifests=args.input_jsonl,
        fields=args.fields,
        text_count=text_count,
        raw_candidate_count=len(frequencies),
        candidates=candidates,
        tokenizer_size=len(tokenizer),
    )
    total_savings = sum(item.estimated_tokens_saved for item in candidates)
    print(
        f"[audio8_tts.mine_tokens] texts={text_count} raw_candidates={len(frequencies)} "
        f"selected={len(candidates)} estimated_occurrence_savings={total_savings} "
        f"output={args.output_json.expanduser().resolve()}"
    )


if __name__ == "__main__":
    main()
