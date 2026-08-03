#!/usr/bin/env python3
"""Materialize a Hugging Face audio dataset as Audio8 raw JSONL manifests."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    text: str
    audio: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a Hugging Face audio/text dataset, materialize its audio, and "
            "create deterministic train/eval JSONL manifests."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="Praha-Labs/TTS-Ml")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--text-column", default="text")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stop after this many rows. Omit for the complete dataset.",
    )
    parser.add_argument("--eval-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def iter_dataset_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is required; install requirements-train.txt first"
        ) from exc

    dataset = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=True,
        cache_dir=os.environ.get("HF_DATASETS_CACHE"),
    )
    dataset = dataset.cast_column(args.audio_column, Audio(decode=False))
    return dataset


def audio_bytes_and_suffix(value: Any) -> tuple[bytes, str]:
    if not isinstance(value, dict):
        raise ValueError(f"expected an audio mapping, received {type(value).__name__}")

    payload = value.get("bytes")
    source_path = value.get("path")
    if payload is None and source_path:
        source = Path(source_path)
        if source.is_file():
            payload = source.read_bytes()
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("audio row has neither embedded bytes nor a readable local path")

    suffix = Path(source_path or "audio.wav").suffix.lower()
    if suffix not in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}:
        suffix = ".wav"
    return payload, suffix


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size == len(payload):
        return
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(path)


def relative_audio_path(audio: Path, manifest: Path) -> str:
    return os.path.relpath(audio, start=manifest.parent)


def write_jsonl(path: Path, rows: Iterable[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            value = {
                "id": row.sample_id,
                "text": row.text,
                "audio": relative_audio_path(row.audio, path),
            }
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    temporary.replace(path)


def split_and_write(
    rows: list[ManifestRow], output_dir: Path, *, eval_samples: int, seed: int
) -> tuple[int, int]:
    if not rows:
        raise ValueError("the dataset produced no usable rows")
    if eval_samples < 1 or eval_samples >= len(rows):
        raise ValueError(
            f"--eval-samples must be between 1 and {len(rows) - 1}; got {eval_samples}"
        )

    eval_indices = set(random.Random(seed).sample(range(len(rows)), eval_samples))
    train_rows = [row for index, row in enumerate(rows) if index not in eval_indices]
    eval_rows = [row for index, row in enumerate(rows) if index in eval_indices]
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "eval.jsonl", eval_rows)
    return len(train_rows), len(eval_rows)


def main() -> None:
    args = parse_args()
    if args.max_samples is not None and args.max_samples < 2:
        raise ValueError("--max-samples must be at least 2")

    output_dir = args.output_dir.resolve()
    rows: list[ManifestRow] = []
    for index, value in enumerate(iter_dataset_rows(args)):
        if args.max_samples is not None and index >= args.max_samples:
            break
        text = value.get(args.text_column)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"row {index} has an empty {args.text_column!r} value")
        payload, suffix = audio_bytes_and_suffix(value.get(args.audio_column))
        sample_id = f"tts_ml_{index:08d}"
        audio_path = output_dir / "audio" / f"{index // 1000:05d}" / f"{sample_id}{suffix}"
        atomic_write_bytes(audio_path, payload)
        rows.append(ManifestRow(sample_id, text.strip(), audio_path))
        if (index + 1) % 500 == 0:
            print(f"[audio8_tts.import_hf] materialized={index + 1}", flush=True)

    train_count, eval_count = split_and_write(
        rows, output_dir, eval_samples=args.eval_samples, seed=args.seed
    )
    print(
        f"[audio8_tts.import_hf] train={train_count} eval={eval_count} "
        f"output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
