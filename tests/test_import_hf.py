import json
from pathlib import Path

import pytest

from audio8_tts_import_hf import (
    ManifestRow,
    atomic_write_bytes,
    audio_bytes_and_suffix,
    split_and_write,
)


def test_audio_bytes_and_suffix_uses_embedded_payload() -> None:
    payload, suffix = audio_bytes_and_suffix({"bytes": b"RIFFdata", "path": "x.wav"})
    assert payload == b"RIFFdata"
    assert suffix == ".wav"


def test_audio_bytes_and_suffix_rejects_missing_audio() -> None:
    with pytest.raises(ValueError, match="neither embedded bytes"):
        audio_bytes_and_suffix({"bytes": None, "path": None})


def test_split_and_write_is_deterministic_and_uses_relative_paths(tmp_path: Path) -> None:
    output_dir = tmp_path / "raw"
    rows = []
    for index in range(10):
        audio = output_dir / "audio" / f"{index}.wav"
        atomic_write_bytes(audio, b"audio")
        rows.append(ManifestRow(f"id-{index}", f"text {index}", audio))

    assert split_and_write(rows, output_dir, eval_samples=2, seed=7) == (8, 2)
    first_eval = (output_dir / "eval.jsonl").read_text(encoding="utf-8")
    split_and_write(rows, output_dir, eval_samples=2, seed=7)
    assert (output_dir / "eval.jsonl").read_text(encoding="utf-8") == first_eval

    values = [json.loads(line) for line in first_eval.splitlines()]
    assert all(value["audio"].startswith("audio/") for value in values)
