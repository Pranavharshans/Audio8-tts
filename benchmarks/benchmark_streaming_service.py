#!/usr/bin/env python3
"""Benchmark Audio8's streaming HTTP path, including time to first audio."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import statistics
import subprocess
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--text", default="Welcome to Audio8 TTS.")
    parser.add_argument("--model", default="audio8/tts-0.6b")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), fraction))


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min_ms": min(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "max_ms": max(values),
    }


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def request_once(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": args.model,
            "input": args.text,
            "response_format": "pcm",
            "stream": True,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/audio/speech",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    first_audio_at: float | None = None
    chunk_arrivals: list[float] = []
    chunks: list[bytes] = []
    sample_rate: int | None = None
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                value = line[6:]
                if value == "[DONE]":
                    break
                event = json.loads(value)
                audio = event.get("audio")
                if not audio:
                    continue
                if audio.get("format") != "pcm":
                    raise RuntimeError(f"Expected PCM stream, got {audio.get('format')}")
                now = time.perf_counter()
                if first_audio_at is None:
                    first_audio_at = now
                chunk_arrivals.append(now - started)
                sample_rate = int(audio["sample_rate"])
                chunks.append(base64.b64decode(audio["data"]))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body}") from error

    finished = time.perf_counter()
    if first_audio_at is None or sample_rate is None or not chunks:
        raise RuntimeError("The service returned no streaming audio")
    pcm = b"".join(chunks)
    if len(pcm) % 2:
        raise RuntimeError("The PCM response has an odd byte count")
    audio_seconds = len(pcm) / 2 / sample_rate
    total_seconds = finished - started
    first_chunk_seconds = len(chunks[0]) / 2 / sample_rate
    return {
        "ttfa_ms": (first_audio_at - started) * 1000,
        "total_ms": total_seconds * 1000,
        "audio_seconds": audio_seconds,
        "rtf": total_seconds / audio_seconds,
        "chunk_count": len(chunks),
        "first_chunk_audio_seconds": first_chunk_seconds,
        "chunk_arrival_ms": [value * 1000 for value in chunk_arrivals],
        "sample_rate_hz": sample_rate,
        "pcm_bytes": len(pcm),
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
        "pcm": pcm,
    }


def main() -> None:
    args = parse_args()
    if args.iterations < 1 or args.warmup < 0:
        raise ValueError("iterations must be positive and warmup cannot be negative")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(args.warmup):
        request_once(args)

    measured = [request_once(args) for _ in range(args.iterations)]
    final_pcm = measured[-1].pop("pcm")
    for sample in measured[:-1]:
        sample.pop("pcm")

    sample_rate = int(measured[-1]["sample_rate_hz"])
    with wave.open(str(args.artifact_dir / "stream.wav"), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(final_pcm)

    result = {
        "git_commit": git_commit(),
        "base_url": args.base_url,
        "model": args.model,
        "text": args.text,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "ttfa_ms": summarize([float(sample["ttfa_ms"]) for sample in measured]),
        "total_ms": summarize([float(sample["total_ms"]) for sample in measured]),
        "rtf": {
            key.replace("_ms", ""): value
            for key, value in summarize(
                [float(sample["rtf"]) * 1000 for sample in measured]
            ).items()
        },
        "audio_seconds": summarize(
            [float(sample["audio_seconds"]) * 1000 for sample in measured]
        ),
        "samples": measured,
    }
    # RTF is dimensionless. The helper scales by 1000 to reuse millisecond
    # percentiles, so convert the aggregate values back here.
    for key in ("min", "mean", "p50", "p95", "p99", "max"):
        result["rtf"][key] = float(result["rtf"][key]) / 1000
    result["audio_seconds"] = {
        key.replace("_ms", "_seconds"): value / 1000
        if isinstance(value, float)
        else value
        for key, value in result["audio_seconds"].items()
    }
    (args.artifact_dir / "benchmark.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
