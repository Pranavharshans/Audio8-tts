#!/usr/bin/env python3
"""Compare deterministic baseline and candidate Audio8 TTS artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--allow-code-difference", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_codes = np.load(args.baseline / "codes.npy")
    candidate_codes = np.load(args.candidate / "codes.npy")
    baseline_audio = np.load(args.baseline / "audio.npy").astype(np.float32)
    candidate_audio = np.load(args.candidate / "audio.npy").astype(np.float32)
    report = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "codes_same_shape": baseline_codes.shape == candidate_codes.shape,
        "codes_exact_equal": bool(np.array_equal(baseline_codes, candidate_codes)),
        "audio_same_shape": baseline_audio.shape == candidate_audio.shape,
    }
    if report["audio_same_shape"]:
        delta = np.abs(baseline_audio - candidate_audio)
        report.update(
            {
                "audio_max_abs_delta": float(delta.max(initial=0.0)),
                "audio_mean_abs_delta": float(delta.mean()),
                "audio_rms_delta": float(np.sqrt(np.mean(np.square(delta)))),
                "baseline_audio_rms": float(np.sqrt(np.mean(np.square(baseline_audio)))),
            }
        )
    report["quality_gate_pass"] = bool(
        report["codes_same_shape"]
        and report["audio_same_shape"]
        and (report["codes_exact_equal"] or args.allow_code_difference)
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["quality_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
