#!/usr/bin/env python3
"""Benchmark Audio8 TTS inference with stage timings and output artifacts."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", default="Welcome to Audio8 TTS.")
    parser.add_argument("--reference-audio")
    parser.add_argument("--reference-text")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument(
        "--inference-mode",
        action="store_true",
        help="Run the request under torch.inference_mode().",
    )
    parser.add_argument(
        "--compile-methods",
        action="store_true",
        help="Compile the model's custom slow and fast decode methods with TorchInductor.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def nvtx(message: str, entering: bool) -> None:
    if not torch.cuda.is_available():
        return
    try:
        if entering:
            torch.cuda.nvtx.range_push(message)
        else:
            torch.cuda.nvtx.range_pop()
    except RuntimeError:
        pass


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), fraction))


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "min_ms": float(min(values)) if values else float("nan"),
        "mean_ms": float(statistics.fmean(values)) if values else float("nan"),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "max_ms": float(max(values)) if values else float("nan"),
    }


def tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_runtime(args: argparse.Namespace) -> tuple[Any, Any, torch.device, torch.dtype]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The inference benchmark requires CUDA")
    dtype = getattr(torch, args.dtype)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=dtype)
    model.eval().to(device)
    if args.compile_methods:
        model._slow_step = torch.compile(
            model._slow_step, mode=args.compile_mode, fullgraph=False
        )
        model._fast_step = torch.compile(
            model._fast_step, mode=args.compile_mode, fullgraph=False
        )
    return processor, model, device, dtype


def make_inputs(processor: Any, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    if bool(args.reference_audio) != bool(args.reference_text):
        raise ValueError("--reference-audio and --reference-text must be provided together")
    texts = [args.text] * args.batch_size
    kwargs: dict[str, Any] = {"text": texts, "return_tensors": "pt"}
    if args.reference_audio:
        kwargs["reference_audio"] = [Path(args.reference_audio)] * args.batch_size
        kwargs["reference_text"] = [args.reference_text] * args.batch_size
    inputs = processor(**kwargs)
    return {
        name: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }


def run_once(
    processor: Any,
    model: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, int]:
    context = torch.inference_mode() if args.inference_mode else nullcontext()
    with context:
        return _run_once(processor, model, device, args)


def _run_once(
    processor: Any,
    model: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, int]:
    start = time.perf_counter()
    nvtx("audio8.request", True)

    nvtx("audio8.preprocess", True)
    inputs = make_inputs(processor, args, device)
    synchronize(device)
    preprocess_end = time.perf_counter()
    nvtx("audio8.preprocess", False)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    nvtx("audio8.generate", True)
    generate_start = time.perf_counter()
    output = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=not args.greedy,
        generator=generator,
        return_dict_in_generate=True,
    )
    synchronize(device)
    generate_end = time.perf_counter()
    nvtx("audio8.generate", False)

    nvtx("audio8.decode", True)
    decode_start = time.perf_counter()
    waveforms, waveform_lengths = model.decode_audio(output.codes)
    synchronize(device)
    decode_end = time.perf_counter()
    nvtx("audio8.decode", False)

    codes = tensor_to_numpy(output.codes)
    audio = tensor_to_numpy(waveforms)
    lengths = tensor_to_numpy(waveform_lengths).astype(np.int64)
    end = time.perf_counter()
    nvtx("audio8.request", False)

    if not np.isfinite(audio).all():
        raise RuntimeError("Inference produced NaN or infinite audio samples")
    timings = {
        "request_ms": (end - start) * 1000,
        "preprocess_and_h2d_ms": (preprocess_end - start) * 1000,
        "generate_ms": (generate_end - generate_start) * 1000,
        "decode_ms": (decode_end - decode_start) * 1000,
        "postprocess_ms": (end - decode_end) * 1000,
    }
    return timings, codes, audio, int(lengths.reshape(-1)[0])


def main() -> None:
    args = parse_args()
    if args.iterations < 1 or args.warmup < 0 or args.batch_size < 1:
        raise ValueError("iterations and batch-size must be positive; warmup cannot be negative")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    processor, model, device, dtype = load_runtime(args)

    for _ in range(args.warmup):
        run_once(processor, model, device, args)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timings: dict[str, list[float]] = {}
    final_codes: np.ndarray | None = None
    final_audio: np.ndarray | None = None
    final_length = 0
    for _ in range(args.iterations):
        sample, final_codes, final_audio, final_length = run_once(processor, model, device, args)
        for name, value in sample.items():
            timings.setdefault(name, []).append(value)

    assert final_codes is not None and final_audio is not None
    np.save(args.artifact_dir / "codes.npy", final_codes)
    np.save(args.artifact_dir / "audio.npy", final_audio)

    code_bytes = final_codes.tobytes(order="C")
    audio_bytes = final_audio.astype(np.float32, copy=False).tobytes(order="C")
    result: dict[str, Any] = {
        "git_commit": git_commit(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "vram_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        "model": args.model,
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "greedy": args.greedy,
        "inference_mode": args.inference_mode,
        "compile_methods": args.compile_methods,
        "compile_mode": args.compile_mode,
        "text": args.text,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "waveform_length_samples": final_length,
        "sample_rate_hz": 44100,
        "timings": {name: summarize(values) for name, values in timings.items()},
        "vram_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "vram_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "codes_sha256": hashlib.sha256(code_bytes).hexdigest(),
        "audio_float32_sha256": hashlib.sha256(audio_bytes).hexdigest(),
        "audio_min": float(final_audio.min()),
        "audio_max": float(final_audio.max()),
    }
    (args.artifact_dir / "benchmark.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
