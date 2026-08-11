#!/usr/bin/env python3
"""Capture a PyTorch CPU/CUDA profile for one warmed Audio8 TTS request."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from benchmark_inference import load_runtime, make_inputs, nvtx, synchronize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", default="Welcome to Audio8 TTS.")
    parser.add_argument("--reference-audio")
    parser.add_argument("--reference-text")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor, model, device, _ = load_runtime(args)
    for _ in range(2):
        inputs = make_inputs(processor, args, device)
        model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            do_sample=not args.greedy,
            return_dict_in_generate=True,
        )
        synchronize(device)

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profile:
        nvtx("audio8.profile_request", True)
        inputs = make_inputs(processor, args, device)
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            do_sample=not args.greedy,
            return_dict_in_generate=True,
        )
        synchronize(device)
        model.decode_audio(output.codes)
        synchronize(device)
        nvtx("audio8.profile_request", False)

    profile.export_chrome_trace(str(args.output_dir / "torch_trace.json"))
    table = profile.key_averages(group_by_input_shape=True).table(
        sort_by="self_cuda_time_total", row_limit=80
    )
    (args.output_dir / "torch_operators.txt").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
