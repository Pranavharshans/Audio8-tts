#!/usr/bin/env python3
"""Compare Audio8's TorchScript Snake activation with a Triton kernel."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import torch
import triton
import triton.language as tl


@torch.jit.script
def torchscript_snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    return x.reshape(shape)


@triton.jit
def _triton_snake_kernel(
    x_ptr,
    alpha_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    channels: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    channel_offsets = (offsets // width) % channels
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    alpha = tl.load(alpha_ptr + channel_offsets, mask=mask).to(tl.float32)
    sinusoid = tl.sin(alpha * x)
    output = x + (1.0 / (alpha + 1e-9)) * (sinusoid * sinusoid)
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3 or alpha.shape != (1, x.shape[1], 1):
        raise ValueError("expected x=[batch, channels, width], alpha=[1, channels, 1]")
    if not x.is_contiguous() or not alpha.is_contiguous():
        raise ValueError("the Triton Snake kernel requires contiguous tensors")
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, 256),)
    _triton_snake_kernel[grid](
        x,
        alpha,
        output,
        n_elements=n_elements,
        channels=x.shape[1],
        width=x.shape[2],
        BLOCK_SIZE=256,
    )
    return output


@dataclass
class Measurement:
    channels: int
    width: int
    elements: int
    exact: bool
    max_abs_error: float
    torchscript_ms: float
    triton_ms: float
    speedup: float


def elapsed_ms(function, x: torch.Tensor, alpha: torch.Tensor, repetitions: int) -> float:
    for _ in range(5):
        function(x, alpha)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        function(x, alpha)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repetitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(0)
    shapes = ((1536, 16), (768, 128), (384, 1024), (192, 8192), (96, 32768))
    measurements = []
    for channels, width in shapes:
        x = torch.randn(1, channels, width, device="cuda", dtype=torch.bfloat16)
        alpha = (
            0.5 + torch.rand(1, channels, 1, device="cuda", dtype=torch.bfloat16)
        ).contiguous()
        reference = torchscript_snake(x, alpha)
        candidate = triton_snake(x, alpha)
        reference_ms = elapsed_ms(torchscript_snake, x, alpha, args.repetitions)
        candidate_ms = elapsed_ms(triton_snake, x, alpha, args.repetitions)
        measurements.append(
            Measurement(
                channels=channels,
                width=width,
                elements=x.numel(),
                exact=torch.equal(reference, candidate),
                max_abs_error=float((reference.float() - candidate.float()).abs().max()),
                torchscript_ms=reference_ms,
                triton_ms=candidate_ms,
                speedup=reference_ms / candidate_ms,
            )
        )
    print(json.dumps([asdict(item) for item in measurements], indent=2))


if __name__ == "__main__":
    main()
