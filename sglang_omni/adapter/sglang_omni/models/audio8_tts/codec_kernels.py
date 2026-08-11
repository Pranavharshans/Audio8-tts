# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch
from torch import nn

import triton
import triton.language as tl


@torch.jit.script
def _torchscript_snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    return x.reshape(shape)


@triton.jit
def _snake_kernel(
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


class TritonSnake1d(nn.Module):
    def __init__(self, alpha: nn.Parameter, min_elements: int) -> None:
        super().__init__()
        self.alpha = alpha
        self.min_elements = int(min_elements)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            not x.is_cuda
            or x.dtype != torch.bfloat16
            or x.ndim != 3
            or not x.is_contiguous()
            or not self.alpha.is_contiguous()
            or x.numel() < self.min_elements
        ):
            return _torchscript_snake(x, self.alpha)
        output = torch.empty_like(x)
        n_elements = x.numel()
        _snake_kernel[(triton.cdiv(n_elements, 256),)](
            x,
            self.alpha,
            output,
            n_elements=n_elements,
            channels=x.shape[1],
            width=x.shape[2],
            BLOCK_SIZE=256,
        )
        return output


def install_triton_snake(module: Any, *, min_elements: int = 1 << 20) -> int:
    """Replace decoder Snake modules while retaining their trained parameters."""

    installed = 0
    for name, child in list(module.named_children()):
        if child.__class__.__name__ == "ArkttsSnake1d":
            setattr(module, name, TritonSnake1d(child.alpha, min_elements))
            installed += 1
        else:
            installed += install_triton_snake(child, min_elements=min_elements)
    return installed
