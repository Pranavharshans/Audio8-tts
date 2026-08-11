# SPDX-License-Identifier: Apache-2.0
"""Attention backend selection for the Audio8 TTS adapter.

FA3 kernels are built for Hopper only. On every other CUDA architecture the
portable FlashInfer / SDPA path is selected automatically so that a default
deployment starts without extra configuration. An explicit environment
override always wins.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

ATTENTION_BACKEND_ENV = "AUDIO8_TTS_ATTENTION_BACKEND"
DEFAULT_ATTENTION_BACKEND = "fa3"
PORTABLE_ATTENTION_BACKEND = "flashinfer"

# H20, H100, and H200 use Hopper compute capability (9, 0). FA3 does not ship
# kernels for Ampere, Ada, or consumer Blackwell, and unknown future
# capabilities should take the portable path until explicitly validated.
_CAPABILITIES_WITH_FA3 = frozenset({(9, 0)})


def _capability_supports_fa3(capability: Optional[Tuple[int, int]]) -> bool:
    # Preserve import and CPU-only tooling behaviour when CUDA is unavailable.
    return capability is None or capability in _CAPABILITIES_WITH_FA3


def _device_capability() -> Optional[Tuple[int, int]]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_capability()
    except Exception:  # never block startup on a probe
        logger.debug("Could not query CUDA compute capability", exc_info=True)
        return None


@functools.lru_cache(maxsize=1)
def fa3_kernels_available() -> bool:
    """Whether the current device is expected to have an FA3 kernel image."""
    capability = _device_capability()
    if capability is None:
        return True
    if not _capability_supports_fa3(capability):
        logger.info(
            "Compute capability %s is not a validated FA3 target; defaulting to the "
            "'%s' attention backend. Set %s to override.",
            capability,
            PORTABLE_ATTENTION_BACKEND,
            ATTENTION_BACKEND_ENV,
        )
        return False
    return True


def resolve_attention_backend() -> str:
    """Return the attention backend name, honouring the environment override."""
    override = os.getenv(ATTENTION_BACKEND_ENV)
    if override:
        return override.lower()
    if fa3_kernels_available():
        return DEFAULT_ATTENTION_BACKEND
    return PORTABLE_ATTENTION_BACKEND
