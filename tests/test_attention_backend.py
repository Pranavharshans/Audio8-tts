from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "sglang_omni"
    / "adapter"
    / "sglang_omni"
    / "models"
    / "audio8_tts"
    / "attention_backend.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("audio8_attention_backend", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fa3_is_limited_to_validated_hopper_capability() -> None:
    module = load_module()
    assert module._capability_supports_fa3((9, 0))
    assert not module._capability_supports_fa3((8, 0))
    assert not module._capability_supports_fa3((8, 9))
    assert not module._capability_supports_fa3((12, 0))


def test_missing_cuda_probe_preserves_cpu_tooling() -> None:
    module = load_module()
    assert module._capability_supports_fa3(None)


def test_explicit_backend_override_wins(monkeypatch) -> None:
    module = load_module()
    monkeypatch.setenv(module.ATTENTION_BACKEND_ENV, "FLASHINFER")
    assert module.resolve_attention_backend() == "flashinfer"
