"""Checkpoint retention and qualitative sampling callbacks for Audio8 TTS SFT."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from transformers import TrainerCallback


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SamplePrompt:
    sample_id: str
    text: str
    seed_offset: int = 0


def load_sample_prompts(path: Path) -> list[SamplePrompt]:
    """Load and validate a JSONL prompt list used for checkpoint comparisons."""
    prompts: list[SamplePrompt] = []
    seen: set[str] = set()
    with path.resolve().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            sample_id = value.get("id")
            text = value.get("text")
            seed_offset = value.get("seed_offset", 0)
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{path}:{line_number}: id must be a non-empty string")
            if not all(character.isalnum() or character in "_-" for character in sample_id):
                raise ValueError(f"{path}:{line_number}: unsafe id {sample_id!r}")
            if sample_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate id {sample_id!r}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_number}: text must be a non-empty string")
            if not isinstance(seed_offset, int) or seed_offset < 0:
                raise ValueError(f"{path}:{line_number}: seed_offset must be non-negative")
            seen.add(sample_id)
            prompts.append(SamplePrompt(sample_id, text.strip(), seed_offset))
    if not prompts:
        raise ValueError(f"no sample prompts found in {path}")
    return prompts


def _json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False) + "\n"


def _actual_model(model: torch.nn.Module) -> torch.nn.Module:
    while hasattr(model, "module"):
        model = model.module
    return model


class PermanentEpochCheckpointCallback(TrainerCallback):
    """Hard-link complete epoch checkpoints outside Trainer's rotating directory set."""

    def __init__(self, processor, *, epochs: tuple[int, ...] = (1, 2, 3)) -> None:
        self.processor = processor
        self.epochs = set(int(epoch) for epoch in epochs)
        self.pending_epoch: int | None = None

    def on_epoch_end(self, args, state, control, **kwargs):
        del args, kwargs
        if state.epoch is None or not state.is_world_process_zero:
            return control
        completed_epoch = int(round(float(state.epoch)))
        if completed_epoch in self.epochs and abs(float(state.epoch) - completed_epoch) < 1e-6:
            self.pending_epoch = completed_epoch
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        del kwargs
        if self.pending_epoch is None or not state.is_world_process_zero:
            return control
        epoch = self.pending_epoch
        self.pending_epoch = None
        source = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        destination_root = Path(args.output_dir) / "epoch_checkpoints"
        destination = destination_root / f"epoch-{epoch}"
        if destination.is_dir():
            LOGGER.info("Permanent epoch checkpoint already exists: %s", destination)
            return control
        if not source.is_dir():
            LOGGER.error("Cannot preserve epoch %d; checkpoint is missing: %s", epoch, source)
            return control

        destination_root.mkdir(parents=True, exist_ok=True)
        temporary = destination_root / f".epoch-{epoch}.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        try:
            # Hard links avoid double allocation while the rotating source exists.
            shutil.copytree(source, temporary, copy_function=os.link, symlinks=True)
        except OSError as exc:
            LOGGER.warning(
                "Hard-link checkpoint copy failed (%s); falling back to file copies", exc
            )
            if temporary.exists():
                shutil.rmtree(temporary)
            shutil.copytree(source, temporary, copy_function=shutil.copy2, symlinks=True)
        self.processor.save_pretrained(temporary)
        (temporary / "epoch_metadata.json").write_text(
            json.dumps(
                {"epoch": epoch, "global_step": int(state.global_step), "source": source.name},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        LOGGER.info("Preserved permanent epoch checkpoint: %s", destination)
        return control


class PeriodicAudioSamplingCallback(TrainerCallback):
    """Generate reference-conditioned comparison audio from the live training model."""

    def __init__(
        self,
        processor,
        *,
        prompts: list[SamplePrompt],
        reference_audio: Path,
        reference_text: str,
        output_dir: Path,
        every_steps: int,
        seed: int = 42,
        max_new_tokens: int = 1024,
        retry_max_new_tokens: int = 2000,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        offload_optimizer: bool = True,
    ) -> None:
        self.processor = processor
        self.prompts = prompts
        self.reference_audio = reference_audio.resolve()
        self.reference_text = reference_text.strip()
        self.output_dir = output_dir.resolve()
        self.every_steps = int(every_steps)
        self.seed = int(seed)
        self.max_new_tokens = int(max_new_tokens)
        self.retry_max_new_tokens = int(retry_max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.offload_optimizer = bool(offload_optimizer)
        self.last_sampled_step = -1

    @staticmethod
    def _offload_optimizer_state(optimizer) -> list[tuple[dict[str, Any], str, torch.device]]:
        moved: list[tuple[dict[str, Any], str, torch.device]] = []
        if optimizer is None:
            return moved
        for values in optimizer.state.values():
            for key, value in list(values.items()):
                if torch.is_tensor(value) and value.device.type == "cuda":
                    device = value.device
                    values[key] = value.detach().to("cpu")
                    moved.append((values, key, device))
        return moved

    @staticmethod
    def _restore_optimizer_state(
        moved: list[tuple[dict[str, Any], str, torch.device]],
    ) -> None:
        for values, key, device in moved:
            values[key] = values[key].to(device)

    @staticmethod
    def _offload_codec(model: torch.nn.Module) -> None:
        codec = model.__dict__.get("_arktts_codec")
        if codec is not None:
            codec.to(device="cpu", dtype=torch.float32)

    def _synthesize(self, model: torch.nn.Module, prompt: SamplePrompt, seed: int):
        device = next(model.parameters()).device
        inputs = self.processor(
            text=[prompt.text],
            reference_audio=[self.reference_audio],
            reference_text=[self.reference_text],
            return_tensors="pt",
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        generator = torch.Generator(device=device).manual_seed(seed)
        output = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            do_sample=True,
            generator=generator,
            return_dict_in_generate=True,
        )
        if not bool(output.finished[0]) and self.retry_max_new_tokens > self.max_new_tokens:
            generator = torch.Generator(device=device).manual_seed(seed)
            output = model.generate(
                **inputs,
                max_new_tokens=self.retry_max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=True,
                generator=generator,
                return_dict_in_generate=True,
            )
        waveforms, waveform_lengths = model.decode_audio(output.codes)
        waveform_length = int(waveform_lengths[0])
        waveform = waveforms[0, :waveform_length].float().cpu().numpy()
        return waveform, int(model.config.codec_sample_rate), bool(output.finished[0])

    def on_step_end(self, args, state, control, **kwargs):
        del args
        step = int(state.global_step)
        if (
            self.every_steps <= 0
            or step <= 0
            or step % self.every_steps != 0
            or step == self.last_sampled_step
            or not state.is_world_process_zero
        ):
            return control
        self.last_sampled_step = step

        model = _actual_model(kwargs["model"])
        optimizer = kwargs.get("optimizer")
        step_dir = self.output_dir / f"step-{step:08d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        moved_optimizer: list[tuple[dict[str, Any], str, torch.device]] = []
        was_training = model.training
        started = time.monotonic()

        LOGGER.info("Generating %d qualitative samples at step %d", len(self.prompts), step)
        try:
            if self.offload_optimizer:
                moved_optimizer = self._offload_optimizer_state(optimizer)
                torch.cuda.empty_cache()
            model.eval()
            with torch.inference_mode():
                for index, prompt in enumerate(self.prompts):
                    output = step_dir / f"{prompt.sample_id}.wav"
                    try:
                        waveform, sample_rate, finished = self._synthesize(
                            model, prompt, self.seed + step + prompt.seed_offset + index
                        )
                        sf.write(output, waveform, sample_rate)
                        records.append(
                            {
                                "id": prompt.sample_id,
                                "text": prompt.text,
                                "output_audio": str(output),
                                "finished": finished,
                                "sample_rate": sample_rate,
                            }
                        )
                    except Exception as exc:  # Keep qualitative monitoring non-fatal.
                        LOGGER.exception("Sample %s failed at step %d", prompt.sample_id, step)
                        failures.append(
                            {
                                "id": prompt.sample_id,
                                "text": prompt.text,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                        torch.cuda.empty_cache()
        finally:
            self._offload_codec(model)
            torch.cuda.empty_cache()
            self._restore_optimizer_state(moved_optimizer)
            if was_training:
                model.train()

        (step_dir / "manifest.jsonl").write_text(
            "".join(_json_line(record) for record in records), encoding="utf-8"
        )
        (step_dir / "failures.jsonl").write_text(
            "".join(_json_line(record) for record in failures), encoding="utf-8"
        )
        (step_dir / "sampling_metadata.json").write_text(
            json.dumps(
                {
                    "global_step": step,
                    "reference_audio": str(self.reference_audio),
                    "reference_text": self.reference_text,
                    "samples": len(records),
                    "failures": len(failures),
                    "duration_seconds": round(time.monotonic() - started, 3),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        LOGGER.info(
            "Qualitative sampling step %d complete: samples=%d failures=%d output=%s",
            step,
            len(records),
            len(failures),
            step_dir,
        )
        return control


__all__ = [
    "PeriodicAudioSamplingCallback",
    "PermanentEpochCheckpointCallback",
    "SamplePrompt",
    "load_sample_prompts",
]
