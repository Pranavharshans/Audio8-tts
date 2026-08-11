#!/usr/bin/env python3
"""Evaluate streamed Audio8 speech against full decode and eager artifacts."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import re
import urllib.request
import wave
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--model", default="audio8/tts-0.6b")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--asr-model")
    parser.add_argument("--speaker-model")
    parser.add_argument("--reuse-audio", action="store_true")
    parser.add_argument("--wer-tolerance", type=float, default=0.0)
    parser.add_argument("--speaker-tolerance", type=float, default=0.01)
    parser.add_argument("--stream-full-cosine-min", type=float, default=0.999)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row["id"])
        if sample_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate id {sample_id!r}")
        seen.add(sample_id)
        if bool(row.get("reference_audio")) != bool(row.get("reference_text")):
            raise ValueError(f"{path}:{line_number}: incomplete reference")
        if row.get("reference_audio"):
            reference = (path.parent / row["reference_audio"]).resolve()
            if not reference.is_file():
                raise FileNotFoundError(reference)
            row["reference_audio"] = str(reference)
        rows.append(row)
    if not rows:
        raise ValueError(f"No prompts found in {path}")
    return rows


def payload_for(row: dict[str, Any], args: argparse.Namespace, *, stream: bool) -> bytes:
    payload: dict[str, Any] = {
        "model": args.model,
        "input": row["text"],
        "response_format": "pcm" if stream else "wav",
        "stream": stream,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    if row.get("reference_audio"):
        payload["references"] = [
            {
                "audio_path": row["reference_audio"],
                "text": row["reference_text"],
            }
        ]
    return json.dumps(payload).encode("utf-8")


def request_stream(row: dict[str, Any], args: argparse.Namespace) -> tuple[int, bytes]:
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/audio/speech",
        data=payload_for(row, args, stream=True),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks: list[bytes] = []
    sample_rate: int | None = None
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            audio = event.get("audio")
            if audio:
                sample_rate = int(audio["sample_rate"])
                chunks.append(base64.b64decode(audio["data"]))
    if sample_rate is None or not chunks:
        raise RuntimeError(f"No streamed audio for {row['id']}")
    return sample_rate, b"".join(chunks)


def request_full(row: dict[str, Any], args: argparse.Namespace) -> bytes:
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/audio/speech",
        data=payload_for(row, args, stream=False),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        return response.read()


def pcm_wav(sample_rate: int, pcm: bytes) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return output.getvalue()


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise ValueError(f"Expected PCM16 WAV: {path}")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return sample_rate, audio


def resample(audio: np.ndarray, source_rate: int, target_rate: int = 16000) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    count = max(1, round(len(audio) * target_rate / source_rate))
    source = np.arange(len(audio), dtype=np.float64)
    target = np.linspace(0, max(len(audio) - 1, 0), count, dtype=np.float64)
    return np.interp(target, source, audio).astype(np.float32)


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, 1):
        current = [i]
        for j, right_item in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def error_rate(reference: str, hypothesis: str, *, characters: bool = False) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    left = list(ref.replace(" ", "")) if characters else ref.split()
    right = list(hyp.replace(" ", "")) if characters else hyp.split()
    return edit_distance(left, right) / max(len(left), 1)


@dataclass
class AsrEvaluator:
    processor: Any
    model: Any
    device: Any
    dtype: Any

    @classmethod
    def load(cls, model_name: str) -> "AsrEvaluator":
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, torch_dtype=dtype)
        return cls(processor, model.eval().to(device), device, dtype)

    def transcribe(self, path: Path) -> str:
        import torch

        sample_rate, audio = read_wav(path)
        audio = resample(audio, sample_rate)
        features = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            token_ids = self.model.generate(features)
        return self.processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()


@dataclass
class SpeakerEvaluator:
    extractor: Any
    model: Any
    device: Any

    @classmethod
    def load(cls, model_name: str) -> "SpeakerEvaluator":
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModelForAudioXVector.from_pretrained(model_name)
        return cls(extractor, model.eval().to(device), device)

    def embed(self, path: Path) -> np.ndarray:
        import torch

        sample_rate, audio = read_wav(path)
        audio = resample(audio, sample_rate)
        inputs = self.extractor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            embedding = self.model(**inputs).embeddings[0]
        embedding = torch.nn.functional.normalize(embedding, dim=0)
        return embedding.float().cpu().numpy()


def waveform_metrics(full: np.ndarray, stream: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "full_samples": len(full),
        "stream_samples": len(stream),
        "same_length": len(full) == len(stream),
    }
    if len(full) != len(stream):
        return result
    delta = full.astype(np.float64) - stream.astype(np.float64)
    result.update(
        cosine=float(np.dot(full, stream) / (np.linalg.norm(full) * np.linalg.norm(stream))),
        snr_db=float(20 * np.log10(np.linalg.norm(full) / max(np.linalg.norm(delta), 1e-12))),
        rms_delta=float(np.sqrt(np.mean(delta * delta))),
        max_abs_delta=float(np.max(np.abs(delta), initial=0.0)),
    )
    return result


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        sample_dir = args.output_dir / row["id"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        if args.reuse_audio and (sample_dir / "stream.wav").is_file() and (
            sample_dir / "full.wav"
        ).is_file():
            continue
        sample_rate, stream_pcm = request_stream(row, args)
        (sample_dir / "stream.wav").write_bytes(pcm_wav(sample_rate, stream_pcm))
        (sample_dir / "full.wav").write_bytes(request_full(row, args))

    asr = AsrEvaluator.load(args.asr_model) if args.asr_model else None
    speaker = SpeakerEvaluator.load(args.speaker_model) if args.speaker_model else None
    details: list[dict[str, Any]] = []
    for row in rows:
        sample_dir = args.output_dir / row["id"]
        _, stream_audio = read_wav(sample_dir / "stream.wav")
        _, full_audio = read_wav(sample_dir / "full.wav")
        result: dict[str, Any] = {
            "id": row["id"],
            "text": row["text"],
            "waveform": waveform_metrics(full_audio, stream_audio),
        }
        baseline_path = args.baseline_dir / f"{row['id']}.wav" if args.baseline_dir else None
        if asr:
            stream_text = asr.transcribe(sample_dir / "stream.wav")
            full_text = asr.transcribe(sample_dir / "full.wav")
            asr_result: dict[str, Any] = {
                "stream_text": stream_text,
                "full_text": full_text,
                "stream_wer": error_rate(row["text"], stream_text),
                "full_wer": error_rate(row["text"], full_text),
                "stream_cer": error_rate(row["text"], stream_text, characters=True),
                "full_cer": error_rate(row["text"], full_text, characters=True),
            }
            if baseline_path and baseline_path.is_file():
                baseline_text = asr.transcribe(baseline_path)
                asr_result.update(
                    baseline_text=baseline_text,
                    baseline_wer=error_rate(row["text"], baseline_text),
                    baseline_cer=error_rate(row["text"], baseline_text, characters=True),
                )
            result["asr"] = asr_result
        if speaker and row.get("reference_audio"):
            reference_embedding = speaker.embed(Path(row["reference_audio"]))
            stream_embedding = speaker.embed(sample_dir / "stream.wav")
            speaker_result: dict[str, Any] = {
                "stream_reference_cosine": float(np.dot(stream_embedding, reference_embedding))
            }
            if baseline_path and baseline_path.is_file():
                baseline_embedding = speaker.embed(baseline_path)
                speaker_result["baseline_reference_cosine"] = float(
                    np.dot(baseline_embedding, reference_embedding)
                )
            result["speaker"] = speaker_result
        details.append(result)

    waveform_pass = all(
        bool(result["waveform"]["same_length"])
        and float(result["waveform"].get("cosine", -1.0))
        >= args.stream_full_cosine_min
        for result in details
    )
    asr_pairs = [
        (float(result["asr"]["stream_wer"]), float(result["asr"]["baseline_wer"]))
        for result in details
        if "baseline_wer" in result.get("asr", {})
    ]
    speaker_pairs = [
        (
            float(result["speaker"]["stream_reference_cosine"]),
            float(result["speaker"]["baseline_reference_cosine"]),
        )
        for result in details
        if "baseline_reference_cosine" in result.get("speaker", {})
    ]
    aggregates: dict[str, Any] = {
        "stream_full_waveform_pass": waveform_pass,
    }
    gates = [waveform_pass]
    if asr_pairs:
        stream_wer = float(np.mean([pair[0] for pair in asr_pairs]))
        baseline_wer = float(np.mean([pair[1] for pair in asr_pairs]))
        asr_pass = stream_wer <= baseline_wer + args.wer_tolerance
        aggregates["asr"] = {
            "stream_macro_wer": stream_wer,
            "baseline_macro_wer": baseline_wer,
            "pass": asr_pass,
        }
        gates.append(asr_pass)
    if speaker_pairs:
        stream_speaker = float(np.mean([pair[0] for pair in speaker_pairs]))
        baseline_speaker = float(np.mean([pair[1] for pair in speaker_pairs]))
        speaker_pass = stream_speaker >= baseline_speaker - args.speaker_tolerance
        aggregates["speaker"] = {
            "stream_mean_reference_cosine": stream_speaker,
            "baseline_mean_reference_cosine": baseline_speaker,
            "delta": stream_speaker - baseline_speaker,
            "pass": speaker_pass,
        }
        gates.append(speaker_pass)
    summary = {
        "quality_gate_pass": all(gates),
        "prompt_count": len(details),
        "thresholds": {
            "wer_tolerance": args.wer_tolerance,
            "speaker_tolerance": args.speaker_tolerance,
            "stream_full_cosine_min": args.stream_full_cosine_min,
        },
        "aggregates": aggregates,
        "details": details,
    }
    (args.output_dir / "quality.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["quality_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
