# Audio8 TTS Inference Profiling

## Scope

- Repository: `Audio8-tts`
- Model: `Audio8/Audio8-TTS-Preview-0.6b`
- Work branch: `codex/audio8-inference-optimization`
- Quality constraint: no degradation from the reference inference path.
- Deployment constraint: retain only optimizations that can be reproduced in a clean deployment environment.

## Profiling status

| Item | Status |
|---|---|
| GPU/runtime inventory | complete |
| Reference inference correctness | complete |
| Warm-up and timing harness | complete |
| CPU/GPU timeline | complete |
| Kernel-level profile | complete |
| Bottleneck summary | complete |

## Hardware and software

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti |
| VRAM | 16,724,393,984 bytes reported by PyTorch (~15.6 GiB usable total) |
| Compute capability | 8.9 |
| vCPUs / RAM | 8 / ~64 GiB |
| NVIDIA driver | 595.71.05 |
| System CUDA toolkit | 13.2 partial image; `nvcc` available |
| PyTorch runtime | 2.10.0+cu126 |
| TorchAudio | 2.10.0+cu126 |
| Transformers | 4.57.6 |
| Python | 3.12.3 |
| OS/kernel | Linux 6.17.0-35-generic x86_64 |
| Reference commit | `3698607ff712e46cde9a8024cc676c36efb76ede` |

The model checkpoint was downloaded to the VM from `Audio8/Audio8-TTS-Preview-0.6b`; model files are intentionally excluded from Git.

## Method

The profile separates model load, processor/preprocessing, host-to-device transfer, generation, codec decoding, device-to-host transfer, and audio serialization. Warm-up runs are excluded from steady-state latency. The first workload is single-stream, greedy BF16 inference with 128 generated tokens and a 44.1 kHz output. Both no-reference and reference-voice workloads were checked.

Tools used:

- `torch.profiler`: one warmed request; trace stored on the VM at `artifacts/profiling/torch/torch_trace.json` and operator table at `artifacts/profiling/torch/torch_operators.txt`.
- Nsight Systems: one warmed request; report stored on the VM at `artifacts/profiling/nsys/baseline.nsys-rep`.

## Findings

### Baseline bottlenecks

The reference no-reference request produced a valid mono 44.1 kHz WAV. The measured steady-state request was approximately 2.385 s for 106,496 samples (~2.414 s of audio), with generation accounting for 2.350 s and codec decoding 35.9 ms.

| Evidence | Finding |
|---|---|
| PyTorch operator profile | Self CUDA time was 1.014 s under profiler overhead; the largest category was repeated BF16 GEMV work, followed by matrix multiplications and elementwise/cache-copy kernels. |
| Nsight kernel summary | GEMV kernels consumed 39.4% of recorded GPU kernel time across 25,508 launches. An additional GEMV category consumed 7.0%; the full-vocabulary BF16 projection consumed 6.0% across 106 launches. |
| Nsight CUDA API summary | `cudaLaunchKernel` consumed 81.9% of recorded API time across 460,273 launches, indicating launch overhead is material for this token-by-token path. |
| Nsight memory summary | Host-to-device copies were 92.3% of recorded memcpy time across 4,165 operations; device-to-device copies were 7.6% across 12,382 operations. This needs to be separated into input transfer versus model-internal cache writes before changing it. |
| PyTorch shapes | The slow head repeatedly projects `[1, 896]` into `[896, 4864]`, `[4864, 896]`, and the full `[896, 155776]` vocabulary. |
| KV/cache operations | Repeated `copy_`, `index_put`, `cat`, mask, and small elementwise kernels are visible around the static KV-cache updates. |

The dominant optimization surface is the autoregressive generation loop. Codec decoding is currently too small to be the first target.

### Candidate optimization targets

1. Remove redundant full-vocabulary work only where the model’s semantic-token mask proves the result is identical; compare greedy codes exactly.
2. Reduce per-step cache/index/copy overhead without changing cache positions or attention semantics.
3. Test compiler fusion and CUDA Graph capture only after the eager reference is correct and deterministic.
4. Investigate targeted custom kernels only for measured GEMV/cache hotspots; package and test them in the deployment environment.

The model’s downloaded `generate()` implementation already carries `@torch.inference_mode()`. A CLI-level wrapper was therefore tested as E001 and rejected as redundant/statistically neutral; the reference CLI remains unchanged.

Reduced precision, quantization, pruning, model changes, and sampling changes are outside the accepted optimization set under the current quality constraint.

## Deployment notes

Every accepted change must be buildable and runnable from the repository’s pinned environment. Profiling-only settings, unsupported GPU-specific assumptions, and quality-changing precision/quantization changes are not eligible for acceptance without an explicit quality review.
