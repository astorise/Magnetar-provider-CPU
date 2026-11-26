## Why

Every GGUF quantization Magnetar loads (`Q8_0`, `Q2_K`..`Q6_K`, `IQ2_*`, `IQ3_*`, `IQ4_*`) now runs packed on CUDA with int8 activations: Mistral 7B decodes in 13-26 ms per token on one RTX 3060 from 2.0-7.7 GB of weights. On the CPU the same checkpoints only run dequantized: `providers/cpu` (the Reference CPU Provider) declines packed weights, so Model Loading expands every tensor to f32 at load. A 7B model then needs about 29 GB of host memory whatever its file size (more than the 31 GB GPU runner host can spare), and every decode step reads those 29 GB, so the CPU path is neither a usable fallback for a machine without a GPU nor a way to run the parts of a model that do not fit in VRAM (expert residency, `execute-and-cache-moe-experts`).

The Reference CPU Provider is deliberately not optimized ("Reference CPU Provider Is Not Optimized Provider") and is the numerical oracle of the conformance suites, so the answer is not to make it fast but to add an optimized CPU Provider beside it.

## What Changes

- A new **optimized CPU Provider** (`magnetar:provider/cpu-native`), shipped from the `magnetar-provider-cpu` repository next to the unchanged Reference CPU Provider, with its own identity, Device and kernel advertisements.
- It **accepts packed weights** (`write_packed_weight_tensor` answers `Stored`) in every `PackedWeightFormat` the Runtime offers, keeping the checkpoint's block bytes in host memory (reorganized only where alignment requires, as on CUDA).
- **Packed matmul kernels** for decode and short prefill rows: activations quantized per 32-value block to int8 (the CUDA default, `ggml`'s `block_q8_1` scheme), integer dot products through the widest instruction available at run time:
  - x86-64: AVX-512 VNNI / AVX-VNNI (`vpdpbusd`), else AVX2 (`vpmaddubsw` + `vpmaddwd`), else SSE4.1;
  - AArch64: `sdot` (dotprod), else NEON `smull`/`sadalp`;
  - a portable scalar kernel everywhere (and on `wasm32`), which is also the test oracle for the SIMD ones.
  Formats in order: `Q8_0`, `Q4_K`, `Q6_K` (the `Q4_K_M` trio), `IQ4_XS`/`IQ4_NL`, `Q2_K`/`Q3_K`, `IQ2_*`/`IQ3_*` (grids from `model_loading::iq_grids`). Gated (`silu(gate) * up`) forms for the MLP, as on CUDA.
- **Multithreading**: output columns split across a persistent worker pool sized from `available_parallelism` (bounded by a control), no new dependency.
- Longer matmuls (prefill beyond the row tile) dequantize one weight panel at a time into a reusable scratch and run a blocked f32 kernel, never the whole model.
- Every other operator (attention, norms, RoPE, softmax, elementwise) delegates to the Reference CPU kernels at first; optimized versions are follow-ups once profiles show them.
- **Controls**: `MAGNETAR_CPU_PACKED_WEIGHTS=0` declines packed weights (f32 at load, today's behavior), `MAGNETAR_CPU_INT8_ACTIVATIONS=0` keeps f32 activations (dequantizing kernels), `MAGNETAR_CPU_THREADS` bounds the pool, `MAGNETAR_CPU_SIMD` caps the instruction set (`scalar`, `avx2`, ...) for tests and comparisons.

Status: **Proposed.** No code changes with this proposal.

## Capabilities

### New Capabilities

- `cpu-quantized-execution`: packed-weight storage, int8-activation quantized matmuls, run-time instruction selection, threading, controls and qualification of the optimized CPU Provider.

### Modified Capabilities

None. The Reference CPU Provider (`cpu-provider`) keeps its contract, including "Reference CPU Quantization Placeholder"; the new Provider has its own identity.

## Impact

- `providers/cpu` (`magnetar-provider-cpu` repository): a second Provider crate or module, its kernels and tests; the submodule pin in Magnetar moves after its own validation.
- `magnetar-runtime`: none required (the `write_packed_weight_tensor` contract, `PackedWeight`, the dequantization ports and `iq_grids` already exist); Resolution chooses between the two CPU Providers through the existing policy and affinity.
- CI: a CPU job in the GPU runner workflow's `gguf` group (its host has 8 cores and 31 GB) and a portable job on the hosted runners for the scalar and AVX2 paths.
- Memory: a 7B `Q4_K_M` instance needs about 4.4 GB of host memory instead of 29 GB.

## Dependencies

- `execute-direct-quantized-weights` (B03): `PackedWeight`, the packed formats and their bit-exact dequantization, the int8-activation scheme and its CUDA evidence.
- `execute-and-cache-moe-experts`: the CPU half of expert residency builds on this Provider.

## Non-goals

- Changing the Reference CPU Provider or its role as the conformance oracle.
- GPU-style capture/replay on the CPU, or a JIT; kernels are compiled ahead of time with run-time dispatch.
- GPTQ/AWQ, and BLAS or other third-party compute libraries.
- `wasm32` SIMD (`simd128`) kernels: the scalar path serves browsers until a follow-up.
