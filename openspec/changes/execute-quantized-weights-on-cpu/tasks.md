## 1. Provider skeleton

- [ ] 1.1 Decide packaging (second crate in `magnetar-provider-cpu` or feature) and add the optimized CPU Provider with its identity, Device, detected `magnetar:cpu-feature/*` capabilities (adding `avx512vnni`, `avxvnni`, `dotprod`, `neon`) and delegation of every operator to the Reference CPU kernels.
- [ ] 1.2 Persistent worker pool bounded by `available_parallelism` and `MAGNETAR_CPU_THREADS`; per-thread scratch for int8 activations and dequantized panels.
- [ ] 1.3 Controls: `MAGNETAR_CPU_PACKED_WEIGHTS`, `MAGNETAR_CPU_INT8_ACTIVATIONS`, `MAGNETAR_CPU_SIMD`, `MAGNETAR_CPU_THREADS`; tests for each.

## 2. Packed storage and scalar kernels

- [ ] 2.1 `write_packed_weight_tensor` storing every `PackedWeightFormat` (with the CUDA alignment reorganizations), Memory Manager admission at packed size, download/expand returning the Runtime's exact dequantization.
- [ ] 2.2 Activation quantization (per 32 values, round to nearest even, block sums) shared with all kernels.
- [ ] 2.3 Scalar int8 kernels for `Q8_0`, `Q4_K`, `Q6_K`, then `IQ4_NL`/`IQ4_XS`, `Q2_K`/`Q3_K`, `IQ2_*`/`IQ3_*`; plain, bias, residual and gated forms; panel-dequantizing path for long matmuls and for f32 activations.
- [ ] 2.4 Conformance tests against the Reference CPU (rounded and f32 activations, 1/3/9/65 rows), per format.

## 3. Vector kernels

- [ ] 3.1 AVX2 kernels (all formats), then AVX-512 VNNI / AVX-VNNI.
- [ ] 3.2 AArch64 `sdot` and NEON kernels.
- [ ] 3.3 Tests forcing every available level against the scalar oracle; skips reported, not hidden.
- [ ] 3.4 Kernel bandwidth diagnostic (Mistral shapes per format and level, GB/s against the host's measured stream bandwidth).

## 4. Qualification

- [ ] 4.1 CI: a CPU job in the GPU runner workflow's `gguf` group (8 cores, 31 GB) running Qwen2.5-0.5B `Q8_0` and Mistral 7B `Q4_K_M`, `IQ4_XS`, `Q2_K` on the optimized CPU Provider: "Paris" required, tokens compared with the CUDA int8 and Reference CPU runs (warnings), TPOT/TTFT/load/peak RSS reported.
- [ ] 4.2 Portable CI on hosted runners: scalar and AVX2 conformance on every push.
- [ ] 4.3 Record evidence per format and level in this change's design; update the support matrix only with qualified results.

## 5. Documentation

- [ ] 5.1 Document the two CPU Providers, how Resolution picks one, the controls and the expected memory and bandwidth bounds.
