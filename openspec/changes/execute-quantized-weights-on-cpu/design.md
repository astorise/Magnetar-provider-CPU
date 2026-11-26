## Context

On CUDA, direct quantized execution rests on four pieces: packed bytes kept per format (`CudaQ8Weight`), activations quantized to int8 per 32-value block before each packed matmul, per-format integer dot kernels (`__dp4a`), and gated forms for the MLP. With them every format reaches 180-310 GB/s and decode cost follows the file size. The CPU side has none of these: Model Loading hands the Reference CPU Provider f32 weights, and its kernels are straightforward loops meant to be read, not to be fast.

A CPU decode step is bandwidth-bound in the same way: on a dual-channel DDR5 desktop (about 60-80 GB/s) a 4.4 GB `Q4_K_M` 7B costs at least 55-75 ms per token, against at least 360-480 ms for the 29 GB f32 form. The goal is to approach that bound.

## Goals / Non-Goals

**Goals:** run every packed format on the CPU from its own bytes, near memory bandwidth on decode, with the same numerics as the CUDA int8 path; keep a scalar oracle and disable controls; no new dependency.

**Non-Goals:** see the proposal.

## Decisions

### 1. A separate optimized Provider, not a faster Reference CPU

`magnetar:provider/cpu-native` is a new Provider identity with its own Device (`DeviceType::Cpu`, compute units, detected instruction sets as `magnetar:cpu-feature/*` capabilities, already produced by `detected_simd_capabilities`). It delegates operators it does not optimize to the Reference CPU kernels (the same crate's public functions), so it is complete from the start. The Reference CPU Provider stays the oracle; conformance compares the two.

Alternative rejected: optimizing `providers/cpu` in place, which would break "Reference CPU Provider Is Not Optimized Provider" and lose the independent oracle.

### 2. Storage

Packed weights stay host-resident as their checkpoint blocks, transposed storage as offered (stored row = output column, contiguous `k`), so a column's dot product streams one contiguous run. Per format, the same reorganizations as CUDA where they help alignment (scales split from quants for `Q6_K`, `Q2_K`, `Q3_K`, the i-quants; `Q3_K` scales unpacked), done once at `write_packed_weight_tensor`. The Memory Manager admits the packed size.

### 3. Int8 activations and integer dots

Per matmul, the activation rows are quantized into a reusable per-thread scratch: per 32 values, `d = max|x| / 127`, `q = round_ties_even(x / d)` (the CUDA rule, so the two Providers agree on rounding), plus each block's sum for the formats with mins (`Q4_K`, `Q2_K`). Per format the kernel decodes weight bytes into signed 8-bit lanes and accumulates `sum(q_w * q_a)` in 32-bit integers per block, then scales in f32:

| ISA | 32-value block dot |
|---|---|
| AVX-512 VNNI / AVX-VNNI | `vpdpbusd` on unsigned-offset activations (bias corrected by the block sum) |
| AVX2 | `vpmaddubsw` (unsigned x signed) + `vpmaddwd` |
| AArch64 dotprod | `sdot` |
| NEON | `smull` + `sadalp` |
| scalar | `i32` loop, the oracle |

Unsigned-times-signed instructions take the weights as the signed side; `IQ4` table lookups use `vpshufb`/`tbl`, IQ2/IQ3 grids load 8-byte entries from `iq_grids` and apply signs with a byte mask (`vpsignb`-style or xor/sub), as on CUDA.

Dispatch is chosen once per Provider from `is_x86_feature_detected!` / `is_aarch64_feature_detected!`, capped by `MAGNETAR_CPU_SIMD`.

### 4. Threads

A persistent pool of `min(available_parallelism, MAGNETAR_CPU_THREADS)` workers (std threads, parked between steps) splits output columns in contiguous ranges aligned to cache lines; activations are quantized once by the caller and shared read-only. Decode matmuls are memory-bound, so the pool is sized for bandwidth, typically the physical core count. Other operators run on the calling thread until profiled.

### 5. Longer matmuls

For `m` beyond the row tile (prefill), weights are dequantized one panel (for example 64 output columns) at a time into a per-thread f32 scratch and multiplied with a blocked f32 kernel; nothing larger than a panel is materialized.

### 6. Numerics and tests

- Each SIMD kernel equals the scalar kernel exactly per block (integer sums) and to f32 summation-order tolerance overall; the scalar kernel matches the Reference CPU on the int8-rounded activations, exactly as the CUDA conformance tests do.
- `MAGNETAR_CPU_INT8_ACTIVATIONS=0` keeps f32 activations through the panel-dequantizing path, matching the Reference CPU on the f32 weights.
- Real checkpoints: tokens compared with the CUDA int8 run and with the Reference CPU f32 run (warning on divergence, "Paris" required), as for CUDA.

## Risks / Trade-offs

- [ISA matrix grows test surface] -> one scalar oracle; each SIMD level forced in CI through `MAGNETAR_CPU_SIMD`; AVX-512 and `sdot` only where the CI hardware has them, reported as skipped otherwise.
- [Threads compete with the GPU Provider's host work] -> the pool bound control; the CPU Provider is resolved only when policy or affinity selects it.
- [Int8 activations change outputs] -> same scheme and disable control as CUDA; divergence reported per checkpoint.
- [Delegated f32 operators (attention over long contexts) dominate after matmuls] -> measured per phase (`A06` ledger) before optimizing them.

## Migration Plan

Additive: a new Provider, unselected unless policy resolves it. Rollback is not resolving it or `MAGNETAR_CPU_PACKED_WEIGHTS=0`.

## Open Questions

- Provider identity and packaging: a second crate in `magnetar-provider-cpu` or a feature of the existing one.
- Default thread count: physical cores (needs topology detection) or `available_parallelism`.
- Whether `wasm32` gets `simd128` kernels in this change or a follow-up.

## Evidence: first increment (commit `43b7745`, local 4-core AVX2 host)

`magnetar:provider/cpu-native` stores every packed format and reads them
in its matmul, matmul-bias, matmul-residual and embedding Kernels; other
operators run the Reference CPU kernels. Conformance tests against the
Reference CPU pass for all twelve formats.

| Kernel, one row, 4 threads, AVX2 | `[4096, 14336]` | `[14336, 4096]` |
|---|---|---|
| `Q8_0` | 3.16 ms (19.8 GB/s) | 3.37 ms (18.5 GB/s) |
| `Q4_K` | 5.85 ms (5.6 GB/s) | 5.15 ms (6.4 GB/s) |
| `Q6_K` | 5.46 ms (8.8 GB/s) | 7.24 ms (6.7 GB/s) |
| `IQ4_XS` | 7.03 ms (4.4 GB/s) | 7.10 ms (4.4 GB/s) |
| `Q2_K` | 6.61 ms (2.9 GB/s) | 7.25 ms (2.7 GB/s) |
| `IQ2_XS` | 17.52 ms (1.0 GB/s) | 17.89 ms (0.9 GB/s) |

`Q8_0` reaches the host's memory bandwidth; the other formats are bound
by per-run decoding and scale handling (next increments).

Qwen2.5-0.5B-Instruct `Q8_0` GGUF end to end: 73.8 ms per token, TTFT
121.8 ms, load 1.7 s, "Paris". The first run took 5.8 s per token: the
Runtime reads every node's inputs through `read_tensor_value`, which
expanded each packed weight to f32 per step; answering `Opaque`, as the
CUDA Provider does, removed it. With int8 activations the tokens diverge
from the CUDA one-GPU run at the sixth token (the same near-tie seen in
tensor parallel: "Paris. The capital of France is Paris ..."); with
`MAGNETAR_CPU_INT8_ACTIVATIONS=0` they equal the CUDA f32 run.

### On the GPU runner host (run 43107091711, commit `ef50d02`)

Intel Core i7-7700 (4 cores, 8 threads), the container granted 3 threads
(`available_parallelism`), first-increment Kernels:

| Model | TPOT | TTFT | Load |
|---|---|---|---|
| Qwen2.5-0.5B `Q8_0` | 62.3 ms | 109 ms | 3.5 s |
| Mistral 7B `Q4_K_M` | 1,368 ms | 2,143 ms | 24.0 s |

Both answer "Paris". Mistral's profile: matmul 4.4 ms and matmul-residual
6.0 ms per call (about 3.2 GB/s over the 4.4 GB of weights), and the
embedding 254 ms per call: its `Q4_K` table is expanded at load and the
Reference CPU Kernel copied the whole f32 table per lookup, now gathered
in place.

With the in-place f32 embedding and the faster Kernels (run 43112709258,
commit `4d66c11`), same host and 3 threads:

| Model | TPOT | TTFT | Load |
|---|---|---|---|
| Qwen2.5-0.5B `Q8_0` | 54.8 ms (was 62.3) | 95.8 ms | 3.5 s |
| Mistral 7B `Q4_K_M` | 885.8 ms (was 1,368) | 1,655 ms | 24.6 s |

Both answer "Paris" and generate the same token ids as their CUDA runs.
Mistral's embedding now costs 0.014 ms per lookup (254 ms before); its
matmuls (3.5 ms per call, matmul-residual 4.8 ms) are about 96 % of a
token, near 5 GB/s over the weights on three threads: Kernel-bound, the
next target, with attention (0.57 ms per call) the next operator.

With the specialized `Q4_K` and `Q6_K` Kernels (run 43164691219, commit
`f7aa61e`), same host and 3 threads:

| Model | TPOT | TTFT | Load |
|---|---|---|---|
| Qwen2.5-0.5B `Q8_0` | 53.5 ms | 93.6 ms | 3.5 s |
| Mistral 7B `Q4_K_M` | 379.0 ms (was 885.8) | 1,014 ms | 23.9 s |

Both answer "Paris" and generate the same token ids as their CUDA runs.
Mistral reads its 4.4 GB of weights at about 11.6 GB/s; its matmuls
(1.4 ms per call, matmul-residual 2.0 ms) are about 91 % of a token and
attention (0.57 ms per call, the Reference CPU kernel) about 5 %.

Specialized AVX2 Kernels for `Q2_K`, `Q3_K`, `IQ4_XS` and `IQ4_NL`
(`magnetar-provider-cpu` commit `cf96c3f`), on the model of `Q4_K` and
`Q6_K`: scales decoded once per super-block, quant bytes decoded straight
into registers (`Q2_K`/`Q3_K` load each half's 32 bytes once for its four
runs; `Q3_K` stays unsigned 0..7 with its −4 offset applied through the
activation sums; the IQ4 formats map two runs of indices through the
table with `vpshufb`). Local 4 cores, Mistral-shaped matmuls
(`packed_matmul_bandwidth`):

| Format | `[1, 4096] x [4096, 14336]` | `[1, 14336] x [14336, 4096]` | Before |
|---|---|---|---|
| `Q2_K` | 2.78 ms (6.9 GB/s) | 2.50 ms (7.7 GB/s) | about 2.5-3 GB/s |
| `Q3_K` | 4.56 ms (5.5 GB/s) | 3.71 ms (6.8 GB/s) | generic run decoder |
| `IQ4_XS` | 2.76 ms (11.3 GB/s) | 3.05 ms (10.2 GB/s) | about 5-5.7 GB/s |
| `IQ4_NL` | 3.16 ms (10.4 GB/s) | 3.15 ms (10.5 GB/s) | generic run decoder |

`Q8_0` already reads at 17-19 GB/s through the generic path and keeps it;
the IQ2/IQ3 grids (about 1 GB/s) remain on the generic decoder. The GPU
smoke `gguf` suite now also runs Mistral `Q3_K_M` and `IQ4_XS` on the
optimized CPU Provider.

GPU smoke run 43177850896 (commit `46e8853`), same host, 3 threads:

| Mistral 7B | CPU TPOT | CPU matmul / matmul-residual per call | CUDA TPOT |
|---|---|---|---|
| `Q4_K_M` | 381.4 ms | 1.43 / 2.04 ms | 17.5 ms |
| `IQ4_XS` | 508.6 ms | 2.24 / 1.99 ms | 17.5 ms |
| `Q3_K_M` | 881.4 ms | 2.87 / 6.35 ms | 16.2 ms |

All three answer "Paris" (`Q3_K_M`'s token ids differ from CUDA's after
the first sentence, a warning). Neither export is all one format: the GGUF
headers show `IQ4_XS` keeps `attn_v` in `Q5_K` in all 32 layers, and
`Q3_K_M` keeps `attn_output` and 30 `ffn_down`/`attn_v` in `Q4_K` but two
`ffn_down` and two `attn_v` in `Q5_K`. `Q5_K` is not a packed format yet,
so those matrices are dequantized to `f32` at load and multiplied dense:
they explain the gap to `Q4_K_M` (and the extra resident memory) rather
than the new Kernels. Packing `Q5_K` (runtime format, CPU and CUDA
Kernels) is the next step for these mixes.

`Q5_K` is now a packed format (Runtime `PackedWeightFormat::Q5K`, its
dequantization bit-exact against gguf-py 0.19.0): the optimized CPU
Provider decodes it with Q4_K's layout plus the 32-byte high-bit plane as
each value's fifth bit (AVX2 kernel: 4.32 ms / 9.3 GB/s and 3.96 ms /
10.2 GB/s on the two Mistral shapes, 4 local cores, against the f32 dense
product before), and CUDA reads it directly too (dequantize, f32 and
`__dp4a` decode Kernels). The `Q3_K_M` and `IQ4_XS` mixes therefore keep
every quantized matrix packed.

GPU smoke run 43612369661 (commit `5d2352e`, `Q5_K` packed), same host,
3 threads; the CUDA `Q5_K` conformance test passes:

| Mistral 7B | CPU TPOT | CPU matmul / matmul-residual per call | CUDA TPOT |
|---|---|---|---|
| `Q4_K_M` | 379.0 ms | 1.42 / 2.03 ms | 17.5 ms |
| `IQ4_XS` | 395.3 ms (was 508.6) | 1.50 / 1.94 ms | 16.5 ms (was 17.5) |
| `Q3_K_M` | 602.8 ms (was 881.4) | 2.86 / 1.87 ms | 15.1 ms (was 16.2) |

All answer "Paris". With `Q5_K` packed, `IQ4_XS` runs at `Q4_K_M`'s pace
and `Q3_K_M`'s matmul-residual (its `Q4_K`/`Q5_K` `attn_output` and
`ffn_down`) drops from 6.35 to 1.87 ms. `Q3_K_M`'s plain matmuls
(`Q3_K` `q`/`k`/`gate`/`up`) stay at 2.86 ms per call: the `Q3_K` kernel
(about 5 GB/s locally, half of `Q4_K`) is now its limit.

The `Q3_K` kernel now unpacks each super-block's sixteen scales with SSE
and places each run's bits with constant shifts: a Mistral-shaped matmul
on 4 local cores drops from about 4.5 ms (5.5 GB/s) to 2.4-3.1 ms
(8-10.5 GB/s), level with `Q4_K`.

GPU smoke run 43620986794 (commit `435bad2`), same host, 3 threads:

| Mistral 7B | CPU TPOT | CPU matmul / matmul-residual per call |
|---|---|---|
| `Q4_K_M` | 373.1 ms | 1.39 / 2.00 ms |
| `IQ4_XS` | 400.3 ms | 1.53 / 1.97 ms |
| `Q3_K_M` | 375.8 ms (was 602.8) | 1.48 (was 2.86) / 1.88 ms |

All answer "Paris". `Q3_K_M` now runs at `Q4_K_M`'s pace on the CPU,
although it reads less (3.5 against 4.4 GB of weights: about 9.4 and
11.8 GB/s per token), so the low-bit decode still costs more per byte
than `Q4_K`'s.

The IQ2/IQ3 formats get a vectorized kernel: each run's four grid entries
are gathered into one AVX2 register and its sign bytes expanded into a
byte mask (shuffle, AND, compare) that flips the activations, instead of a
scalar loop over every value. Mistral-shaped matmuls on 4 local cores:
`IQ2_XXS`/`XS`/`S` from 15-17 ms (about 1 GB/s) to 2.8-5.7 ms
(3-6.7 GB/s), `IQ3_XXS`/`S` from 14-17 ms to 3.1-5.2 ms (4.3-8.2 GB/s).
The GPU smoke `gguf` suite now also runs Mistral `IQ3_M` and `IQ2_M` on
the optimized CPU Provider.

GPU smoke run 43638297268 (commit `f43947f`; its CUDA build step failed on
a stale CUDA pin's lockfile, fixed in `2687d39`, but every CPU step ran),
same host, 3 threads, Mistral 7B on the optimized CPU Provider:

| Export | CPU TPOT | matmul / matmul-residual per call | CUDA TPOT |
|---|---|---|---|
| `Q4_K_M` | 374.9 ms | 1.41 / 2.01 ms | 17.4 ms |
| `IQ3_M` | 581.5 ms | 2.36 / 2.71 ms | 17.7 ms |
| `IQ2_M` | 485.2 ms | 1.81 / 2.61 ms | 17.7 ms |

Both answer "Paris". With the scalar decoder these exports' matmuls ran at
about 1 GB/s (14-17 ms per Mistral-shaped matmul locally against 3 ms for
`Q4_K`), so the vectorized grid kernels bring them within 1.3-1.6x of
`Q4_K_M` despite reading fewer bytes: the per-run grid lookups remain
their cost.

The optimized CPU Provider runs `attention` natively: K and V are read in
place up to `key_count` (the Reference CPU kernel copied both whole
caches per call), every (head, query) pair goes to the pool, and the dot
products and weighted value sums vectorize. Mistral-shaped decode step
over a 2048-row cache, 4 local cores: 64 keys 6.2 ms to 0.07 ms, 2048
keys 121 ms to 2.4 ms.

GPU smoke run 43649106260 (commit `23580a5`), same host, 3 threads: the
native attention costs 0.067 ms per call on Mistral 7B (0.56 ms through
the Reference CPU kernel) and 0.027 ms on Qwen2.5-0.5B (0.116 ms). TPOT:

| Model | TPOT (before) |
|---|---|
| Qwen2.5-0.5B `Q8_0` | 51.7 ms (53.5) |
| Mistral 7B `Q4_K_M` | 358.3 ms (375) |
| Mistral 7B `Q3_K_M` | 358.6 ms (376) |
| Mistral 7B `IQ4_XS` | 374.6 ms (400) |
| Mistral 7B `IQ3_M` | 558.6 ms (582) |
| Mistral 7B `IQ2_M` | 468.6 ms (485) |

All answer "Paris". Attention is now under 1 % of a Mistral token; the
matmuls remain the cost.

The IQ2_XXS, IQ2_XS and IQ3_XXS kernels computed each group's eighth sign
bit with `count_ones` (emulated: the kernels' target features exclude
`popcnt`); a 128-entry table built at compile time replaces it. Local
Mistral-shaped matmuls, 4 cores, noisy host: `IQ2_XXS` about 4.6 to
3.1 ms, `IQ3_XXS` about 4.5 to 3.5 ms, `IQ2_XS` about 5.3 to 3.8 ms, with
`Q4_K` at 2.6-3.3 ms on the same runs.

GPU smoke run 43653339471 showed no change for Mistral `IQ3_M` (563.9 ms)
and `IQ2_M` (467.3 ms): their GGUF headers show they are mostly `IQ3_S`
and `IQ2_S`, which have no parity-coded signs. Their cost was the grid
lookups: the kernels now fetch a run's magnitudes with one AVX2 gather
(`vpgatherdd` for IQ3's eight 32-bit entries, `vpgatherdq` for IQ2's four
64-bit ones) from indices built in vector registers. Every IQ2/IQ3
format then runs a Mistral-shaped matmul in 2.0-2.5 ms on 4 local cores
(3-5.7 ms before), below `Q4_K`'s 2.3-3.0 ms on the same runs.

GPU smoke run 43684729957 (commit `2458d29`), same host, 3 threads,
Mistral 7B on the optimized CPU Provider:

| Export | TPOT (before) | matmul / matmul-residual per call |
|---|---|---|
| `Q4_K_M` | 361.4 ms (358.7) | 1.42 / 2.02 ms |
| `IQ3_M` | 433.8 ms (563.9) | 1.78 / 2.16 ms |
| `IQ2_M` | 441.7 ms (467.3) | 1.77 / 2.32 ms |

All answer "Paris". On this host (i7-7700) the gathered IQ kernels stay
about 25 % above `Q4_K` per matmul call, where the local host measured
them level or below; why the two hosts differ is not measured yet.

Why `IQ3_M` and `IQ2_M` trail `Q4_K_M` on the GPU runner (GPU smoke run
43688842289, CPU packed matmul bandwidth diagnostic, 3 threads): not a
Gather Data Sampling mitigation, which that i7-7700 does not carry
("Vulnerable: No microcode"). Gathers are faster there too: `IQ3_S` reads
its grids in 3.5 ms with gathers against 5.3-5.5 ms with scalar loads,
`IQ2_S` 3.3 against 3.9-4.3 ms, while `Q4_K` takes 2.4-2.6 ms. The
remaining gap is the IQ decode's per-run work (grid gather, sign mask,
two scales) on that core; the scalar-load path tried for it was
reverted.
