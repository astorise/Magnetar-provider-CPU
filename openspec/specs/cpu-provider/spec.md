# cpu-provider Specification

## Purpose
This specification defines the reference CPU Provider and kernel baseline used to prove portable local inference without GPU or external service dependencies.
## Requirements
### Requirement: Reference CPU Provider

Magnetar SHALL define a Reference CPU Provider as a correctness-first baseline
Provider for inference execution.

#### Scenario: Register Reference CPU Provider

Given Runtime starts with Reference CPU enabled

When Provider registration runs

Then Runtime registers the Reference CPU Provider if policy allows it.

---

### Requirement: Reference CPU Provider Is Not Optimized Provider

Reference CPU Provider SHALL prioritize correctness and conformance over
performance.

#### Scenario: Slow implementation

Given a reference attention kernel is slow

When conformance validates correctness

Then it remains acceptable for the reference baseline.

---

### Requirement: Reference CPU Provider Is Not Model Architecture

Reference CPU Provider SHALL not represent model families.

#### Scenario: Qwen graph on CPU

Given a Qwen Model Component emits portable Operators

When Runtime executes on CPU

Then Reference CPU Provider executes compatible Operators and is not a
QwenProvider.

---

### Requirement: Reference CPU Provider Identity

Reference CPU Provider SHALL expose stable Provider identity.

#### Scenario: Inspect provider

Given Reference CPU Provider is registered

When Runtime lists Providers

Then stable redacted provider metadata is returned.

---

### Requirement: Reference CPU Device

Reference CPU Provider SHALL expose at least one CPU Device.

#### Scenario: CPU Device available

Given Reference CPU Provider is ready

When Runtime lists Devices

Then at least one CPU Device is visible through Runtime-owned metadata.

---

### Requirement: Host Memory Execution

Reference CPU Provider SHALL primarily operate on host memory.

#### Scenario: Device memory input

Given an input tensor is Device-only

When Reference CPU Kernel is considered

Then Runtime requires explicit movement or rejects dispatch according to policy.

---

### Requirement: Reference CPU Layout Support

Reference CPU Provider SHALL declare supported layouts explicitly.

#### Scenario: Paged layout unsupported

Given an invocation requires paged KV layout

And Reference CPU Provider does not support it

When Kernel selection runs

Then the CPU Kernel is not selected.

---

### Requirement: Reference CPU DType Support

Reference CPU Provider SHALL declare dtype support explicitly.

No silent dtype conversion SHALL occur.

#### Scenario: BF16 unsupported

Given BF16 execution is requested

And Reference CPU Provider supports only f32 compute

When dispatch is planned

Then Runtime rejects or inserts explicit dtype conversion according to policy.

---

### Requirement: Reference CPU Kernel Advertisements

Reference CPU Provider SHALL advertise only implemented Kernels.

#### Scenario: Missing kernel

Given GELU is not implemented

When Runtime queries Kernel Registry

Then no GELU CPU Kernel is assumed unless advertised.

---

### Requirement: Reference CPU Matmul Baseline

Reference CPU Provider SHALL implement correctness-first matmul for supported
dtypes and layouts.

#### Scenario: Matmul known output

Given a small matmul fixture

When CPU matmul runs

Then output matches the reference expectation within tolerance.

---

### Requirement: Reference CPU Embedding Baseline

Reference CPU Provider SHALL implement correctness-first embedding lookup.

#### Scenario: Token out of range

Given token ID exceeds vocabulary size

When embedding runs

Then execution fails with a structured error.

---

### Requirement: Reference CPU RMSNorm Baseline

Reference CPU Provider SHALL implement correctness-first RMSNorm.

#### Scenario: RMSNorm fixture

Given RMSNorm input, weight, and epsilon

When CPU RMSNorm runs

Then output matches expected values within tolerance.

---

### Requirement: Reference CPU RoPE Baseline

Reference CPU Provider SHALL implement or define a valid placeholder for RoPE.

#### Scenario: Unsupported RoPE variant

Given a RoPE variant is unsupported

When dispatch is planned

Then Runtime returns structured unsupported error.

---

### Requirement: Reference CPU Attention Baseline

Reference CPU Provider SHALL implement correctness-first attention where in
scope.

#### Scenario: Causal attention

Given causal attention fixture

When CPU attention runs

Then future tokens are masked according to causal semantics.

---

### Requirement: Reference CPU Softmax Baseline

Reference CPU Provider SHALL implement numerically stable softmax where
feasible.

#### Scenario: Softmax axis

Given softmax request has invalid axis

When validation runs

Then execution fails with structured shape or attribute error.

---

### Requirement: Reference CPU Activation Baseline

Reference CPU Provider SHALL implement supported activation kernels such as
SiLU.

#### Scenario: SiLU fixture

Given input tensor values

When CPU SiLU runs

Then output matches expected values within tolerance.

---

### Requirement: Reference CPU Elementwise Baseline

Reference CPU Provider SHALL implement supported elementwise kernels such as
add, mul, and residual-add.

#### Scenario: Shape mismatch

Given incompatible elementwise shapes

When CPU add runs

Then execution fails with structured shape error.

---

### Requirement: Reference CPU Explicit Conversion

Reference CPU Provider SHALL avoid silent dtype or layout conversion.

#### Scenario: Conversion required

Given Kernel requires f32 but input is f16 storage

When Runtime permits conversion

Then graph planning inserts explicit dtype conversion.

---

### Requirement: Reference CPU Quantization Placeholder

Unsupported quantization features SHALL fail explicitly.

#### Scenario: Unsupported quantized matmul

Given quantized matmul is requested

And no CPU quantized kernel is advertised

When selection runs

Then Runtime returns kernel candidate not found or quantization unsupported.

---

### Requirement: Reference CPU Memory Manager Integration

Reference CPU Provider SHALL use Runtime resource references and Memory Manager
accounting for Runtime-visible resources.

#### Scenario: Output tensor

Given CPU Kernel writes output

When dispatch completes

Then Memory Manager records output readiness and host residency.

---

### Requirement: Reference CPU Registry Integration

Reference CPU Kernels SHALL enter use through Kernel Registry validation.

#### Scenario: CPU kernel advertised

Given Reference CPU Provider advertises matmul

When Runtime validates advertisement

Then matmul becomes a Kernel Registry candidate.

---

### Requirement: Reference CPU Runtime Dispatch

Reference CPU Kernels SHALL execute only Runtime-created Kernel Invocations.

#### Scenario: Direct call denied

Given a Component attempts to invoke CPU Kernel directly

When Runtime enforces boundary

Then direct invocation is denied.

---

### Requirement: Reference CPU Fallback Is Explicit

Reference CPU fallback SHALL be used only when Runtime policy allows it.

#### Scenario: Fallback denied

Given optimized Provider Kernel is unavailable

And CPU fallback is disabled

When Kernel selection runs

Then Runtime rejects instead of silently using CPU.

---

### Requirement: Reference CPU Conformance Role

Reference CPU Provider SHALL support conformance fixtures for portable Operator
semantics.

#### Scenario: Generate reference output

Given a small matmul fixture

When conformance runs

Then Reference CPU Provider may generate or validate reference output.

---

### Requirement: Browser-Compatible CPU Provider Contract

Reference CPU Provider contract SHALL be platform-neutral but SHALL not require
browser implementation.

#### Scenario: Browser target

Given browser Runtime lacks Reference CPU Provider

When CPU Provider is requested

Then Runtime reports reference-cpu-browser-feature-unsupported or provider
unavailable.

---

### Requirement: Reference CPU Error Categories

Reference CPU Provider failures SHALL use structured error categories.

#### Scenario: Unsupported layout

Given a CPU Kernel receives unsupported layout

When validation runs

Then Runtime returns reference-cpu-layout-unsupported or kernel-layout-unsupported.

---

### Requirement: Reference CPU Observability

Runtime SHALL emit Reference CPU observations.

Observability SHALL not expose raw tensor values, prompts, model weights, KV
cache contents, memory pointers, Provider handles, or Device handles by default.

#### Scenario: CPU dispatch failed

Given CPU Kernel dispatch fails

When observability records it

Then Runtime emits redacted structured error metadata.

### Requirement: Reference CPU Covers Required-Now Operators

Reference CPU Provider SHALL provide or explicitly fail validation for
required-now operators.

#### Scenario: Required kernel missing

Given `softmax` is required-now

And no CPU softmax kernel is advertised

When first scope validation runs

Then Runtime reports first-scope-kernel-missing.

---

### Requirement: Reference CPU Does Not Advertise Placeholders As Implemented

Reference CPU Provider SHALL not advertise placeholder operators as implemented
unless they are truly implemented.

#### Scenario: Placeholder advertised falsely

Given `paged-attention` is placeholder

When CPU Provider advertises it as implemented

Then Runtime validates metadata and conformance before accepting it.

---

### Requirement: Reference CPU Uses F32 Baseline

Reference CPU Provider SHALL prioritize f32 compute for the first operator
scope.

#### Scenario: F32 matmul

Given f32 matmul graph

When CPU Provider dispatches it

Then dispatch uses f32-compatible baseline kernel.

### Requirement: Reference CPU Uses Host Contiguous Tensor Resources

Reference CPU Provider SHALL support host contiguous Tensor Resources for the first operator implementation scope.

#### Scenario: CPU matmul input

Given CPU matmul receives host contiguous f32 tensors

When dispatch runs

Then validation succeeds if shapes are compatible.

---

### Requirement: Reference CPU Rejects Unsupported Tensor Layouts

Reference CPU Provider SHALL reject unsupported layouts unless Runtime plans an explicit conversion.

#### Scenario: Blocked input

Given CPU Kernel receives blocked layout

When no conversion is planned

Then dispatch fails with layout unsupported.

---

### Requirement: Reference CPU Rejects Unsupported Tensor Memory Classes

Reference CPU Provider SHALL reject unsupported memory classes unless Runtime plans explicit movement.

#### Scenario: Device memory input

Given CPU Kernel receives Device-only memory

When no host movement is planned

Then dispatch fails with memory class unsupported.

---

### Requirement: Reference CPU May Execute Qwen Baseline Graphs

Reference CPU Provider SHALL be permitted to execute Qwen baseline graphs when
all required Operators have compatible CPU kernels.

#### Scenario: CPU Qwen smoke path

Given Qwen graph uses required-now operators

And Reference CPU kernels are available

When Runtime dispatches the graph

Then execution may proceed on Reference CPU.

---

### Requirement: Reference CPU Missing Coverage Fails Explicitly

If Reference CPU lacks a required Qwen baseline operator, Runtime SHALL fail
explicitly.

#### Scenario: Missing RoPE

Given Qwen graph requires RoPE

And Reference CPU lacks RoPE kernel

When Kernel Registry selection runs

Then Runtime returns structured missing kernel error.

---

### Requirement: E2E Executes Through Reference CPU Provider

The first E2E local inference suite SHALL execute supported inference through
Reference CPU Provider.

#### Scenario: Reference CPU dispatch

Given required operators have Reference CPU Kernels

When E2E generation runs

Then Kernel Dispatch invokes Reference CPU kernels through Runtime.

---

### Requirement: E2E CPU Execution Is Not Hidden Fallback

Reference CPU use in E2E SHALL be explicit and policy-controlled.

#### Scenario: CPU fallback hidden

Given E2E expects explicit CPU policy

When Runtime uses CPU without policy

Then E2E fails with boundary or fallback violation.

---

### Requirement: Reference CPU Provider Implemented Before E2E

Reference CPU Provider baseline SHALL be implemented before E2E local inference
success path.

#### Scenario: CPU execution

Given required-now operators are used

When E2E runs

Then Reference CPU Provider supplies compatible kernels.

---

### Requirement: Reference CPU Prioritizes Correctness

Reference CPU implementation SHALL prioritize deterministic correctness over
performance for the baseline.

#### Scenario: Slow attention

Given CPU attention is slow

When output is correct for fixture

Then baseline acceptance may pass.

---

### Requirement: Reference CPU Remains Baseline After Optimized CPU

Optimized CPU Provider SHALL not replace Reference CPU as correctness baseline.

#### Scenario: Optimized CPU added

Given optimized CPU Provider is available

When conformance generates reference outputs

Then Reference CPU remains the baseline unless policy explicitly selects another
approved reference.

---

### Requirement: Optimized CPU Provider May Add Performance Features

Any performance feature added by optimized CPU Provider SHALL preserve portable Operator semantics.
Optimized CPU Provider MAY add SIMD, BLAS, thread pools, cache-aware kernels, and
fused kernels.

#### Scenario: SIMD matmul

Given optimized CPU Provider advertises SIMD matmul

When Runtime validates it

Then it must still implement portable matmul semantics.

### Requirement: Reference CPU Release Gate Required

Reference CPU Provider gate SHALL be required for `v0.1`.

#### Scenario: Reference CPU absent

Given Reference CPU Provider is unavailable

When release validation runs

Then stable release is blocked.

---

### Requirement: Reference CPU Correctness Report

Release report SHALL include Reference CPU conformance status.

#### Scenario: CPU report

Given conformance runs

When release report is generated

Then Reference CPU status is included.

### Requirement: Optimized CPU Provider Runs Packed Q4_0 Weights

The optimized CPU Provider SHALL accept packed `Q4_0` weights, and its matmuls over them SHALL match the Reference CPU over their dequantized values.

#### Scenario: Q4_0 matmul
- **WHEN** a matmul reads a packed `Q4_0` weight with int8 activations
- **THEN** its result matches the Reference CPU's on the rounded activations and the dequantized weight

### Requirement: CPU Providers Run The Layered And Latent-Attention Forms

The Reference CPU Provider and the optimized CPU Provider of `providers/cpu` SHALL implement and advertise the optional operator forms the Runtime's own Reference CPU implements: the embedding `scale`; per-head and weightless `rmsnorm`; partial and YaRN `rope`; the attention `scale` and `value_head_dimension`; `concat` by heads; `moe-experts` with its activation and per-expert scale; `row-scale`; the `sigmoid`, `softcap` and `scale` activations; and a broadcasting `mul`. The optimized CPU Provider's own attention and embedding Kernels SHALL honour the forms they read, or leave the invocation to the Reference CPU Kernel.

#### Scenario: Tiny checkpoints against transformers
- **WHEN** the tiny Gemma 4 and DeepSeek-V2 checkpoints run on either CPU Provider, the optimized one with f32 activations
- **THEN** every step's top-8 logits match the `transformers` references within 1e-3, with identical tokens

### Requirement: CPU Providers Hold TurboQuant KV Stores

The Reference CPU Provider and the optimized CPU Provider SHALL create TurboQuant KV stores, SHALL keep each store's offsets, and SHALL write each `kv-write` row as its TurboQuant dequantization with offsets, so that their attention defines the reference result over a quantized history.

#### Scenario: Rounded rows
- **WHEN** a row is written into a `tq4` store
- **THEN** reading the store returns that row's TurboQuant dequantization with the store's offsets
