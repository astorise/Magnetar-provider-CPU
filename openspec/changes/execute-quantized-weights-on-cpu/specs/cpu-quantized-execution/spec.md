## ADDED Requirements

### Requirement: Optimized CPU Provider identity

An optimized CPU Provider SHALL have a Provider identity, Device and kernel advertisements distinct from the Reference CPU Provider, SHALL advertise the instruction-set capabilities it detected at run time, and SHALL leave the Reference CPU Provider's contract and role as conformance oracle unchanged.

#### Scenario: Both CPU Providers registered

- **WHEN** a Runtime registers the Reference CPU Provider and the optimized CPU Provider
- **THEN** Resolution can tell them apart by identity and capabilities, and conformance suites keep using the Reference CPU Provider as the oracle

### Requirement: Packed weights stored on the CPU

The optimized CPU Provider SHALL store packed weights it can compute from, in every packed format it advertises, as their block bytes (reorganized only for alignment) without expanding them to f32, SHALL answer `Declined` for any other format, and SHALL decline all packed weights when its packed-weight disable control is set.

#### Scenario: A Q4_K_M checkpoint on the CPU

- **WHEN** a 7B `Q4_K_M` GGUF checkpoint is loaded for the optimized CPU Provider
- **THEN** its `Q4_K` and `Q6_K` projections are stored packed and the instance's host memory stays within its file size plus activations and KV cache

#### Scenario: Disable control

- **WHEN** the packed-weight disable control is set
- **THEN** the Provider declines packed weights and Model Loading writes the dequantized weights, as before this change

### Requirement: Quantized matmuls with int8 activations on the CPU

For decode and short prefill rows, the optimized CPU Provider SHALL multiply packed weights by activations quantized per 32-value block to signed 8-bit values with one f32 scale (`d = max |x| / 127`, round to nearest even), accumulating integer block products, with plain, bias, residual and gated (`silu(gate) * up`) forms. Its result SHALL equal the Reference CPU's on the rounded activations within f32 summation-order tolerance. With the int8-activation disable control set, it SHALL match the Reference CPU on f32 activations.

#### Scenario: Kernel conformance

- **WHEN** a random packed weight of each advertised format is multiplied by 1, 3, 9 and 65 activation rows
- **THEN** rows within the tile match the Reference CPU on the rounded activations, longer ones on the f32 activations

### Requirement: Instruction-set dispatch with a scalar oracle

The optimized CPU Provider SHALL select its kernels at run time from the instruction sets the processor reports, SHALL provide a portable scalar implementation of every kernel, and SHALL let an instruction-set cap force a lower level. Every vectorized kernel SHALL produce the same integer block sums as the scalar kernel.

#### Scenario: Forced levels agree

- **WHEN** the same matmul runs with the cap at scalar and at each available vector level
- **THEN** the outputs agree within f32 summation-order tolerance

### Requirement: Bounded threading

The optimized CPU Provider SHALL split packed matmuls across a persistent pool of worker threads bounded by the available parallelism and by a thread-count control, and SHALL NOT spawn threads per invocation.

#### Scenario: Thread bound

- **WHEN** the thread-count control is 2 on an 8-core host
- **THEN** at most two workers compute matmuls and decode still completes

### Requirement: CPU quantized execution qualification

The optimized CPU Provider SHALL NOT be reported production-validated for a format until a real checkpoint in that format generates on real CPU hardware with the answer checked, tokens compared with the CUDA int8 run, and TPOT, TTFT, load time and peak host memory recorded alongside the bandwidth bound of the host.

#### Scenario: Evidence is incomplete

- **WHEN** only fixtures, a scalar run on a different host, or skipped instruction-set levels are available
- **THEN** the support matrix does not claim the format production-validated on the CPU
