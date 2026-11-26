## ADDED Requirements

### Requirement: B08 CPU SIMD and locality execution

The CPU Provider SHALL dispatch qualified AVX2, AVX-512 or NEON kernels only after runtime feature checks and SHALL bound threading and account for bandwidth and NUMA locality.

#### Scenario: B08 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** the same model runs on scalar and available SIMD paths
- **THEN** correctness, thread count, affinity policy, bandwidth and end-to-end latency are reported

#### Scenario: B08 Ineligible or failed execution

- **WHEN** a binary runs on a CPU missing the compiled optimization feature
- **THEN** dispatch uses a safe scalar/reference path without illegal instructions or oversubscription

### Requirement: cpu-simd-execution promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
