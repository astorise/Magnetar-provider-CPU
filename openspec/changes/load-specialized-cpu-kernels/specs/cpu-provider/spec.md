## ADDED Requirements

### Requirement: Feature-Matched External CPU Variants

The optimized CPU Provider SHALL prepare, for each operator, the most capable eligible `magnetar:cpu-elf@1` variant whose required features the host provides, within any `MAGNETAR_CPU_SIMD` cap. It SHALL keep its built-in kernel when no variant matches or preparation fails.

#### Scenario: An AVX-512 variant on an AVX2 host

- **WHEN** a bundle offers baseline, AVX2 and AVX-512 variants on a host without AVX-512F
- **THEN** the AVX2 variant is prepared

#### Scenario: No variant matches

- **WHEN** every variant requires AVX-512F and the host lacks it
- **THEN** no external kernel is prepared, and the built-in kernel serves the operator

### Requirement: External CPU Kernels Run On The Pool And In Captures

A prepared external CPU kernel SHALL run over its task ranges on the Provider's pool, with workspace from the Provider. A step capture SHALL record it with its generation, and replay SHALL call the same prepared kernel.

#### Scenario: Captured external kernel

- **WHEN** a decode step with a prepared external `rmsnorm` is captured and replayed
- **THEN** the replay's outputs equal a per-node run's
