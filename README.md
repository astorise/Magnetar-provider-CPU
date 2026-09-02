# magnetar-provider-cpu

## Purpose

The Reference CPU Provider for the
[Magnetar](https://github.com/astorise/Magnetar) local AI Runtime: a
correctness-first, host-visible Kernel execution baseline proving portable
local inference without a GPU or external service dependency. Implements
`ProviderExecutionApi` and Magnetar's Provider contract -- Kernel
advertisement, submission/completion, memory-admitted tensor storage, and
Kernel-level compute for the portable Operator set Magnetar's first-native
execution path uses (matmul, embedding, rmsnorm, rope, attention, and
related tensor operators).

## Status

**Real extraction, not a template.** This crate now holds the actual
`ReferenceCpuExecutor`/`ReferenceCpuProvider` implementation, its numeric
kernels (matmul, attention, rope, rmsnorm, softmax, silu, gelu, dtype/layout
conversion), and SIMD-feature-flag plumbing, ported out of
`magnetar-runtime/src/reference_cpu.rs`. `magnetar-runtime` keeps its own
copy of that file in-crate, deliberately: it is still what magnetar-runtime's
~1000-test suite instantiates directly as a generic test double, and it is
never referenced by this crate or vice versa. The two are independent,
intentionally near-duplicate implementations, not one moved file -- see
"Relationship to magnetar-runtime" below for why a literal move was not
possible yet.

## Governing contract

[`cpu-provider`](https://github.com/astorise/Magnetar/blob/main/openspec/specs/cpu-provider/spec.md)
in the main Magnetar repository's OpenSpec capability set defines this
Provider's baseline requirement: Magnetar SHALL define a Reference CPU
Provider as a correctness-first baseline Provider for inference execution.
The broader `provider` capability spec governs the full
`ProviderExecutionApi`/`Provider` contract this crate implements.

## Relationship to magnetar-runtime

This crate depends only on `magnetar-runtime`'s public Provider/Device/
Kernel/Tensor contracts (`providers/cpu -> magnetar-runtime`, never the
reverse); `magnetar-runtime` compiles and tests cleanly without this crate
present. It is pinned into the main
[Magnetar](https://github.com/astorise/Magnetar) repository as a git
submodule at `providers/cpu`.

One detail keeps this from being a clean, self-contained port: `HostTensor`,
`ReferenceCpuError`, and `ReferenceCpuErrorCode` are *imported* from
`magnetar_runtime` here rather than redefined locally, because
`magnetar-runtime/src/provider.rs`'s `ProviderExecutionApi` trait
(`write_tensor`/`read_tensor`/`write_tensor_admitted`) is still typed
directly against `HostTensor` -- an explicitly documented, provisional
transport ahead of a fully Resource-based rewrite (`magnetar-runtime` task
group 5 / Correctif 5), not yet complete. `HostTensor::new`/`rows_cols`
return `ReferenceCpuError`, and `magnetar-runtime`'s own
`impl From<ReferenceCpuError> for KernelError` cannot be duplicated here
(Rust's orphan rule forbids `impl ForeignTrait for ForeignType` regardless of
which side is "local"), so that error type is pulled in too. This is the
smallest set of types load-bearing enough to force sharing: everything else
in this crate (`ReferenceCpuFeatureFlags`, the kernel functions,
`ReferenceCpuExecutor`, `ReferenceCpuProvider`, SIMD detection, conformance
reporting) is this crate's own, independent of magnetar-runtime's in-crate
copy. Once magnetar-runtime's Resource-based rewrite removes `HostTensor`
from the generic trait boundary, this crate can define its own `HostTensor`/
`ReferenceCpuError` instead of importing magnetar-runtime's.
