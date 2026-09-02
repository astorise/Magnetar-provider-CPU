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

**Empty template.** This crate is currently a bare `cargo new --lib`
scaffold. The real, working Reference CPU Provider implementation lives
today inside `magnetar-runtime` itself (`reference_cpu.rs`), as a
deliberate, tracked architectural debt -- not the target state. Extracting
it into this crate is `reach-architecture-freeze-1` task group 14
("Extract Reference CPU into `providers/cpu`"), which cited two
prerequisites: task group 13 (does `ProviderExecutionApi` need a new
contract, or do ordinary additive methods suffice?) and task group 5/task
3.3 (is `HostTensor` still the concrete type `magnetar-runtime`'s *generic*
graph-execution code carries, not just Reference CPU's own Kernel bodies?).
Both are now resolved. Task group 13 resolved without needing a new
contract. `define-provider-prepared-kernel-execution-contract` was opened
for a *different* reason (a Provider-agnostic tensor *value* type,
`TensorValue`) and initially added it additively without migrating
`magnetar-runtime`'s generic dispatch/transport code off `HostTensor` --
but per the post-freeze équipe review's explicit decision to migrate the
generic transport now rather than wait for a real CUDA Provider, that
Change's task group 5 completed the migration: `execute_qwen_graph_nodes`'s
per-node transport reads/writes exclusively through `TensorValue` today,
materializing to `HostTensor` only at four explicit boundaries (weight
binding, KV-history concatenation, final logits extraction, per-node
Kernel-input resolution). `HostTensor` still exists and is still what
Reference CPU's own Kernel bodies compute over -- that is expected and
correct, not a remaining blocker -- but it is no longer required at the
generic `ProviderExecutionApi` trait boundary, so extracting Reference CPU
into this crate would no longer be moving a layering problem across the
repository boundary. Extraction itself (task group 14's actual code move)
has not started yet; both of its prerequisites are simply no longer
outstanding.

## Governing contract

[`cpu-provider`](https://github.com/astorise/Magnetar/blob/main/openspec/specs/cpu-provider/spec.md)
in the main Magnetar repository's OpenSpec capability set defines this
Provider's baseline requirement: Magnetar SHALL define a Reference CPU
Provider as a correctness-first baseline Provider for inference execution.
The broader `provider` capability spec governs the full
`ProviderExecutionApi`/`Provider` contract this crate would implement.

## Relationship to magnetar-runtime

Once extracted, this crate depends only on `magnetar-runtime`'s public
Provider/Device/Kernel/Tensor contracts (`providers/cpu -> magnetar-runtime`,
never the reverse) and registers itself the same way any Provider does --
`magnetar-runtime` compiling and testing cleanly without this crate present
is one of task group 14's own verification requirements. It is pinned into
the main [Magnetar](https://github.com/astorise/Magnetar) repository as a
git submodule at `providers/cpu`.
