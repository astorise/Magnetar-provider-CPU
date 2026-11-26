## 1. Scope and compatibility

- [ ] 1.1 Reconcile B08 with the existing contracts linked in proposal.md and record the exact initial supported profile.
- [ ] 1.2 Confirm upstream exit-gate dependencies, hardware access, checkpoint authorization, numeric/quality thresholds and memory/SLO budgets.
- [ ] 1.3 Prefer current WIT; if new portable operators/state/results are needed, design a versioned additive extension, migration and unsupported behavior before implementation.

## 2. B08 CPU SIMD and locality execution

- [ ] 2.1 Implement B08: The CPU Provider SHALL dispatch qualified AVX2, AVX-512 or NEON kernels only after runtime feature checks and SHALL bound threading and account for bandwidth and NUMA locality.
- [ ] 2.2 Add the positive conformance case: when the same model runs on scalar and available SIMD paths, verify correctness, thread count, affinity policy, bandwidth and end-to-end latency are reported.
- [ ] 2.3 Add the failure/isolation case: when a binary runs on a CPU missing the compiled optimization feature, verify dispatch uses a safe scalar/reference path without illegal instructions or oversubscription.

## 3. Qualification, rollout and documentation

- [ ] 3.1 Run affected cargo fmt/check/clippy/tests and conformance in both relevant workspaces and affected extension repositories; keep baseline F32 tests passing.
- [ ] 3.2 Validate any changed WIT and old/new linking, OpenSpec deltas and native/wasm boundaries; test Linux, Windows/macOS and ARM where applicable, with explicit hardware skips.
- [ ] 3.3 Exercise cancellation, failure, memory pressure, invalid artifact/state reuse and tenant/session isolation; prove lease retirement waits for completion.
- [ ] 3.4 Execute versioned before/after real-checkpoint campaigns on appropriate real hardware using shared V0 evidence, recording tail latency, total memory, I/O, numeric and downstream quality. Do not mark skipped hardware passed.
- [ ] 3.5 Check third-party licenses, SBOM, signatures/provenance and artifact reproducibility where native libraries or generated artifacts change.
- [ ] 3.6 Verify targeted disable, safe fallback/Unsupported, pinned mode and generation rollback; document support limits, public architecture/API migration and measurements.
- [ ] 3.7 Update the roadmap coverage/support matrix with evidence and actual maturity only after implementation gates pass; archive only completed scope.
