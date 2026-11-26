Status: **Proposed (RFC lot 3).** No task is implemented yet. Requires lots 0 and 1, and lot 2 for hot swap.

## 1. ABI and format

- [ ] 1.1 `providers/cpu/include/magnetar_cpu_kernel.h` (design D1), with a Rust mirror and layout assertions.
- [ ] 1.2 `magnetar-kernel`: the `magnetar:cpu-elf@1` format identity, the CPU feature bit set, and a target compatibility check against host features.

## 2. Loader (Linux)

- [ ] 2.1 ELF checks on the verified bytes: machine, `ET_DYN`, no initializer sections, no `DT_TEXTREL`, the `DT_NEEDED` allowlist, the exported table symbol.
- [ ] 2.2 In-memory loading (`memfd_create`, write seals, `RTLD_NOW | RTLD_LOCAL`); table and entry checks against the manifest; `dlclose` after drain (lot 2) or at shutdown.
- [ ] 2.3 Excluded from `wasm32` builds; other platforms report the format unsupported.

## 3. CPU Provider

- [ ] 3.1 The preparation capability (lot 1 contract) for `magnetar:cpu-elf@1`: variant matching by features under `MAGNETAR_CPU_SIMD`, fallback to built-in kernels.
- [ ] 3.2 Pool dispatch over task ranges, per-task workspace, structured failure on a non-zero return.
- [ ] 3.3 Step capture and replay of prepared kernels, with generation-carrying keys.

## 4. Fixture and experiment

- [ ] 4.1 C sources for `rmsnorm` and a dense decode `matmul`, built in CI into baseline, AVX2 and AVX-512 variants; bundled and signed by the job key (lot 1 tool).
- [ ] 4.2 Tests:
  - an ABI mismatch, a constructor, a disallowed dependency, a missing symbol;
  - variant selection on each feature cap; no matching variant;
  - GPU-only permission; capture and replay; task-range coverage.
- [ ] 4.3 Differential qualification against the Reference CPU kernel.
- [ ] 4.4 CPU benchmark job:
  - pinned external `rmsnorm` and `matmul` against `cpu-native` on Qwen2.5-0.5B, with identical tokens;
  - TTFT, TPOT, p50/p95 and RSS recorded here.
- [ ] 4.5 The WebAssembly SIMD128 tier on the same shapes (design D5): cost against native variants recorded here.
- [ ] 4.6 `cargo fmt`, `clippy -D warnings`, tests of both workspaces and the CPU Provider's, and the `wasm32` check.

## 5. Documentation

- [ ] 5.1 The CPU Provider README: external kernels, the ABI, the loader checks, the permission, and the explicit limit that in-process native code is not sandboxed.
- [ ] 5.2 `docs/architecture/kernel-artifact-sources.md`: the CPU path.
- [ ] 5.3 RFC question 6: the measured decision input.
