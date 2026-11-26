## Why

Lot 3 of the [accelerator packages RFC](../../../docs/rfcs/accelerator-packages-2026-11-23.md) asks for proof that the protocol does not depend on CUDA. That means a stable invocation ABI for specialized CPU kernels, a policy authorizing native code, at least two conditional variants (a baseline and AVX2/AVX-512) with a tested fallback, and a comparison on a real workload.

The CPU side differs from CUDA in one important way: a CPU kernel runs in the Runtime's own process and address space. A CUBIN runs on a separate device context. A native CPU kernel can read the process's memory, call the system and run code when its library is loaded. Trust (lot 0) and an explicit execution permission are therefore necessary, and the loader must reduce what loading itself can do.

The in-tree optimized CPU Provider (`cpu-native`) already has AVX2 and AVX-512 kernels behind runtime feature detection, a persistent pool and step capture. It is the baseline any external variant must beat.

## What Changes

- **ABI `magnetar:cpu-kernel@1`.** A shared object exports one entry point that returns a versioned table of kernels. Each entry gives:
  - the semantic binding, specialization and required CPU features (a bit set: SSE4.2, AVX2, FMA, AVX-512F/BW/BF16, AMX-BF16, NEON, SVE);
  - a function taking tensor views (pointer, dtype, shape, strides), attributes, a workspace and a task range.

  The kernel computes one task range. The Provider's pool owns the threads: the kernel creates no threads, allocates nothing and keeps no state between calls.
- **Format `magnetar:cpu-elf@1`**: an ELF shared object for `x86_64-linux-gnu` or `aarch64-linux-gnu`. Mach-O and PE are follow-ups.
- **Loader checks before any code runs:**
  - no initializers (`DT_INIT`, `DT_INIT_ARRAY`, `DT_PREINIT_ARRAY`) and no text relocations;
  - dependencies only from an allowlist (`libc`, `libm`);
  - the expected entry symbol present;
  - the matching architecture.

  The library is then loaded from the verified bytes through an anonymous in-memory file, never from a path the manifest names, local and resolved immediately.
- **The CPU Provider** implements lot 1's preparation capability for `magnetar:cpu-elf@1`:
  - it picks, per operator, the variant whose required features the host has;
  - it falls back to its built-in kernels when none matches or preparation fails;
  - prepared kernels run through its pool and appear in step captures like built-in ones.
- **The execution permission** for `magnetar:cpu-elf@1` is distinct from every GPU format and denied by default.
- **The experiment:**
  - a checked-in C source for `rmsnorm` and a dense `matmul` decode shape (`[1, 4864] × [4864, 896]`), built in CI into baseline, AVX2 and AVX-512 variants;
  - each bundled and signed by the job, qualified against the Reference CPU kernel, then pinned and compared with `cpu-native` on Qwen2.5-0.5B in the CPU benchmark job.
- **A sandboxed tier, evaluated only:** the same ABI marshalled into a WebAssembly module (SIMD128) run by Wasmtime. Its cost is measured against native variants so the team can decide RFC question 6. It is not enabled.

## Capabilities

### New Capabilities

- `cpu-kernel-abi`: the `magnetar:cpu-kernel@1` invocation ABI and the `magnetar:cpu-elf@1` format with its loader checks.

### Modified Capabilities

- `cpu-provider`: preparation of external CPU kernels with feature-matched variants and fallback (added requirements).

## Impact

- `providers/cpu`: an ELF inspector, an in-memory loader (Linux), the ABI shim, variant matching, preparation, pool dispatch and capture integration. The wasm32 build excludes the loader.
- `magnetar-kernel`: the format identity, the feature bit set and its compatibility check.
- `tools/kernel-bundle-fixture`: C sources built into variants, bundled and signed (CI only).
- `.github/workflows/benchmark-cpu.yml`: the comparison step.
- No WIT change.

## Dependencies

- Lot 0 ([`define-kernel-artifact-sources`](../define-kernel-artifact-sources/proposal.md)) and lot 1's preparation capability ([`prepare-prebuilt-cuda-kernels`](../prepare-prebuilt-cuda-kernels/proposal.md)).
- Lot 2 for hot swap of CPU kernels ([`serve-kernel-bundles-and-swap-generations`](../serve-kernel-bundles-and-swap-generations/proposal.md)). Without lot 2, prepared CPU kernels are only released at shutdown.

## Compatibility and authority

- The CPU Provider and `cpu-native` behave exactly as today without a permitted, eligible bundle.
- Native loading happens only on Linux in this lot. Other platforms report the format unsupported.
- The loader never runs code from a library before the checks pass. Even then, a loaded kernel runs with the process's authority: this is why trust and a distinct permission are required, and why the checks are a reduction, not a sandbox.

## Non-goals

- Out-of-process isolation, JIT compilation, Mach-O or PE loading.
- Kernels that create threads, allocate, perform I/O or keep state.
- Enabling the WebAssembly tier.
