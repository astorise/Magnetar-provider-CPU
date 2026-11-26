## Context

The optimized CPU Provider (`providers/cpu/src/native`) holds weights compact or packed. It runs matmuls with AVX2/AVX-512 kernels selected at runtime (`SimdLevel`, `dense::avx512()`), shares work through a persistent pool sized to the physical cores, and captures and replays decode steps. Every operator it does not optimize runs on the Reference CPU kernel it wraps.

An external CPU kernel must fit that model. The pool calls it on index ranges, compact weights are handed to it as views, and step capture records it.

## Goals / Non-Goals

**Goals:**

- A small, versioned C ABI that external kernels implement, with no Runtime types in it.
- A loader that rejects libraries able to run code at load, or that link outside an allowlist.
- Variants chosen by host features, with a fallback that is tested, not assumed.
- Data to decide between native AOT and a sandboxed tier (RFC question 6).

**Non-Goals:**

- Isolation guarantees for native code, Windows and macOS loading.

## Decisions

### D1. The ABI

```c
#define MAGNETAR_CPU_KERNEL_ABI 1

typedef struct { const void *data; uint32_t dtype; uint32_t rank;
                 const int64_t *shape; const int64_t *strides; } magnetar_tensor_view;
typedef struct { const void *data; uint32_t dtype; uint32_t rank;
                 const int64_t *shape; const int64_t *strides; } magnetar_tensor_view_mut;
typedef struct { const char *name; uint32_t kind; int64_t i; double f; } magnetar_attribute;
typedef struct { uint64_t first; uint64_t count; uint64_t total; } magnetar_task_range;

typedef int32_t (*magnetar_cpu_kernel_fn)(
    const magnetar_tensor_view *inputs, uint32_t input_count,
    const magnetar_tensor_view_mut *outputs, uint32_t output_count,
    const magnetar_attribute *attributes, uint32_t attribute_count,
    void *workspace, uint64_t workspace_bytes,
    magnetar_task_range range);

typedef struct {
    const char *operator_name;      /* semantic binding, e.g. "rmsnorm" */
    uint32_t operator_version;
    uint64_t required_features;     /* MAGNETAR_CPU_FEATURE_* bits */
    uint64_t workspace_bytes_per_task;
    uint64_t (*task_count)(const magnetar_tensor_view *inputs, uint32_t input_count);
    magnetar_cpu_kernel_fn run;
} magnetar_cpu_kernel_entry;

typedef struct {
    uint32_t abi_version;           /* MAGNETAR_CPU_KERNEL_ABI */
    uint32_t entry_count;
    const magnetar_cpu_kernel_entry *entries;
} magnetar_cpu_kernel_table;

const magnetar_cpu_kernel_table *magnetar_cpu_kernel_table_v1(void);
```

- The table is static data. The table function only returns its address.
- `task_count` and `run` are pure functions of their arguments. A non-zero return from `run` is a structured kernel failure.
- Compact bfloat16 weights reach the kernel as a `bf16` view of the stored bits. The stored orientation (`speed-up-dense-cpu-matmuls`) is expressed by strides, so a kernel never assumes a layout the view does not state.

### D2. Loading without running code first

- **ELF checks**, done on the verified bytes before any load:
  - `e_machine` matches the host;
  - `ET_DYN` with no `DT_INIT`, `DT_INIT_ARRAY` or `DT_PREINIT_ARRAY`, and no `DT_TEXTREL`;
  - every `DT_NEEDED` is in the allowlist (`libc.so.6`, `libm.so.6`);
  - `magnetar_cpu_kernel_table_v1` is exported.
- **Loading.** The bytes are written to a `memfd_create` file sealed against writes, opened through `/proc/self/fd/N` with `RTLD_NOW | RTLD_LOCAL`, and the table symbol resolved. The ABI version and every entry's operator, features and workspace are checked against the manifest's semantic binding and specialization.
- **Release.** `dlclose` happens only after the generation drains (lot 2). Without lot 2, libraries are released at Provider shutdown.

These checks stop the common ways a library runs code at load. They do not make native code safe; trust and the execution permission decide whether it runs at all.

### D3. Variants and fallback

- A bundle may carry several `magnetar:cpu-elf@1` artifacts for one semantic binding. The manifest's `target` gives architecture and required features.
- The Provider prepares the most capable variant whose features the host has. `MAGNETAR_CPU_SIMD=avx2` caps it as for built-in kernels.
- If no variant matches, the ELF checks fail, or preparation fails, the operator keeps its built-in kernel. Each path has a test.

### D4. Dispatch and capture

- The Provider calls `task_count`, then runs `run` over the task ranges on its pool, as built-in kernels do. It allocates `workspace_bytes_per_task` per task from its own buffers.
- Step capture records the invocation with the prepared kernel's identity. Replay calls the same function. Capture keys carry the generation (lot 2).

### D5. The sandboxed tier, measured only

- The same entries are compiled to a WebAssembly module with SIMD128, linked to nothing, and run by Wasmtime. Inputs are copied into linear memory and outputs copied back.
- The experiment records its cost against native AVX2 and AVX-512 variants and `cpu-native` on the same shapes. The team decides from those numbers whether a sandboxed tier is worth enabling for untrusted publishers.

## Risks / Trade-offs

- **In-process native code can do anything the process can.** The mitigation is policy (trust plus a distinct permission, denied by default) and load-time reduction, not isolation, and this is stated in the documentation.
- **`memfd` and `/proc` are Linux-only.** Other platforms report the format unsupported until they get their own loader.
- **An external matmul must beat `cpu-native`,** which now streams decode weights at about 40 GB/s locally. The experiment may show that external kernels pay off only for operators the Provider lacks, and that result is recorded.

## Open Questions

- RFC question 6: native AOT only, or a sandboxed tier for publishers without native permission. To be decided after D5's measurements.
