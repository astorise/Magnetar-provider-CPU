## ADDED Requirements

### Requirement: Versioned CPU Kernel ABI

A `magnetar:cpu-elf@1` artifact SHALL export `magnetar_cpu_kernel_table_v1`, returning a static table with the ABI version and one entry per kernel. Each entry SHALL give its operator and version, required CPU features, workspace bytes per task, a task-count function and a run function over tensor views, attributes, a workspace and a task range. The Runtime SHALL refuse an artifact whose ABI version, entries or features do not match its manifest.

#### Scenario: An ABI version mismatch

- **WHEN** a library's table reports an ABI version other than 1
- **THEN** preparation fails, and the operator keeps its built-in kernel

### Requirement: Kernels Are Stateless Range Functions

A CPU kernel SHALL compute only the outputs of the task range it is given, from its inputs, attributes and workspace. It SHALL NOT create threads, allocate, perform I/O or keep state between calls, and the Provider's pool SHALL own all parallelism.

#### Scenario: Task ranges cover the output

- **WHEN** the pool runs every task range of a prepared `rmsnorm`
- **THEN** the output equals the Reference CPU kernel's within the qualification tolerance

### Requirement: Load-Time Reduction

Before loading a `magnetar:cpu-elf@1` artifact, the Provider SHALL check that it matches the host architecture, has no initializer sections and no text relocations, links only allowlisted libraries, and exports the table symbol. It SHALL load the library from the verified bytes through an in-memory file, never from a path named by the manifest, and only after trust and execution permission.

#### Scenario: A library with a constructor

- **WHEN** a library has a `DT_INIT_ARRAY` entry
- **THEN** it is rejected before any of its code runs

#### Scenario: A disallowed dependency

- **WHEN** a library needs `libpthread.so.0` or any library outside the allowlist
- **THEN** it is rejected before loading

### Requirement: Native CPU Execution Permission

Execution of `magnetar:cpu-elf@1` artifacts SHALL require its own deployment permission, denied by default and distinct from any GPU format's permission.

#### Scenario: GPU permission only

- **WHEN** a publisher holds `nvidia:cubin` permission for the CUDA Provider only
- **THEN** its CPU library is not loaded
