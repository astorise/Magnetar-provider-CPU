## Why

Magnetar's change `load-specialized-cpu-kernels` (https://github.com/astorise/Magnetar/tree/main/openspec/changes/load-specialized-cpu-kernels) spans the Magnetar core and this repository. This companion change holds the part that modifies this repository's `cpu-provider`, `cpu-kernel-abi` specification, next to the specification itself. The change's proposal, design and tasks stay in Magnetar, since they also cover work outside this repository.

## What Changes

- The requirements the delta in `specs/` adds or modifies.

## Impact

- This repository's `cpu-provider`, `cpu-kernel-abi` specification. The rest of `load-specialized-cpu-kernels` is tracked in Magnetar.
