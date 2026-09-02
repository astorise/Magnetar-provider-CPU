//! Reference CPU Provider: a correctness-first, host-memory execution baseline
//! for portable Operators. It is not required to be fast; it exists to prove
//! portable semantics before optimized Providers (CUDA, Metal, OpenVINO, QNN,
//! WebGPU) define behavior.
//!
//! Tensor storage is Provider-owned and opaque to the rest of the Runtime, the
//! same way any other Provider's device buffers would be: the Kernel Contract
//! and Memory Manager only see resource identity and accounting, never raw
//! bytes. [`ReferenceCpuExecutor`] keeps the actual host-visible [`HostTensor`]
//! data behind [`TensorResourceId`] keys.
//!
//! # Versus an optimized CPU Provider
//!
//! An optimized CPU Provider (SIMD-vectorized, multi-threaded, quantization-
//! aware, kernel-fused) would advertise the same portable Operators but with
//! richer `KernelPrecisionMetadata`/`KernelDeterminism` (approximate math,
//! fused semantics, hardware-feature-dependent determinism) and would beat
//! Reference CPU on `estimated_cost` during Kernel candidate ranking.
//! Reference CPU exists to define what "correct" means first, independent of
//! any such optimization; it advertises `approximate_math: false` and
//! `deterministic: true` by leaving `KernelPrecisionMetadata`/`KernelDeterminism`
//! at their conservative defaults, and never advertises fused or quantized
//! kernels.
//!
//! # Versus CUDA/Metal/OpenVINO/QNN Providers
//!
//! Hardware-accelerated Providers execute on a different [`DeviceType`]
//! (`Gpu`/`Npu`) with device-resident memory classes and Provider-specific
//! ABI/dynamic-loading concerns (`ProviderAbiDescriptor`, `ProviderLoader::
//! load_dynamic`) that Reference CPU has no need for: it is always built-in
//! (`REFERENCE_CPU_BUILT_IN`), never dynamically loaded, and only ever
//! targets `DeviceType::Cpu`/`KernelMemoryClass::Host`. Reference CPU's
//! numeric outputs are meant to be the correctness baseline those Providers'
//! outputs get compared against within a declared tolerance profile (see the
//! Conformance capability), not a competing execution target.
//!
//! # Versus Model Component
//!
//! A Model Component owns model-level concerns (weights, tokenizer,
//! architecture, generation policy) and is Provider-agnostic; it never picks
//! a Provider or Device directly. Reference CPU sits one layer below: it is
//! one of the Providers a Model Component's execution graph might ultimately
//! be dispatched to (via Runtime-owned resolution), advertising Kernels for
//! individual portable Operators (matmul, attention, ...), never model-level
//! concepts. Reference CPU has no knowledge of Model Components at all — the
//! dependency only ever points from higher layers down to Kernel Contract
//! Providers like this one.
//!
//! # DType support
//!
//! Reference CPU only ever stores `f32` elements. Every advertised kernel
//! declares `Float32` as its only supported input/output dtype; any other
//! portable dtype (or a requested `accumulation_dtype` other than `f32`, for
//! `matmul`) is rejected explicitly via [`dtype_conversion`] and
//! `reference-cpu-dtype-unsupported` rather than silently converted.
//!
//! # Layout support
//!
//! Reference CPU only stores contiguous, row-major tensors. [`TensorLayoutKind::Strided`]
//! targets get a distinct "defined placeholder, not yet implemented" rejection from
//! [`layout_conversion`]; blocked, paged, and other provider-opaque layouts get the
//! generic `reference-cpu-layout-unsupported` rejection. Nothing is ever silently
//! reinterpreted into a different layout.
//!
//! # Attention baseline
//!
//! [`attention`] implements causal or unmasked scaled dot-product attention.
//! It supports grouped-query attention (`kv_head_count` dividing `head_count`,
//! each group of query heads sharing one key/value head) and sliding-window
//! attention (`window_size` restricting each query to its most recent keys).
//! Paged KV cache is not implemented: the `paged-attention` Operator is never
//! advertised, so Runtime never assumes Reference CPU can serve it.
//!
//! # RoPE baseline
//!
//! [`rope`] implements real rotary position embedding (not a stub): it
//! rotates consecutive element pairs within the first `dimension` elements of
//! each row using `position = row index`, `base`, and `scale`. Base and scale
//! must both be finite and positive.
//!
//! # RMSNorm baseline
//!
//! [`rmsnorm`] implements RMSNorm with `f32` accumulation: each row is scaled
//! by `1 / sqrt(mean(x^2) + epsilon)`, then multiplied element-wise by the
//! weight vector. `epsilon` must be positive and the weight width must match
//! the row width.
//!
//! # Browser limitations
//!
//! The Provider, Device, and Kernel contracts here are platform-neutral and
//! build cleanly on `wasm32-unknown-unknown` (`std::thread::available_parallelism`
//! and `std::env::consts::ARCH` both work there). Nothing in this module
//! requires native dynamic-library loading. `reference-cpu-browser-feature-unsupported`
//! is defined in the error model for future use if a browser-specific limitation
//! is identified; none is known today.

use magnetar_runtime::affinity::{
    DeviceBinding, FallbackClass, ProviderBinding, ProviderHealth, ProviderPressureLevel,
    ProviderStatusSnapshot, ResourceAffinity,
};
use magnetar_runtime::capability::{CapabilityId, CapabilityVersion};
use magnetar_runtime::compute::{
    ComputeDType, DTypeDescriptor, ShapeDescriptor, TensorDescriptor, TensorDescriptorLimits,
    TensorResourceDescriptor, TensorResourceId,
};
use magnetar_runtime::device::{
    Device, DeviceDescriptor, DeviceExecutionLimits, DeviceId, DeviceMetadata, DeviceType,
};
use magnetar_runtime::kernel::{
    KernelAdvertisement, KernelCancellationSupport, KernelError, KernelId,
    KernelImplementationFamily, KernelInvocation, KernelKvCacheMetadata, KernelMemoryClass,
    KernelObservation, KernelObservationKind, KernelOperatorVersionRange, KernelResult,
    KernelResultStatus, KernelWorkspaceRequirements,
};
use magnetar_runtime::memory::{
    MemoryAllocationClass, MemoryAllocationId, MemoryAllocationOwner, MemoryAllocationRequest,
    MemoryError, MemoryManager, MemoryPlacement, TensorResidency,
};
use magnetar_runtime::operator::{
    OperatorAttributeValue, OperatorFamily, OperatorId, OperatorSpec, TensorLayoutKind, TensorRole,
};
use magnetar_runtime::provider::{
    PROVIDER_API_VERSION, Provider, ProviderError, ProviderExecutionApi, ProviderMetadata,
    ProviderRegistry, TensorValue,
};
use magnetar_runtime::scheduler::{
    ProviderCancellationOutcome, ProviderExecutionError, ProviderExecutionErrorCode,
    ProviderExecutionHandle, ProviderExecutionId, ProviderExecutionPhase, ProviderExecutionRequest,
    ProviderExecutionResult, ProviderExecutionStatus, ScheduledOperationId, SchedulingState,
};
use magnetar_runtime::tensor::{TensorLifecycleState, TensorReadiness, TensorResource};
use magnetar_runtime::{ExecutionPlanId, HostTensor, ReferenceCpuError, ReferenceCpuErrorCode};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

/// Stable, package-qualified Reference CPU Provider identity.
pub const REFERENCE_CPU_PROVIDER_NAME: &str = "magnetar:provider/reference-cpu";
pub const REFERENCE_CPU_PROVIDER_VERSION: &str = "0.1.0";
pub const REFERENCE_CPU_PROVIDER_VENDOR: &str = "magnetar";
pub const REFERENCE_CPU_DEVICE_ID: &str = "reference-cpu:host:0";
pub const REFERENCE_CPU_CONFORMANCE_PROFILE: &str = "reference-cpu-conformance-v1";
pub const REFERENCE_CPU_KERNEL_FAMILY: KernelImplementationFamily =
    KernelImplementationFamily::CpuScalar;

/// Reference CPU always ships built into the Runtime binary; it is never
/// loaded as an external dynamic library.
pub const REFERENCE_CPU_BUILT_IN: bool = true;

/// Runtime Provider ABI version range this build of Reference CPU was
/// validated against, expressed as `(min, max)` over [`PROVIDER_API_VERSION`].
pub const REFERENCE_CPU_SUPPORTED_RUNTIME_VERSION_RANGE: (u32, u32) =
    (PROVIDER_API_VERSION, PROVIDER_API_VERSION);

/// Explicit, non-default feature toggles for the Reference CPU Provider.
///
/// Registration decisions must be explicit (never an implicit default), so
/// callers construct flags deliberately rather than relying on `Default`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReferenceCpuFeatureFlags {
    pub attention: bool,
    pub rope: bool,
    pub quantization: bool,
}

impl ReferenceCpuFeatureFlags {
    /// The correctness baseline: attention and RoPE implemented, quantization
    /// explicitly unsupported.
    pub const fn baseline() -> Self {
        Self {
            attention: true,
            rope: true,
            quantization: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Error model
// ---------------------------------------------------------------------------

// `ReferenceCpuError`/`ReferenceCpuErrorCode` are also imported from
// `magnetar_runtime` rather than redefined here, for the same reason as
// `HostTensor` above: `HostTensor::new`/`HostTensor::rows_cols` (defined in
// `magnetar_runtime`, since inherent impls must live in the crate that
// defines the type) already return `magnetar_runtime::ReferenceCpuError`,
// so every kernel below that constructs or inspects a `HostTensor` is
// already working in terms of that type. A locally redefined
// `ReferenceCpuError` would also be unable to carry magnetar-runtime's own
// `impl From<ReferenceCpuError> for KernelError` conversion (used pervasively
// by [`ReferenceCpuExecutor`] below): Rust's orphan rule forbids
// `impl From<Local> for Foreign` just as much as `impl From<Foreign> for
// Foreign`, since `KernelError` is foreign to this crate either way -- that
// conversion can only be written once, in `magnetar_runtime`, against its
// own canonical `ReferenceCpuError`, and is already available here via
// `.into()`/`?` because the imported type is literally that same type.

// ---------------------------------------------------------------------------
// Host tensor storage (Provider-owned, opaque)
// ---------------------------------------------------------------------------

// `HostTensor` is defined in `magnetar_runtime` itself (re-exported at its
// crate root), not here: `ProviderExecutionApi::write_tensor`/`read_tensor`/
// `write_tensor_admitted` (in `magnetar-runtime/src/provider.rs`) are typed
// directly against it, as the trait's provisional host-tensor-shaped
// transport ahead of the fully Resource-based rewrite tracked separately
// (task group 5 / Correctif 5). Redefining a same-named, unrelated local
// struct here would not satisfy that trait -- Reference CPU must produce
// and consume the one canonical type the Runtime's generic contract already
// commits to. Imported at the top of this file alongside `ReferenceCpuError`.

fn same_shape(a: &HostTensor, b: &HostTensor) -> Result<(), ReferenceCpuError> {
    if a.shape != b.shape {
        return Err(ReferenceCpuError::new(
            ReferenceCpuErrorCode::ShapeUnsupported,
            format!("shape mismatch: {:?} vs {:?}", a.shape, b.shape),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Correctness-first numeric kernels (pure functions, independently testable)
// ---------------------------------------------------------------------------

/// Correctness-first matmul: `a @ b` with optional logical transpose of
/// either operand and an explicit accumulation dtype (always `f32` here).
pub fn matmul(
    a: &HostTensor,
    b: &HostTensor,
    transpose_a: bool,
    transpose_b: bool,
) -> Result<HostTensor, ReferenceCpuError> {
    let (a_rows, a_cols) = a.rows_cols()?;
    let (b_rows, b_cols) = b.rows_cols()?;
    let (m, k) = if transpose_a {
        (a_cols, a_rows)
    } else {
        (a_rows, a_cols)
    };
    let (k2, n) = if transpose_b {
        (b_cols, b_rows)
    } else {
        (b_rows, b_cols)
    };
    if k != k2 {
        return Err(ReferenceCpuError::new(
            ReferenceCpuErrorCode::ShapeUnsupported,
            format!("matmul inner dimension mismatch: {k} vs {k2}"),
        ));
    }
    let (m, k, n) = (m as usize, k as usize, n as usize);
    // Both operands are row-major, so a logical transpose is just a swap of
    // the row and column strides. Hoisting them turns what was a per-element
    // branch inside the innermost loop into two values computed once.
    let (a_row_stride, a_inner_stride) = if transpose_a {
        (1, a_cols as usize)
    } else {
        (a_cols as usize, 1)
    };
    let (b_inner_stride, b_col_stride) = if transpose_b {
        (1, b_cols as usize)
    } else {
        (b_cols as usize, 1)
    };
    let mut out = vec![0.0_f32; m * n];
    // row -> inner -> col keeps the write row and, when b is untransposed, the
    // b row contiguous. The classic row -> col -> inner order walks b with
    // stride n on the innermost loop, which is the cache-hostile direction.
    for row in 0..m {
        let out_row = &mut out[row * n..(row + 1) * n];
        for inner in 0..k {
            // No zero-skip: this kernel is the correctness oracle other
            // Providers are checked against, and skipping a zero would drop
            // the NaN that 0.0 * NaN must produce.
            let a_value = a.data[row * a_row_stride + inner * a_inner_stride];
            let b_base = inner * b_inner_stride;
            for (col, out_value) in out_row.iter_mut().enumerate() {
                *out_value += a_value * b.data[b_base + col * b_col_stride];
            }
        }
    }
    HostTensor::new([m as u64, n as u64], out)
}

/// Correctness-first embedding lookup: `ids` select rows out of `table`.
pub fn embedding_lookup(
    table: &HostTensor,
    ids: &HostTensor,
) -> Result<HostTensor, ReferenceCpuError> {
    let (vocab, dim) = table.rows_cols()?;
    let mut out = Vec::with_capacity(ids.data.len() * dim as usize);
    for &raw_id in &ids.data {
        if raw_id < 0.0 || raw_id.fract() != 0.0 {
            return Err(ReferenceCpuError::new(
                ReferenceCpuErrorCode::ShapeUnsupported,
                format!("token id {raw_id} is not a non-negative integer"),
            ));
        }
        let id = raw_id as u64;
        if id >= vocab {
            return Err(ReferenceCpuError::new(
                ReferenceCpuErrorCode::ShapeUnsupported,
                format!("token id {id} exceeds vocabulary size {vocab}"),
            ));
        }
        let start = (id * dim) as usize;
        out.extend_from_slice(&table.data[start..start + dim as usize]);
    }
    HostTensor::new([ids.data.len() as u64, dim], out)
}

/// Correctness-first RMSNorm with `f32` accumulation.
pub fn rmsnorm(
    input: &HostTensor,
    weight: &HostTensor,
    epsilon: f32,
) -> Result<HostTensor, ReferenceCpuError> {
    let cols = *input.shape.last().ok_or_else(|| {
        ReferenceCpuError::new(
            ReferenceCpuErrorCode::ShapeUnsupported,
            "RMSNorm expects at least one dimension",
        )
    })?;
    if cols == 0 {
        return Err(ReferenceCpuError::new(
            ReferenceCpuErrorCode::ShapeUnsupported,
            "RMSNorm hidden dimension must be non-zero",
        ));
    }
    if !input.data.len().is_multiple_of(cols as usize) {
        return Err(ReferenceCpuError::new(
            ReferenceCpuErrorCode::ShapeUnsupported,
            format!(
                "RMSNorm data length {} is not divisible by hidden dimension {cols}",
                input.data.len()
            ),
        ));
    }
    let rows = input.data.len() / cols as usize;
    let row_weight_stride = if weight.shape == [cols] || weight.shape == [1, cols] {
        0
    } else if weight.shape == input.shape || weight.shape == [rows as u64, cols] {
        cols as usize
    } else {
        return Err(ReferenceCpuError::new(
            ReferenceCpuErrorCode::ShapeUnsupported,
            format!(
                "RMSNorm weight shape must be [{cols}], [1, {cols}], input shape {:?}, or [{rows}, {cols}], got {:?}",
                input.shape, weight.shape
            ),
        ));
    };
    if epsilon <= 0.0 {
        return Err(ReferenceCpuError::new(
            ReferenceCpuErrorCode::ShapeUnsupported,
            "RMSNorm epsilon must be positive",
        ));
    }
    let mut out = vec![0.0_f32; input.data.len()];
    for row in 0..rows {
        let slice = &input.data[row * cols as usize..(row + 1) * cols as usize];
        let weight_base = row * row_weight_stride;
        let weight_values = &weight.data[weight_base..weight_base + cols as usize];
        let mean_square = slice.iter().map(|value| value * value).sum::<f32>() / cols as f32;
        let scale = 1.0 / (mean_square + epsilon).sqrt();
        for (col, value) in slice.iter().enumerate() {
            out[row * cols as usize + col] = value * scale * weight_values[col];
        }
    }
    HostTensor::new(input.shape.clone(), out)
}

/// Rotary position embedding, rotating consecutive pairs within the first
/// `dimension` elements of each row.
///
/// The absolute position of row `r` is `position_offset + r`. The offset is an
/// explicit parameter rather than a default because a decode step passes a
/// single row whose true position is however many tokens precede it: deriving
/// position from the row index alone would rotate every generated token as if
/// it were the first, which is silently wrong rather than an error.
pub fn rope(
    input: &HostTensor,
    base: f32,
    scale: f32,
    dimension: u64,
    position_offset: u64,
) -> Result<HostTensor, ReferenceCpuError> {
    let (rows, cols) = input.rows_cols()?;
    if dimension == 0 || !dimension.is_multiple_of(2) || dimension > cols {
        return Err(ReferenceCpuError::new(
            ReferenceCpuErrorCode::ShapeUnsupported,
            format!(
                "RoPE dimension {dimension} must be positive, even, and at most the row width {cols}"
            ),
        ));
    }
    if !base.is_finite() || base <= 0.0 {
        return Err(ReferenceCpuError::new(
            ReferenceCpuErrorCode::ShapeUnsupported,
            "RoPE base must be finite and positive",
        ));
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(ReferenceCpuError::new(
            ReferenceCpuErrorCode::ShapeUnsupported,
            "RoPE scale must be finite and positive",
        ));
    }
    let mut out = input.data.clone();
    let half = (dimension / 2) as usize;
    for row in 0..rows as usize {
        let position = ((position_offset as usize + row) as f32) * scale;
        let row_start = row * cols as usize;
        for pair in 0..half {
            let frequency = base.powf(-2.0 * (pair as f32) / dimension as f32);
            let angle = position * frequency;
            let (sin, cos) = angle.sin_cos();
            let even = input.data[row_start + 2 * pair];
            let odd = input.data[row_start + 2 * pair + 1];
            out[row_start + 2 * pair] = even * cos - odd * sin;
            out[row_start + 2 * pair + 1] = even * sin + odd * cos;
        }
    }
    HostTensor::new(input.shape.clone(), out)
}

/// Numerically stable softmax, applied per row (subtracts the row max before
/// exponentiating).
pub fn softmax_rows(input: &HostTensor) -> Result<HostTensor, ReferenceCpuError> {
    let (rows, cols) = input.rows_cols()?;
    let mut out = vec![0.0_f32; input.data.len()];
    for row in 0..rows as usize {
        let slice = &input.data[row * cols as usize..(row + 1) * cols as usize];
        let max = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        // A fully masked row (every entry -inf) would make `value - max`
        // NaN for every column and silently poison the output with Ok.
        // Reject it explicitly instead, matching how the other Reference CPU
        // kernels refuse cases they cannot represent.
        if !max.is_finite() {
            return Err(ReferenceCpuError::new(
                ReferenceCpuErrorCode::ExecutionFailed,
                format!("softmax row {row} has no finite entry to normalize"),
            ));
        }
        let exponentials = slice
            .iter()
            .map(|value| (value - max).exp())
            .collect::<Vec<_>>();
        let sum: f32 = exponentials.iter().sum();
        for (col, value) in exponentials.into_iter().enumerate() {
            out[row * cols as usize + col] = value / sum;
        }
    }
    HostTensor::new(input.shape.clone(), out)
}

/// Simple causal (or unmasked) scaled dot-product attention over `head_count`
/// query heads of `head_dimension` each. `q` has shape
/// `[sequence_length, head_count * head_dimension]`; `k`/`v` have shape
/// `[sequence_length, kv_head_count * head_dimension]`, where `kv_head_count`
/// defaults to `head_count` (standard multi-head attention) but may be
/// smaller as long as it evenly divides `head_count` (grouped-query
/// attention: each group of `head_count / kv_head_count` query heads shares
/// one key/value head). `window_size`, when set, additionally restricts each
/// query to the last `window_size` keys (sliding-window attention).
#[allow(clippy::too_many_arguments)]
pub fn attention(
    q: &HostTensor,
    k: &HostTensor,
    v: &HostTensor,
    head_count: u64,
    head_dimension: u64,
    kv_head_count: Option<u64>,
    window_size: Option<u64>,
    causal: bool,
) -> Result<HostTensor, ReferenceCpuError> {
    same_shape(k, v)?;
    let kv_head_count = kv_head_count.unwrap_or(head_count);
    if head_count == 0 || head_dimension == 0 || kv_head_count == 0 {
        return Err(ReferenceCpuError::new(
            ReferenceCpuErrorCode::ShapeUnsupported,
            "head_count, kv_head_count, and head_dimension must all be positive",
        ));
    }
    if !head_count.is_multiple_of(kv_head_count) {
        return Err(ReferenceCpuError::new(
            ReferenceCpuErrorCode::ShapeUnsupported,
            format!(
                "head_count {head_count} must be an exact multiple of kv_head_count {kv_head_count}"
            ),
        ));
    }
    if window_size == Some(0) {
        return Err(ReferenceCpuError::new(
            ReferenceCpuErrorCode::ShapeUnsupported,
            "window_size must be positive; a zero window admits no keys",
        ));
    }
    // The sliding window is anchored at the query position and bounds only the
    // oldest admissible key, which is a complete description of the mask only
    // when the newest admissible key is already the query itself. Bidirectional
    // attention has no such anchor, so the combination has no single defined
    // meaning here and is rejected rather than silently given one.
    if window_size.is_some() && !causal {
        return Err(ReferenceCpuError::new(
            ReferenceCpuErrorCode::ShapeUnsupported,
            "window_size is only defined for causal attention",
        ));
    }
    let (seq_len, q_model_dim) = q.rows_cols()?;
    let (kv_seq_len, kv_model_dim) = k.rows_cols()?;
    if seq_len > kv_seq_len {
        return Err(ReferenceCpuError::new(
            ReferenceCpuErrorCode::ShapeUnsupported,
            format!("q sequence length {seq_len} cannot exceed k/v sequence length {kv_seq_len}"),
        ));
    }
    if head_count * head_dimension != q_model_dim {
        return Err(ReferenceCpuError::new(
            ReferenceCpuErrorCode::ShapeUnsupported,
            format!(
                "head_count * head_dimension must equal q row width {q_model_dim}, got {head_count} * {head_dimension}"
            ),
        ));
    }
    if kv_head_count * head_dimension != kv_model_dim {
        return Err(ReferenceCpuError::new(
            ReferenceCpuErrorCode::ShapeUnsupported,
            format!(
                "kv_head_count * head_dimension must equal k/v row width {kv_model_dim}, got {kv_head_count} * {head_dimension}"
            ),
        ));
    }
    let group_size = (head_count / kv_head_count) as usize;
    let seq_len = seq_len as usize;
    let kv_seq_len = kv_seq_len as usize;
    let q_model_dim = q_model_dim as usize;
    let kv_model_dim = kv_model_dim as usize;
    let head_dimension = head_dimension as usize;
    let scale = 1.0 / (head_dimension as f32).sqrt();
    // Queries are the *last* `seq_len` positions of the sequence the keys
    // cover. For prefill the two lengths match and this offset is zero; for a
    // decode step against a populated cache it is what places the new token at
    // its true position, so causal masking and the sliding window bound the
    // right keys instead of treating the token as if it were at position 0.
    let query_position_offset = kv_seq_len - seq_len;
    let mut out = vec![0.0_f32; seq_len * q_model_dim];
    // Both scratch buffers are rewritten from scratch for every (head, query)
    // pair and carry nothing between iterations, so they are allocated once
    // here instead of twice per pair. Only the admitted key window is ever
    // read, so they are sized to the widest window rather than pre-filled with
    // a sentinel the code never consumes.
    let mut scores = Vec::with_capacity(seq_len);
    let mut exponentials = Vec::with_capacity(seq_len);
    for head in 0..head_count as usize {
        let kv_head = head / group_size;
        let q_offset = head * head_dimension;
        let kv_offset = kv_head * head_dimension;
        for query_index in 0..seq_len {
            let query_position = query_position_offset + query_index;
            let key_upper = if causal {
                query_position + 1
            } else {
                kv_seq_len
            };
            let key_lower = window_size
                .map(|window| query_position.saturating_sub((window as usize).saturating_sub(1)))
                .unwrap_or(0)
                .min(key_upper);
            let query_base = query_index * q_model_dim + q_offset;
            scores.clear();
            for key_index in key_lower..key_upper {
                let key_base = key_index * kv_model_dim + kv_offset;
                let mut dot = 0.0_f32;
                for dim in 0..head_dimension {
                    dot += q.data[query_base + dim] * k.data[key_base + dim];
                }
                scores.push(dot * scale);
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            exponentials.clear();
            exponentials.extend(scores.iter().map(|value| (value - max).exp()));
            let sum: f32 = exponentials.iter().sum();
            for dim in 0..head_dimension {
                let mut accumulator = 0.0_f32;
                for (offset, weight) in exponentials.iter().enumerate() {
                    let key_index = key_lower + offset;
                    accumulator +=
                        (weight / sum) * v.data[key_index * kv_model_dim + kv_offset + dim];
                }
                out[query_base + dim] = accumulator;
            }
        }
    }
    HostTensor::new(q.shape.clone(), out)
}

/// SiLU activation: `x * sigmoid(x)`.
pub fn silu(input: &HostTensor) -> HostTensor {
    let data = input
        .data
        .iter()
        .map(|&x| x * (1.0 / (1.0 + (-x).exp())))
        .collect::<Vec<_>>();
    HostTensor {
        shape: input.shape.clone(),
        data,
    }
}

/// GELU activation using the tanh approximation.
pub fn gelu(input: &HostTensor) -> HostTensor {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    let data = input
        .data
        .iter()
        .map(|&x| 0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh()))
        .collect::<Vec<_>>();
    HostTensor {
        shape: input.shape.clone(),
        data,
    }
}

pub fn add(a: &HostTensor, b: &HostTensor) -> Result<HostTensor, ReferenceCpuError> {
    same_shape(a, b)?;
    let data = a
        .data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| x + y)
        .collect::<Vec<_>>();
    Ok(HostTensor {
        shape: a.shape.clone(),
        data,
    })
}

pub fn mul(a: &HostTensor, b: &HostTensor) -> Result<HostTensor, ReferenceCpuError> {
    same_shape(a, b)?;
    let data = a
        .data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| x * y)
        .collect::<Vec<_>>();
    Ok(HostTensor {
        shape: a.shape.clone(),
        data,
    })
}

pub fn residual_add(
    input: &HostTensor,
    residual: &HostTensor,
) -> Result<HostTensor, ReferenceCpuError> {
    add(input, residual)
}

/// Reference CPU only stores `f32`; any other portable dtype requires an
/// explicit conversion step and is never converted silently.
pub fn dtype_conversion(
    input: &HostTensor,
    from: ComputeDType,
    to: ComputeDType,
) -> Result<HostTensor, ReferenceCpuError> {
    if from != ComputeDType::Float32 || to != ComputeDType::Float32 {
        return Err(ReferenceCpuError::new(
            ReferenceCpuErrorCode::DTypeUnsupported,
            format!("Reference CPU only supports explicit f32; requested {from:?} -> {to:?}"),
        ));
    }
    Ok(input.clone())
}

/// Reference CPU only stores contiguous, row-major layouts. Any other target
/// layout is explicitly rejected rather than silently reinterpreted. Strided
/// targets get a distinct, clearly-labeled placeholder rejection (strided
/// support is defined but not yet implemented); blocked, paged, and other
/// provider-opaque layouts get the generic unsupported-layout rejection.
pub fn layout_conversion(
    input: &HostTensor,
    from: TensorLayoutKind,
    to: TensorLayoutKind,
) -> Result<HostTensor, ReferenceCpuError> {
    if from == TensorLayoutKind::Contiguous && to == TensorLayoutKind::Contiguous {
        return Ok(input.clone());
    }
    if to == TensorLayoutKind::Strided || from == TensorLayoutKind::Strided {
        return Err(ReferenceCpuError::new(
            ReferenceCpuErrorCode::LayoutUnsupported,
            format!(
                "strided layout conversion is a defined placeholder, not yet implemented (requested {from:?} -> {to:?})"
            ),
        ));
    }
    Err(ReferenceCpuError::new(
        ReferenceCpuErrorCode::LayoutUnsupported,
        format!("Reference CPU only supports contiguous layout; requested {from:?} -> {to:?}"),
    ))
}

/// Quantized formats are not implemented; this always fails explicitly.
pub fn dequantize_placeholder() -> ReferenceCpuError {
    ReferenceCpuError::new(
        ReferenceCpuErrorCode::DTypeUnsupported,
        "Reference CPU has no quantized kernel implementation",
    )
}

/// Quantization mirrors [`dequantize_placeholder`]: Reference CPU declares
/// the operation explicitly rather than silently accepting a quantized
/// output, but does not implement it in the first scope.
pub fn quantize_placeholder() -> ReferenceCpuError {
    ReferenceCpuError::new(
        ReferenceCpuErrorCode::DTypeUnsupported,
        "Reference CPU has no quantized kernel implementation",
    )
}

// ---------------------------------------------------------------------------
// Fallback policy
// ---------------------------------------------------------------------------

/// Policy inputs that decide whether Reference CPU fallback is permitted for
/// one candidate resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FallbackPolicyContext {
    pub policy_allows_fallback: bool,
    pub dtype_conversion_required: bool,
    pub dtype_conversion_allowed: bool,
    pub layout_conversion_required: bool,
    pub layout_conversion_allowed: bool,
}

impl FallbackPolicyContext {
    /// The strictest default: fallback itself, and any conversion it would
    /// require, must each be explicitly permitted.
    pub const fn new(policy_allows_fallback: bool) -> Self {
        Self {
            policy_allows_fallback,
            dtype_conversion_required: false,
            dtype_conversion_allowed: false,
            layout_conversion_required: false,
            layout_conversion_allowed: false,
        }
    }

    pub const fn with_dtype_conversion(mut self, required: bool, allowed: bool) -> Self {
        self.dtype_conversion_required = required;
        self.dtype_conversion_allowed = allowed;
        self
    }

    pub const fn with_layout_conversion(mut self, required: bool, allowed: bool) -> Self {
        self.layout_conversion_required = required;
        self.layout_conversion_allowed = allowed;
        self
    }
}

/// Whether Reference CPU fallback is permitted for one candidate resource.
///
/// Fallback is denied unless the caller explicitly permits it, is always
/// denied when Resource Affinity forbids host movement, and is denied when it
/// would require a dtype or layout conversion that policy does not allow.
pub fn evaluate_fallback(
    affinity: &ResourceAffinity,
    context: &FallbackPolicyContext,
) -> Result<(), ReferenceCpuError> {
    if !context.policy_allows_fallback {
        return Err(ReferenceCpuError::new(
            ReferenceCpuErrorCode::FallbackDenied,
            "Runtime policy does not permit Reference CPU fallback",
        ));
    }
    if matches!(affinity.fallback(), FallbackClass::ProviderPinned) {
        return Err(ReferenceCpuError::new(
            ReferenceCpuErrorCode::FallbackDenied,
            "Resource Affinity forbids host movement",
        ));
    }
    if context.dtype_conversion_required && !context.dtype_conversion_allowed {
        return Err(ReferenceCpuError::new(
            ReferenceCpuErrorCode::FallbackDenied,
            "fallback would require a dtype conversion that policy forbids",
        ));
    }
    if context.layout_conversion_required && !context.layout_conversion_allowed {
        return Err(ReferenceCpuError::new(
            ReferenceCpuErrorCode::FallbackDenied,
            "fallback would require a layout conversion that policy forbids",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Provider identity, Device, Kernel advertisement
// ---------------------------------------------------------------------------

pub fn reference_cpu_provider_metadata() -> ProviderMetadata {
    ProviderMetadata::new(
        REFERENCE_CPU_PROVIDER_NAME,
        REFERENCE_CPU_PROVIDER_VERSION,
        REFERENCE_CPU_PROVIDER_VENDOR,
        "Correctness-first, host-memory reference implementation of portable Operators",
    )
}

/// Runtime-detected SIMD feature ids, expressed as [`CapabilityId`]s under the
/// `magnetar:cpu-feature/*` namespace so they use the same extensibility
/// point as any other Device execution capability.
#[cfg(target_arch = "x86_64")]
fn detected_simd_capabilities() -> Vec<CapabilityId> {
    let mut features = Vec::new();
    if std::is_x86_feature_detected!("sse4.2") {
        features.push(CapabilityId::new("magnetar:cpu-feature/sse4.2"));
    }
    if std::is_x86_feature_detected!("avx2") {
        features.push(CapabilityId::new("magnetar:cpu-feature/avx2"));
    }
    if std::is_x86_feature_detected!("fma") {
        features.push(CapabilityId::new("magnetar:cpu-feature/fma"));
    }
    features
}

#[cfg(not(target_arch = "x86_64"))]
fn detected_simd_capabilities() -> Vec<CapabilityId> {
    Vec::new()
}

pub fn reference_cpu_device() -> DeviceDescriptor {
    let mut metadata = DeviceMetadata::new(
        DeviceId::new(REFERENCE_CPU_DEVICE_ID),
        "Reference CPU",
        DeviceType::Cpu,
        REFERENCE_CPU_PROVIDER_NAME,
    );
    metadata.vendor = REFERENCE_CPU_PROVIDER_VENDOR.into();
    metadata.architecture = std::env::consts::ARCH.into();
    metadata.compute_units = std::thread::available_parallelism()
        .map(|count| count.get() as u32)
        .unwrap_or(1);
    metadata
        .execution_capabilities
        .extend(detected_simd_capabilities());
    // Mirrors the dtype/layout/memory-class support already advertised at
    // Kernel granularity (see `baseline_advertisement`): Reference CPU only
    // ever executes `f32`, contiguous, host-resident tensors.
    metadata.dtype_support = [ComputeDType::Float32].into_iter().collect();
    metadata.layout_support = [TensorLayoutKind::Contiguous].into_iter().collect();
    metadata.memory_class_support = [KernelMemoryClass::Host].into_iter().collect();
    metadata.execution_limits = DeviceExecutionLimits {
        max_concurrent_operations: Some(metadata.compute_units),
        max_workspace_bytes: Some(1 << 20),
    };
    metadata.pressure = ProviderPressureLevel::Low;
    DeviceDescriptor::new(metadata)
}

fn reference_cpu_kernel_id(operator: OperatorId, name: &str) -> KernelId {
    KernelId::new(
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        name,
        CapabilityVersion::new(1, 0, 0),
        operator,
        KernelOperatorVersionRange::exact(1),
        REFERENCE_CPU_KERNEL_FAMILY,
    )
    .with_conformance_profile(REFERENCE_CPU_CONFORMANCE_PROFILE)
}

fn baseline_advertisement(name: &str, family: OperatorFamily) -> KernelAdvertisement {
    let operator = OperatorId::magnetar(name, 1, family);
    let id = reference_cpu_kernel_id(operator, name);
    let mut advertisement = KernelAdvertisement::new(id)
        .with_dtypes(TensorRole::Input, [ComputeDType::Float32])
        .with_dtypes(TensorRole::Output, [ComputeDType::Float32])
        .with_layouts([TensorLayoutKind::Contiguous])
        .with_memory_classes([KernelMemoryClass::Host])
        .with_devices([DeviceBinding::new(DeviceId::new(REFERENCE_CPU_DEVICE_ID))]);
    // Reference CPU executes synchronously and cannot cooperatively cancel
    // mid-kernel, but it does honor a deadline that has already elapsed
    // before dispatch starts (see `execute_invocation`).
    advertisement.cancellation = KernelCancellationSupport::TimeoutOnly;
    // Kernels whose tensors are always rank-2 in this implementation advertise
    // that constraint explicitly; kernels that mix ranks across resources
    // (e.g. embedding's rank-1 ids against its rank-2 table, or RMSNorm's
    // vector/full-shape weights against rank-2 activations) or accept
    // arbitrary rank (elementwise, activations, conversions) are left
    // unconstrained.
    if matches!(name, "matmul" | "rope" | "attention" | "softmax") {
        advertisement.shape.rank = Some(2);
    }
    advertisement
}

/// The initial, correctness-first Reference CPU kernel set. Kernels that are
/// not implemented (quantized formats) are intentionally absent: unsupported
/// Operators are never assumed available just because Reference CPU exists.
pub fn reference_cpu_kernel_advertisements() -> Vec<KernelAdvertisement> {
    let mut attention = baseline_advertisement("attention", OperatorFamily::Attention);
    // Attention is the one Kernel that genuinely needs scratch space (the
    // per-query score buffer), so it requests it through the Memory
    // Manager rather than allocating it invisibly to the Runtime. It is
    // also not advertised as browser-compatible: Host-class workspace
    // allocation is not meaningful against browser linear memory (see
    // `run_invocation`, which explicitly rejects it on `wasm32`).
    attention.workspace =
        KernelWorkspaceRequirements::required(1 << 20, KernelMemoryClass::Host, 4);
    attention.browser_compatible = false;
    // No incremental/paged KV cache is implemented; state that explicitly
    // rather than leaving the metadata slot silently absent.
    attention.kv_cache = Some(KernelKvCacheMetadata {
        layouts: BTreeSet::new(),
        paged_cache: false,
        append: false,
        read: false,
        dtypes: BTreeSet::new(),
        memory_classes: BTreeSet::new(),
        affinity: None,
    });
    let embedding = baseline_advertisement("embedding", OperatorFamily::Tensor);

    vec![
        baseline_advertisement("matmul", OperatorFamily::LinearAlgebra),
        embedding,
        baseline_advertisement("rmsnorm", OperatorFamily::Normalization),
        baseline_advertisement("rope", OperatorFamily::PositionEncoding),
        attention,
        baseline_advertisement("softmax", OperatorFamily::Activation),
        baseline_advertisement("silu", OperatorFamily::Activation),
        baseline_advertisement("gelu", OperatorFamily::Activation),
        baseline_advertisement("activation", OperatorFamily::Activation),
        baseline_advertisement("add", OperatorFamily::Tensor),
        baseline_advertisement("mul", OperatorFamily::Tensor),
        baseline_advertisement("residual-add", OperatorFamily::Tensor),
        baseline_advertisement("dtype-conversion", OperatorFamily::Tensor),
        baseline_advertisement("layout-conversion", OperatorFamily::Layout),
    ]
}

/// One check within a [`ReferenceCpuConformanceReport`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReferenceCpuConformanceCheck {
    pub name: &'static str,
    pub passed: bool,
    pub detail: Option<String>,
}

/// Reference CPU's own conformance report: a small, fixed set of
/// known-input/known-output checks against its Kernel functions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReferenceCpuConformanceReport {
    pub profile: &'static str,
    pub checks: Vec<ReferenceCpuConformanceCheck>,
}

impl ReferenceCpuConformanceReport {
    pub fn is_conformant(&self) -> bool {
        self.checks.iter().all(|check| check.passed)
    }
}

/// Validates the Tensor Resource and Layout contract for the host
/// contiguous case Reference CPU supports: a valid `TensorDescriptor`
/// (shape, dtype, contiguous layout), a lifecycle walk from `Declared` to
/// `Ready`, and a readiness state Kernels can dispatch against, all
/// without touching raw pointers or Provider handles.
fn reference_cpu_tensor_resource_conformance_check() -> ReferenceCpuConformanceCheck {
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let descriptor_valid = descriptor
        .validate(&TensorDescriptorLimits::default())
        .is_ok();
    let residency = TensorResidency::new(
        TensorResourceId::new("conformance-tensor-resource"),
        MemoryPlacement::HostOrdinary,
        ResourceAffinity::new(FallbackClass::Transparent),
    );
    let host_visible = residency.is_host_visible();
    let mut resource = TensorResource::new(
        TensorResourceId::new("conformance-tensor-resource"),
        descriptor,
        residency,
    );
    let lifecycle_ok = resource
        .transition_to(TensorLifecycleState::Planned)
        .is_ok()
        && resource
            .transition_to(TensorLifecycleState::Allocating)
            .is_ok()
        && resource.mark_ready().is_ok();
    let ready = resource.ensure_usable().is_ok() && resource.readiness == TensorReadiness::Ready;
    let passed = descriptor_valid && host_visible && lifecycle_ok && ready;
    ReferenceCpuConformanceCheck {
        name: "tensor-resource-metadata-and-readiness",
        passed,
        detail: (!passed).then(|| {
            format!(
                "descriptor_valid={descriptor_valid} host_visible={host_visible} lifecycle_ok={lifecycle_ok} ready={ready}"
            )
        }),
    }
}

// ---------------------------------------------------------------------------
// Executor: opaque host storage + ProviderExecutionApi + KernelInvocation execution
// ---------------------------------------------------------------------------

/// Reference CPU's execution boundary. Holds Provider-owned, opaque tensor
/// storage keyed by [`TensorResourceId`]; the Runtime only ever references
/// resources by identity.
pub struct ReferenceCpuExecutor {
    storage: Mutex<BTreeMap<TensorResourceId, HostTensor>>,
    observations: Mutex<Vec<KernelObservation>>,
    submitted: Mutex<BTreeMap<ProviderExecutionId, ProviderExecutionRequest>>,
    /// Results of Kernel invocations submitted through
    /// [`Self::submit_kernel_invocation`], keyed by the
    /// [`ProviderExecutionHandle`] returned at submission time and removed
    /// on the first [`Self::complete_kernel_invocation`] call that consumes
    /// them (single-consumption; task 5.3/Correctif 2 causality).
    kernel_executions: Mutex<BTreeMap<ProviderExecutionId, KernelResult>>,
    /// The current `MemoryManager` allocation backing each resource id
    /// written through [`Self::write_tensor_admitted`], so a later write
    /// to the *same* resource id can release the allocation it replaces
    /// instead of leaving it accounted forever. Resources written through
    /// plain [`Self::write_tensor`] are never tracked here -- that path
    /// remains genuinely unadmitted, unchanged.
    resource_allocations: Mutex<BTreeMap<TensorResourceId, MemoryAllocationId>>,
    next_execution_ordinal: AtomicU64,
}

impl Default for ReferenceCpuExecutor {
    fn default() -> Self {
        Self {
            storage: Mutex::new(BTreeMap::new()),
            observations: Mutex::new(Vec::new()),
            submitted: Mutex::new(BTreeMap::new()),
            kernel_executions: Mutex::new(BTreeMap::new()),
            resource_allocations: Mutex::new(BTreeMap::new()),
            next_execution_ordinal: AtomicU64::new(0),
        }
    }
}

impl ReferenceCpuExecutor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn write_tensor(&self, id: TensorResourceId, tensor: HostTensor) {
        self.storage.lock().unwrap().insert(id, tensor);
    }

    pub fn read_tensor(&self, id: &TensorResourceId) -> Option<HostTensor> {
        self.storage.lock().unwrap().get(id).cloned()
    }

    /// Drops a tensor from this executor's opaque storage. Returns `true`
    /// if `id` was present and removed, `false` if it was already absent.
    pub fn release_tensor(&self, id: &TensorResourceId) -> bool {
        self.storage.lock().unwrap().remove(id).is_some()
    }

    /// See [`ProviderExecutionApi::release_admitted_tensor`]: releases the
    /// `MemoryManager` allocation this executor tracked for `id` (from a
    /// prior [`Self::write_tensor_admitted`] call), if any, then drops `id`
    /// from storage exactly like [`Self::release_tensor`].
    pub fn release_admitted_tensor(
        &self,
        memory: &mut MemoryManager,
        id: &TensorResourceId,
    ) -> bool {
        if let Some(allocation) = self.resource_allocations.lock().unwrap().remove(id) {
            let _ = memory.release(allocation);
        }
        self.storage.lock().unwrap().remove(id).is_some()
    }

    /// See [`ProviderExecutionApi::write_tensor_admitted`]: admits `tensor`
    /// through `memory` before writing it, and releases whatever allocation
    /// this executor previously admitted for the same `id`, if any.
    pub fn write_tensor_admitted(
        &self,
        memory: &mut MemoryManager,
        id: TensorResourceId,
        tensor: HostTensor,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<(), MemoryError> {
        let byte_size = tensor.data.len() as u64 * std::mem::size_of::<f32>() as u64;
        let allocation = memory.allocate(MemoryAllocationRequest::new(
            class,
            byte_size,
            MemoryPlacement::ProviderOwnedOpaque(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
            owner,
        ))?;
        let previous = self
            .resource_allocations
            .lock()
            .unwrap()
            .insert(id.clone(), allocation.id);
        if let Some(previous) = previous {
            let _ = memory.release(previous);
        }
        self.storage.lock().unwrap().insert(id, tensor);
        Ok(())
    }

    /// See [`ProviderExecutionApi::read_tensor_value`]: always
    /// [`TensorValue::Host`] -- Reference CPU never declines host
    /// materialization.
    pub fn read_tensor_value(&self, id: &TensorResourceId) -> Option<TensorValue> {
        self.read_tensor(id).map(TensorValue::Host)
    }

    /// See [`ProviderExecutionApi::write_tensor_value`]. Reference CPU only
    /// ever receives [`TensorValue::Host`]; an [`TensorValue::Opaque`] write
    /// (which would mean "store this, but I have no bytes for it") is not
    /// meaningful for a host-visible-only Provider, so it is a documented
    /// no-op rather than a panic.
    pub fn write_tensor_value(&self, id: TensorResourceId, value: TensorValue) {
        if let TensorValue::Host(tensor) = value {
            self.write_tensor(id, tensor);
        }
    }

    /// See [`ProviderExecutionApi::write_tensor_value_admitted`]. Same
    /// `Opaque`-is-a-no-op reasoning as [`Self::write_tensor_value`]; an
    /// `Opaque` write admits nothing and succeeds trivially, since there is
    /// no byte size to account for it.
    pub fn write_tensor_value_admitted(
        &self,
        memory: &mut MemoryManager,
        id: TensorResourceId,
        value: TensorValue,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<(), MemoryError> {
        match value {
            TensorValue::Host(tensor) => {
                self.write_tensor_admitted(memory, id, tensor, class, owner)
            }
            TensorValue::Opaque => Ok(()),
        }
    }

    pub fn observations(&self) -> Vec<KernelObservation> {
        self.observations.lock().unwrap().clone()
    }

    fn observe(&self, observation: KernelObservation) {
        self.observations.lock().unwrap().push(observation);
    }

    /// Runs a small, fixed set of known-input/known-output checks against
    /// this Provider's own correctness-first Kernel functions and records
    /// the result as a Kernel observation. This is Reference CPU's own
    /// conformance report; it is distinct from the shared
    /// `ProviderConformanceSuite` (which exercises Provider/Device/Kernel
    /// contract structure generically, not per-Operator numeric semantics).
    pub fn run_conformance_checks(&self) -> ReferenceCpuConformanceReport {
        let close = |a: &[f32], b: &[f32]| -> bool {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
        };
        let mut checks = Vec::new();

        let matmul_result = matmul(
            &HostTensor::new([2, 2], [1.0, 2.0, 3.0, 4.0]).unwrap(),
            &HostTensor::new([2, 2], [5.0, 6.0, 7.0, 8.0]).unwrap(),
            false,
            false,
        );
        checks.push(ReferenceCpuConformanceCheck {
            name: "matmul-known-output",
            passed: matches!(&matmul_result, Ok(output) if close(&output.data, &[19.0, 22.0, 43.0, 50.0])),
            detail: matmul_result.err().map(|error| error.to_string()),
        });

        let embedding_result = embedding_lookup(
            &HostTensor::new([2, 2], [1.0, 2.0, 3.0, 4.0]).unwrap(),
            &HostTensor::new([1], [1.0]).unwrap(),
        );
        checks.push(ReferenceCpuConformanceCheck {
            name: "embedding-known-output",
            passed: matches!(&embedding_result, Ok(output) if close(&output.data, &[3.0, 4.0])),
            detail: embedding_result.err().map(|error| error.to_string()),
        });

        let rmsnorm_result = rmsnorm(
            &HostTensor::new([1, 2], [3.0, 4.0]).unwrap(),
            &HostTensor::new([2], [1.0, 1.0]).unwrap(),
            1e-6,
        );
        // mean_square = (3^2 + 4^2) / 2 = 12.5; scale = 1 / sqrt(12.5).
        let rmsnorm_scale = 1.0_f32 / 12.5_f32.sqrt();
        checks.push(ReferenceCpuConformanceCheck {
            name: "rmsnorm-known-output",
            passed: matches!(&rmsnorm_result, Ok(output) if close(&output.data, &[3.0 * rmsnorm_scale, 4.0 * rmsnorm_scale])),
            detail: rmsnorm_result.err().map(|error| error.to_string()),
        });

        let rope_result = rope(
            &HostTensor::new([1, 2], [1.0, 0.0]).unwrap(),
            10000.0,
            1.0,
            2,
            0,
        );
        checks.push(ReferenceCpuConformanceCheck {
            name: "rope-baseline-known-output",
            passed: matches!(&rope_result, Ok(output) if close(&output.data, &[1.0, 0.0])),
            detail: rope_result.err().map(|error| error.to_string()),
        });

        let attention_result = attention(
            &HostTensor::new([1, 2], [1.0, 0.0]).unwrap(),
            &HostTensor::new([1, 2], [1.0, 0.0]).unwrap(),
            &HostTensor::new([1, 2], [5.0, 6.0]).unwrap(),
            1,
            2,
            None,
            None,
            true,
        );
        checks.push(ReferenceCpuConformanceCheck {
            name: "attention-single-token-known-output",
            passed: matches!(&attention_result, Ok(output) if close(&output.data, &[5.0, 6.0])),
            detail: attention_result.err().map(|error| error.to_string()),
        });

        let softmax_result = softmax_rows(&HostTensor::new([1, 2], [0.0, 0.0]).unwrap());
        checks.push(ReferenceCpuConformanceCheck {
            name: "softmax-uniform-input",
            passed: matches!(&softmax_result, Ok(output) if close(&output.data, &[0.5, 0.5])),
            detail: softmax_result.err().map(|error| error.to_string()),
        });

        let silu_result = silu(&HostTensor::new([1], [0.0]).unwrap());
        checks.push(ReferenceCpuConformanceCheck {
            name: "silu-zero-input",
            passed: close(&silu_result.data, &[0.0]),
            detail: None,
        });

        let add_result = add(
            &HostTensor::new([2], [1.0, 2.0]).unwrap(),
            &HostTensor::new([2], [3.0, 4.0]).unwrap(),
        );
        checks.push(ReferenceCpuConformanceCheck {
            name: "add-known-output",
            passed: matches!(&add_result, Ok(output) if close(&output.data, &[4.0, 6.0])),
            detail: add_result.err().map(|error| error.to_string()),
        });

        let mul_result = mul(
            &HostTensor::new([2], [2.0, 3.0]).unwrap(),
            &HostTensor::new([2], [4.0, 5.0]).unwrap(),
        );
        checks.push(ReferenceCpuConformanceCheck {
            name: "mul-known-output",
            passed: matches!(&mul_result, Ok(output) if close(&output.data, &[8.0, 15.0])),
            detail: mul_result.err().map(|error| error.to_string()),
        });

        let residual_add_result = residual_add(
            &HostTensor::new([2], [1.0, 2.0]).unwrap(),
            &HostTensor::new([2], [10.0, 20.0]).unwrap(),
        );
        checks.push(ReferenceCpuConformanceCheck {
            name: "residual-add-known-output",
            passed: matches!(&residual_add_result, Ok(output) if close(&output.data, &[11.0, 22.0])),
            detail: residual_add_result.err().map(|error| error.to_string()),
        });

        let dtype_conversion_result = dtype_conversion(
            &HostTensor::new([1], [1.0]).unwrap(),
            ComputeDType::Float32,
            ComputeDType::Float32,
        );
        checks.push(ReferenceCpuConformanceCheck {
            name: "dtype-conversion-f32-explicit",
            passed: matches!(&dtype_conversion_result, Ok(output) if close(&output.data, &[1.0])),
            detail: dtype_conversion_result.err().map(|error| error.to_string()),
        });

        let layout_conversion_result = layout_conversion(
            &HostTensor::new([1], [1.0]).unwrap(),
            TensorLayoutKind::Contiguous,
            TensorLayoutKind::Contiguous,
        );
        checks.push(ReferenceCpuConformanceCheck {
            name: "layout-conversion-contiguous-explicit",
            passed: matches!(&layout_conversion_result, Ok(output) if close(&output.data, &[1.0])),
            detail: layout_conversion_result
                .err()
                .map(|error| error.to_string()),
        });

        checks.push(reference_cpu_tensor_resource_conformance_check());

        let report = ReferenceCpuConformanceReport {
            profile: REFERENCE_CPU_CONFORMANCE_PROFILE,
            checks,
        };
        self.observe(
            KernelObservation::new(KernelObservationKind::KernelConformanceResult)
                .with_redacted_metadata("profile", report.profile)
                .with_redacted_metadata("passed", report.is_conformant().to_string())
                .with_redacted_metadata("checks", report.checks.len().to_string()),
        );
        report
    }

    fn attribute_float(
        attributes: &BTreeMap<String, OperatorAttributeValue>,
        key: &str,
        default: f32,
    ) -> f32 {
        match attributes.get(key) {
            Some(OperatorAttributeValue::Float(value)) => *value as f32,
            _ => default,
        }
    }

    fn attribute_integer(
        attributes: &BTreeMap<String, OperatorAttributeValue>,
        key: &str,
    ) -> Option<u64> {
        match attributes.get(key) {
            Some(OperatorAttributeValue::Integer(value)) if *value >= 0 => Some(*value as u64),
            _ => None,
        }
    }

    fn attribute_bool(
        attributes: &BTreeMap<String, OperatorAttributeValue>,
        key: &str,
        default: bool,
    ) -> bool {
        match attributes.get(key) {
            Some(OperatorAttributeValue::Boolean(value)) => *value,
            _ => default,
        }
    }

    /// Evaluates [`evaluate_fallback`] and records the decision as a Kernel
    /// observation (considered, then used or failed) so fallback use is
    /// observable rather than silent.
    pub fn evaluate_fallback_observed(
        &self,
        kernel: &KernelId,
        affinity: &ResourceAffinity,
        context: &FallbackPolicyContext,
    ) -> Result<(), ReferenceCpuError> {
        self.observe(
            KernelObservation::new(KernelObservationKind::KernelFallbackConsidered)
                .with_kernel(kernel),
        );
        match evaluate_fallback(affinity, context) {
            Ok(()) => {
                self.observe(
                    KernelObservation::new(KernelObservationKind::KernelFallbackUsed)
                        .with_kernel(kernel),
                );
                Ok(())
            }
            Err(error) => {
                self.observe(
                    KernelObservation::new(KernelObservationKind::KernelFallbackFailed)
                        .with_kernel(kernel)
                        .with_redacted_metadata("error", error.id()),
                );
                Err(error)
            }
        }
    }

    /// Requests scratch-space workspace through the Runtime's
    /// [`MemoryManager`] (used by Kernels, such as `attention`, that
    /// advertise a required workspace) and returns the allocation to attach
    /// to the [`KernelInvocation`] via `with_workspace`.
    pub fn allocate_workspace(
        &self,
        memory: &mut MemoryManager,
        size_bytes: u64,
    ) -> Result<MemoryAllocationId, MemoryError> {
        let provider = ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME);
        let request = MemoryAllocationRequest::new(
            MemoryAllocationClass::TemporaryWorkspace,
            size_bytes,
            MemoryPlacement::HostOrdinary,
            MemoryAllocationOwner::Provider(provider),
        );
        memory.allocate(request).map(|allocation| allocation.id)
    }

    /// Admits Runtime memory for every declared output of one
    /// [`KernelInvocation`] *before* dispatching it, then executes it and
    /// records tensor residency through the Runtime's [`MemoryManager`], so
    /// Provider materialization can never precede Runtime admission.
    ///
    /// If admission for any output is denied, the Kernel is never dispatched
    /// (`self.execute_invocation` is not called) and no bytes are written
    /// into Provider-owned storage. Reservations already admitted for this
    /// invocation are released before returning.
    pub fn execute_invocation_with_memory_manager(
        &self,
        advertisement: &KernelAdvertisement,
        operator: &OperatorSpec,
        invocation: &KernelInvocation,
        memory: &mut MemoryManager,
    ) -> KernelResult {
        let provider = ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME);
        let mut admitted: Vec<(TensorResourceId, MemoryAllocationId)> =
            Vec::with_capacity(invocation.outputs.len());
        for output in &invocation.outputs {
            let resource = &output.resource;
            let byte_size = match resource.descriptor.byte_size() {
                Ok(byte_size) => byte_size,
                Err(error) => {
                    for (_, allocation_id) in &admitted {
                        let _ = memory.release(*allocation_id);
                    }
                    return KernelResult::failure(
                        invocation.id.clone(),
                        KernelError::KernelExecutionFailed {
                            reason: format!(
                                "cannot admit output {}: invalid tensor descriptor ({error:?})",
                                resource.id
                            ),
                        },
                    );
                }
            };
            let request = MemoryAllocationRequest::new(
                MemoryAllocationClass::Tensor,
                byte_size,
                MemoryPlacement::ProviderOwnedOpaque(provider.clone()),
                MemoryAllocationOwner::Provider(provider.clone()),
            )
            .with_affinity(resource.affinity.clone());
            match memory.allocate(request) {
                Ok(allocation) => admitted.push((resource.id.clone(), allocation.id)),
                Err(error) => {
                    self.observe(
                        KernelObservation::new(
                            KernelObservationKind::KernelMemoryFeasibilityFailed,
                        )
                        .with_kernel(&invocation.kernel)
                        .with_invocation(invocation.id.clone()),
                    );
                    for (_, allocation_id) in &admitted {
                        let _ = memory.release(*allocation_id);
                    }
                    return KernelResult::failure(
                        invocation.id.clone(),
                        KernelError::KernelExecutionFailed {
                            reason: format!(
                                "memory admission denied for output {}: {error:?}",
                                resource.id
                            ),
                        },
                    );
                }
            }
        }

        let result = self.execute_invocation(advertisement, operator, invocation);
        if result.status != KernelResultStatus::Succeeded {
            for (_, allocation_id) in &admitted {
                let _ = memory.release(*allocation_id);
            }
            return result;
        }
        for resource in &result.updated_resources {
            let Some((_, allocation_id)) = admitted
                .iter()
                .find(|(resource_id, _)| *resource_id == resource.id)
                .map(|(resource_id, allocation_id)| (resource_id.clone(), *allocation_id))
            else {
                continue;
            };
            let _ = memory.record_tensor_residency(
                TensorResidency::new(
                    resource.id.clone(),
                    MemoryPlacement::ProviderOwnedOpaque(provider.clone()),
                    resource.affinity.clone(),
                )
                .with_allocation(allocation_id),
            );
        }
        result
    }

    fn next_provider_execution_id(&self, label: &str) -> ProviderExecutionId {
        let ordinal = self.next_execution_ordinal.fetch_add(1, Ordering::Relaxed);
        ProviderExecutionId::new(format!("{REFERENCE_CPU_PROVIDER_NAME}:{label}:{ordinal}"))
    }

    /// Submits one Runtime-created [`KernelInvocation`] for execution and
    /// returns the [`ProviderExecutionHandle`] that causally identifies it:
    /// since Reference CPU is a synchronous Provider, the numerical work
    /// (via [`Self::execute_invocation_with_memory_manager`]) runs as part
    /// of this call, and the resulting [`KernelResult`] is stored under the
    /// returned handle for a later, single [`Self::complete_kernel_invocation`]
    /// call to observe. No [`ProviderExecutionHandle`] is ever constructed
    /// without a corresponding submission (Correctif 2: Provider submit and
    /// complete are causal, not post-hoc evidence).
    pub fn submit_kernel_invocation(
        &self,
        advertisement: &KernelAdvertisement,
        operator: &OperatorSpec,
        invocation: &KernelInvocation,
        memory: &mut MemoryManager,
    ) -> ProviderExecutionHandle {
        let provider = ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME);
        let execution_id = self.next_provider_execution_id(invocation.id.as_str());
        let handle = ProviderExecutionHandle {
            id: execution_id.clone(),
            operation: ScheduledOperationId::new(
                self.next_execution_ordinal.load(Ordering::Relaxed),
            ),
            plan: ExecutionPlanId::new(invocation.id.as_str().to_string()),
            provider,
            device: None,
        };
        let result = self.execute_invocation_with_memory_manager(
            advertisement,
            operator,
            invocation,
            memory,
        );
        self.kernel_executions
            .lock()
            .unwrap()
            .insert(execution_id, result);
        handle
    }

    /// Observes the result of a Kernel invocation previously submitted
    /// through [`Self::submit_kernel_invocation`]. Fails with a structured
    /// error if `handle` was never returned by that method, or has already
    /// been completed once (single consumption: the stored result is
    /// removed on success).
    pub fn complete_kernel_invocation(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<KernelResult, ProviderExecutionError> {
        self.kernel_executions
            .lock()
            .unwrap()
            .remove(&handle.id)
            .ok_or_else(|| {
                ProviderExecutionError::new(
                    ProviderExecutionErrorCode::ExecutionFailed,
                    ProviderExecutionPhase::Complete,
                    handle.provider.clone(),
                    handle.device.clone(),
                    "no Kernel execution is associated with this handle: it was never \
                     submitted through submit_kernel_invocation, or has already been \
                     completed once",
                )
            })
    }

    /// Executes one Runtime-created [`KernelInvocation`] against this
    /// Provider's advertised Kernel, dispatching to the matching pure kernel
    /// function and recording its output(s) in opaque host storage.
    pub fn execute_invocation(
        &self,
        advertisement: &KernelAdvertisement,
        operator: &OperatorSpec,
        invocation: &KernelInvocation,
    ) -> KernelResult {
        self.observe(
            KernelObservation::new(KernelObservationKind::KernelDispatchStarted)
                .with_kernel(&invocation.kernel)
                .with_invocation(invocation.id.clone()),
        );
        if invocation.deadline_millis == Some(0) {
            let error = KernelError::KernelTimeout;
            self.observe(
                KernelObservation::new(KernelObservationKind::KernelTimeout)
                    .with_kernel(&invocation.kernel)
                    .with_invocation(invocation.id.clone()),
            );
            return KernelResult::failure(invocation.id.clone(), error);
        }
        match advertisement
            .validate_invocation(operator, invocation)
            .and_then(|()| self.run_invocation(invocation))
        {
            Ok(result) => {
                self.observe(
                    KernelObservation::new(KernelObservationKind::KernelDispatchCompleted)
                        .with_kernel(&invocation.kernel)
                        .with_invocation(invocation.id.clone()),
                );
                result
            }
            Err(error) => {
                self.observe(
                    KernelObservation::new(KernelObservationKind::KernelDispatchFailed)
                        .with_kernel(&invocation.kernel)
                        .with_invocation(invocation.id.clone())
                        .with_redacted_metadata("error", error.id()),
                );
                KernelResult::failure(invocation.id.clone(), error)
            }
        }
    }

    fn input_tensor(
        &self,
        invocation: &KernelInvocation,
        index: usize,
    ) -> Result<HostTensor, KernelError> {
        let resource =
            invocation
                .inputs
                .get(index)
                .ok_or_else(|| KernelError::KernelExecutionFailed {
                    reason: format!("missing input at index {index}"),
                })?;
        self.read_tensor(&resource.resource.id)
            .ok_or_else(|| KernelError::KernelExecutionFailed {
                reason: format!(
                    "no materialized data for input resource {}",
                    resource.resource.id
                ),
            })
    }

    fn store_output(
        &self,
        invocation: &KernelInvocation,
        index: usize,
        tensor: HostTensor,
    ) -> Result<TensorResourceDescriptor, KernelError> {
        let resource =
            invocation
                .outputs
                .get(index)
                .ok_or_else(|| KernelError::KernelExecutionFailed {
                    reason: format!("missing output at index {index}"),
                })?;
        self.write_tensor(resource.resource.id.clone(), tensor);
        Ok(resource.resource.clone())
    }

    fn run_invocation(&self, invocation: &KernelInvocation) -> Result<KernelResult, KernelError> {
        let name = invocation.kernel.name.as_str();
        let mut result = KernelResult::success(invocation.id.clone());
        let output = match name {
            "matmul" => {
                let a = self.input_tensor(invocation, 0)?;
                let b = self.input_tensor(invocation, 1)?;
                let transpose_a =
                    Self::attribute_bool(&invocation.attributes, "transpose_a", false);
                let transpose_b =
                    Self::attribute_bool(&invocation.attributes, "transpose_b", false);
                if let Some(OperatorAttributeValue::DType(dtype)) =
                    invocation.attributes.get("accumulation_dtype")
                    && *dtype != ComputeDType::Float32
                {
                    return Err(KernelError::KernelDTypeUnsupported {
                        dtype: format!(
                            "Reference CPU only accumulates in f32; requested {dtype:?}"
                        ),
                    });
                }
                matmul(&a, &b, transpose_a, transpose_b).map_err(KernelError::from)?
            }
            "embedding" => {
                let table = self.input_tensor(invocation, 0)?;
                let ids = self.input_tensor(invocation, 1)?;
                embedding_lookup(&table, &ids).map_err(KernelError::from)?
            }
            "rmsnorm" => {
                let input = self.input_tensor(invocation, 0)?;
                let weight = self.input_tensor(invocation, 1)?;
                let epsilon = Self::attribute_float(&invocation.attributes, "epsilon", 1e-6);
                rmsnorm(&input, &weight, epsilon).map_err(KernelError::from)?
            }
            "rope" => {
                let input = self.input_tensor(invocation, 0)?;
                let base = Self::attribute_float(&invocation.attributes, "base", 10000.0);
                let scale = Self::attribute_float(&invocation.attributes, "scale", 1.0);
                let dimension = Self::attribute_integer(&invocation.attributes, "dimension")
                    .ok_or_else(|| KernelError::KernelAttributeUnsupported {
                        attribute: "dimension".into(),
                    })?;
                // The portable `rope` Operator's attribute schema defines an
                // optional `position_mode` string. Reference CPU only
                // implements the default sequential mode (positions advance by
                // one per row from the offset below); anything else is
                // explicitly rejected rather than silently treated as
                // sequential.
                if let Some(OperatorAttributeValue::String(mode)) =
                    invocation.attributes.get("position_mode")
                    && mode != "sequential"
                {
                    return Err(KernelError::KernelAttributeUnsupported {
                        attribute: format!("position_mode '{mode}' is not implemented"),
                    });
                }
                // Absolute position of the first row. Absent for prefill,
                // where the sequence starts at zero; a decode step carries the
                // number of tokens already in the cache.
                // Read raw rather than through `attribute_integer`, which maps a
                // negative value to `None`. That would make a negative offset
                // indistinguishable from an absent one and silently rotate at
                // position zero, which is the exact failure this attribute
                // exists to prevent.
                let position_offset = match invocation.attributes.get("position_offset") {
                    None => 0,
                    Some(OperatorAttributeValue::Integer(offset)) if *offset >= 0 => *offset as u64,
                    Some(OperatorAttributeValue::Integer(offset)) => {
                        return Err(KernelError::KernelAttributeUnsupported {
                            attribute: format!("position_offset {offset} must not be negative"),
                        });
                    }
                    Some(_) => {
                        return Err(KernelError::KernelAttributeUnsupported {
                            attribute: "position_offset must be an integer".into(),
                        });
                    }
                };
                rope(&input, base, scale, dimension, position_offset).map_err(KernelError::from)?
            }
            "attention" => {
                if cfg!(target_arch = "wasm32") {
                    // Attention requires a Memory-Manager-backed workspace
                    // (see the `Host` workspace requirement on its
                    // advertisement); Host-class workspace allocation is not
                    // meaningful against browser linear memory, so this
                    // Kernel is explicitly unsupported there rather than
                    // silently degraded.
                    return Err(KernelError::KernelBrowserFeatureUnsupported {
                        feature: "reference-cpu-attention-workspace".into(),
                    });
                }
                let q = self.input_tensor(invocation, 0)?;
                let k = self.input_tensor(invocation, 1)?;
                let v = self.input_tensor(invocation, 2)?;
                let head_count = Self::attribute_integer(&invocation.attributes, "head_count")
                    .ok_or_else(|| KernelError::KernelAttributeUnsupported {
                        attribute: "head_count".into(),
                    })?;
                let head_dimension =
                    Self::attribute_integer(&invocation.attributes, "head_dimension").ok_or_else(
                        || KernelError::KernelAttributeUnsupported {
                            attribute: "head_dimension".into(),
                        },
                    )?;
                let kv_head_count =
                    Self::attribute_integer(&invocation.attributes, "kv_head_count");
                let window_size = Self::attribute_integer(&invocation.attributes, "window_size");
                let causal = Self::attribute_bool(&invocation.attributes, "causal", false);
                // `attention_mask_kind` is a portable Operator attribute
                // already in the shared schema. Reference CPU supports the
                // two mask kinds expressible without a dedicated mask
                // tensor input (arity is fixed at q/k/v by the shared
                // Operator schema): "causal" and "bidirectional", and
                // requires them to agree with the `causal` boolean.
                if let Some(OperatorAttributeValue::String(mask_kind)) =
                    invocation.attributes.get("attention_mask_kind")
                {
                    let expected_causal = match mask_kind.as_str() {
                        "causal" => true,
                        "bidirectional" => false,
                        other => {
                            return Err(KernelError::KernelAttributeUnsupported {
                                attribute: format!(
                                    "attention_mask_kind '{other}' is not implemented"
                                ),
                            });
                        }
                    };
                    if expected_causal != causal {
                        return Err(KernelError::KernelAttributeUnsupported {
                            attribute: format!(
                                "attention_mask_kind '{mask_kind}' is inconsistent with causal={causal}"
                            ),
                        });
                    }
                }
                attention(
                    &q,
                    &k,
                    &v,
                    head_count,
                    head_dimension,
                    kv_head_count,
                    window_size,
                    causal,
                )
                .map_err(KernelError::from)?
            }
            "softmax" => {
                let input = self.input_tensor(invocation, 0)?;
                softmax_rows(&input).map_err(KernelError::from)?
            }
            "silu" => silu(&self.input_tensor(invocation, 0)?),
            "gelu" => gelu(&self.input_tensor(invocation, 0)?),
            "activation" => {
                let input = self.input_tensor(invocation, 0)?;
                let kind = match invocation.attributes.get("kind") {
                    Some(OperatorAttributeValue::String(kind)) => kind.as_str(),
                    _ => {
                        return Err(KernelError::KernelAttributeUnsupported {
                            attribute: "kind".into(),
                        });
                    }
                };
                match kind {
                    "silu" => silu(&input),
                    "gelu" => gelu(&input),
                    other => {
                        return Err(KernelError::KernelAttributeUnsupported {
                            attribute: format!("activation kind '{other}' is not implemented"),
                        });
                    }
                }
            }
            "add" => {
                let a = self.input_tensor(invocation, 0)?;
                let b = self.input_tensor(invocation, 1)?;
                add(&a, &b).map_err(KernelError::from)?
            }
            "mul" => {
                let a = self.input_tensor(invocation, 0)?;
                let b = self.input_tensor(invocation, 1)?;
                mul(&a, &b).map_err(KernelError::from)?
            }
            "residual-add" => {
                let input = self.input_tensor(invocation, 0)?;
                let residual = self.input_tensor(invocation, 1)?;
                residual_add(&input, &residual).map_err(KernelError::from)?
            }
            "dtype-conversion" => {
                let input = self.input_tensor(invocation, 0)?;
                dtype_conversion(&input, ComputeDType::Float32, ComputeDType::Float32)
                    .map_err(KernelError::from)?
            }
            "layout-conversion" => {
                let input = self.input_tensor(invocation, 0)?;
                layout_conversion(
                    &input,
                    TensorLayoutKind::Contiguous,
                    TensorLayoutKind::Contiguous,
                )
                .map_err(KernelError::from)?
            }
            "quantize" | "dequantize" => return Err(dequantize_placeholder().into()),
            other => {
                return Err(KernelError::KernelNotFound {
                    kernel: other.into(),
                });
            }
        };
        let descriptor = self.store_output(invocation, 0, output)?;
        result
            .output_readiness
            .insert(descriptor.id.to_string(), true);
        result.updated_resources.push(descriptor);
        Ok(result)
    }
}

impl ProviderExecutionApi for ReferenceCpuExecutor {
    fn submit(
        &self,
        request: ProviderExecutionRequest,
    ) -> Result<ProviderExecutionHandle, ProviderExecutionError> {
        let handle = ProviderExecutionHandle::new(
            request.operation,
            request.plan.id.clone(),
            request.provider.clone(),
            request.device.clone(),
        );
        self.submitted
            .lock()
            .unwrap()
            .insert(handle.id.clone(), request);
        Ok(handle)
    }

    fn status(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionStatus, ProviderExecutionError> {
        if !self.submitted.lock().unwrap().contains_key(&handle.id) {
            return Err(ProviderExecutionError::new(
                ProviderExecutionErrorCode::ExecutionFailed,
                ProviderExecutionPhase::Observe,
                handle.provider.clone(),
                handle.device.clone(),
                "no submission is associated with this handle: it was never submitted, \
                 or has already been completed and released",
            ));
        }
        Ok(ProviderExecutionStatus::new(
            handle.clone(),
            SchedulingState::Completed,
        ))
    }

    fn cancel(
        &self,
        _handle: &ProviderExecutionHandle,
    ) -> Result<ProviderCancellationOutcome, ProviderExecutionError> {
        Ok(ProviderCancellationOutcome::Unsupported)
    }

    fn complete(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionResult, ProviderExecutionError> {
        // Single consumption: a handle's submission is removed the first
        // time it is completed, so a second `complete()` call on the same
        // handle -- or a handle that was never `submit()`-ted -- is
        // rejected rather than silently echoed back as evidence of work
        // that never causally happened.
        self.submitted
            .lock()
            .unwrap()
            .remove(&handle.id)
            .ok_or_else(|| {
                ProviderExecutionError::new(
                    ProviderExecutionErrorCode::ExecutionFailed,
                    ProviderExecutionPhase::Complete,
                    handle.provider.clone(),
                    handle.device.clone(),
                    "no submission is associated with this handle: it was never \
                     submitted, or has already been completed once",
                )
            })?;
        Ok(ProviderExecutionResult::completed(
            handle.clone(),
            Vec::new(),
        ))
    }

    fn release(&self, handle: ProviderExecutionHandle) -> Result<(), ProviderExecutionError> {
        self.submitted.lock().unwrap().remove(&handle.id);
        Ok(())
    }

    fn submit_kernel(
        &self,
        advertisement: &KernelAdvertisement,
        operator: &OperatorSpec,
        invocation: &KernelInvocation,
        memory: &mut MemoryManager,
    ) -> Result<ProviderExecutionHandle, ProviderExecutionError> {
        Ok(self.submit_kernel_invocation(advertisement, operator, invocation, memory))
    }

    fn complete_kernel(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<KernelResult, ProviderExecutionError> {
        self.complete_kernel_invocation(handle)
    }

    fn write_tensor(&self, id: TensorResourceId, tensor: HostTensor) {
        ReferenceCpuExecutor::write_tensor(self, id, tensor)
    }

    fn read_tensor(&self, id: &TensorResourceId) -> Option<HostTensor> {
        ReferenceCpuExecutor::read_tensor(self, id)
    }

    fn release_tensor(&self, id: &TensorResourceId) -> bool {
        ReferenceCpuExecutor::release_tensor(self, id)
    }

    fn release_admitted_tensor(&self, memory: &mut MemoryManager, id: &TensorResourceId) -> bool {
        ReferenceCpuExecutor::release_admitted_tensor(self, memory, id)
    }

    fn write_tensor_admitted(
        &self,
        memory: &mut MemoryManager,
        resource_id: TensorResourceId,
        tensor: HostTensor,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<(), MemoryError> {
        ReferenceCpuExecutor::write_tensor_admitted(self, memory, resource_id, tensor, class, owner)
    }

    fn read_tensor_value(&self, id: &TensorResourceId) -> Option<TensorValue> {
        ReferenceCpuExecutor::read_tensor_value(self, id)
    }

    fn write_tensor_value(&self, id: TensorResourceId, value: TensorValue) {
        ReferenceCpuExecutor::write_tensor_value(self, id, value)
    }

    fn write_tensor_value_admitted(
        &self,
        memory: &mut MemoryManager,
        resource_id: TensorResourceId,
        value: TensorValue,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<(), MemoryError> {
        ReferenceCpuExecutor::write_tensor_value_admitted(
            self,
            memory,
            resource_id,
            value,
            class,
            owner,
        )
    }

    fn allocate_workspace(
        &self,
        memory: &mut MemoryManager,
        size_bytes: u64,
    ) -> Result<MemoryAllocationId, MemoryError> {
        ReferenceCpuExecutor::allocate_workspace(self, memory, size_bytes)
    }

    fn observations(&self) -> Vec<KernelObservation> {
        ReferenceCpuExecutor::observations(self)
    }
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// The Reference CPU Provider itself: correctness baseline for portable
/// Operator semantics, enabled explicitly by policy (development, test,
/// conformance, local-runtime, fallback-policy).
pub struct ReferenceCpuProvider {
    metadata: ProviderMetadata,
    device: Arc<DeviceDescriptor>,
    executor: Arc<ReferenceCpuExecutor>,
    features: ReferenceCpuFeatureFlags,
    pressure: Mutex<ProviderPressureLevel>,
}

impl ReferenceCpuProvider {
    pub fn new() -> Self {
        Self::with_features(ReferenceCpuFeatureFlags::baseline())
    }

    pub fn with_features(features: ReferenceCpuFeatureFlags) -> Self {
        Self {
            metadata: reference_cpu_provider_metadata(),
            device: Arc::new(reference_cpu_device()),
            executor: Arc::new(ReferenceCpuExecutor::new()),
            features,
            pressure: Mutex::new(ProviderPressureLevel::Low),
        }
    }

    pub fn features(&self) -> ReferenceCpuFeatureFlags {
        self.features
    }

    pub fn executor(&self) -> Arc<ReferenceCpuExecutor> {
        self.executor.clone()
    }

    /// Reports this Provider's current pressure. Reference CPU has no
    /// automatic load model of its own (it executes synchronously with no
    /// queue), so pressure is reported explicitly by whoever observes real
    /// load (e.g. the Runtime, from concurrent invocation counts) rather
    /// than silently defaulting to `Low` regardless of actual usage.
    pub fn report_pressure(&self, level: ProviderPressureLevel) {
        *self.pressure.lock().unwrap() = level;
    }
}

impl Default for ReferenceCpuProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Provider for ReferenceCpuProvider {
    fn metadata(&self) -> ProviderMetadata {
        self.metadata.clone()
    }

    fn register(&self, _registry: &mut ProviderRegistry) -> Result<(), ProviderError> {
        // Device registration is performed by the Runtime/ProviderLoader via
        // `devices()`; registering here too would double-register them.
        Ok(())
    }

    fn health(&self) -> ProviderHealth {
        ProviderHealth::Available
    }

    fn status_snapshot(&self) -> ProviderStatusSnapshot {
        let mut snapshot = ProviderStatusSnapshot::from_health_report(self.health_report());
        snapshot.pressure = *self.pressure.lock().unwrap();
        snapshot
    }

    fn devices(&self) -> Vec<Arc<dyn Device>> {
        vec![self.device.clone()]
    }

    fn kernel_advertisements(&self) -> Vec<KernelAdvertisement> {
        let mut advertisements = reference_cpu_kernel_advertisements();
        if !self.features.attention {
            advertisements.retain(|advertisement| advertisement.id.name != "attention");
        }
        if !self.features.rope {
            advertisements.retain(|advertisement| advertisement.id.name != "rope");
        }
        advertisements
    }

    fn initialize(&self) -> Result<(), ProviderError> {
        // No Runtime-wide provider/device registration observability channel
        // exists yet for any Provider (see `design.md`), so these are
        // recorded on this Provider's own executor, inspectable by tests and
        // by any caller holding the executor handle.
        self.executor.observe(
            KernelObservation::new(KernelObservationKind::ProviderRegistered)
                .with_redacted_metadata("provider", REFERENCE_CPU_PROVIDER_NAME),
        );
        self.executor.observe(
            KernelObservation::new(KernelObservationKind::DeviceDetected)
                .with_redacted_metadata("device", REFERENCE_CPU_DEVICE_ID),
        );
        Ok(())
    }

    fn shutdown(&self) -> Result<(), ProviderError> {
        Ok(())
    }

    fn execution_api(&self) -> Option<Arc<dyn ProviderExecutionApi>> {
        Some(self.executor.clone())
    }
}

#[cfg(test)]
mod tests;
