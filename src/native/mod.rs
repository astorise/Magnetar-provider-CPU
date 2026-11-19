//! The optimized CPU Provider (`execute-quantized-weights-on-cpu`): a
//! Provider identity of its own, beside the unchanged Reference CPU
//! Provider, that keeps packed weights in their checkpoint encoding and
//! multiplies them with int8 activations across a persistent thread pool
//! ([`quant`]). Every operator it does not optimize runs the Reference
//! CPU kernels, through a [`ReferenceCpuExecutor`] it wraps.
//!
//! Controls (read once per Provider):
//! - [`PACKED_WEIGHTS_ENV`] `=0` declines packed weights (f32 at load);
//! - [`INT8_ACTIVATIONS_ENV`] `=0` keeps f32 activations;
//! - [`THREADS_ENV`] bounds the worker pool;
//! - [`SIMD_ENV`] `=scalar` disables the vector kernels;
//! - [`COMPACT_WEIGHTS_ENV`] `=0` keeps every dense weight f32 instead of
//!   storing an exact bfloat16 one as its bfloat16 bits ([`dense`]);
//! - [`STEP_CAPTURE_ENV`] `=0` declines step capture ([`capture`]).

mod capture;
mod dense;
mod pool;
pub mod quant;

use super::*;
use magnetar_provider_api::{packed_weight::PackedWeight, packed_weight::PackedWeightWrite};
use std::sync::OnceLock;

/// Stable, package-qualified optimized CPU Provider identity.
pub use magnetar_provider_api::NATIVE_CPU_PROVIDER_NAME;
pub const NATIVE_CPU_DEVICE_ID: &str = "cpu-native:host:0";
pub const NATIVE_CPU_PROVIDER_VERSION: &str = "0.1.0";
pub const PACKED_WEIGHTS_ENV: &str = "MAGNETAR_CPU_PACKED_WEIGHTS";
pub const INT8_ACTIVATIONS_ENV: &str = "MAGNETAR_CPU_INT8_ACTIVATIONS";
pub const THREADS_ENV: &str = "MAGNETAR_CPU_THREADS";
pub const SIMD_ENV: &str = "MAGNETAR_CPU_SIMD";
pub const COMPACT_WEIGHTS_ENV: &str = "MAGNETAR_CPU_COMPACT_WEIGHTS";
pub const STEP_CAPTURE_ENV: &str = "MAGNETAR_CPU_STEP_CAPTURE";
/// Dense weights at least this large are stored compact when every value
/// is exactly a bfloat16 value; smaller ones (biases, norm weights) stay
/// f32, as on the CUDA Provider.
pub const COMPACT_WEIGHT_MIN_ELEMENTS: usize = 1 << 16;
/// Set to `1` to time every Kernel by name and print the totals when the
/// executor drops (a diagnostic).
pub const PROFILE_ENV: &str = "MAGNETAR_CPU_PROFILE";

/// The vector instruction level the kernels use.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum SimdLevel {
    Scalar,
    Avx2,
}

/// The detected level, capped by [`SIMD_ENV`] (`scalar`).
pub fn simd_level() -> SimdLevel {
    static LEVEL: OnceLock<SimdLevel> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        if std::env::var(SIMD_ENV).as_deref() == Ok("scalar") {
            return SimdLevel::Scalar;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("fma")
                && std::is_x86_feature_detected!("f16c")
            {
                return SimdLevel::Avx2;
            }
        }
        SimdLevel::Scalar
    })
}

/// What a [`ReferenceCpuExecutor`] wrapped by the optimized CPU Provider
/// consults: packed weights by resource id and the kernels that read them.
pub(crate) trait PackedWeights: Send + Sync {
    /// Runs `invocation` if it reads a packed weight this Provider computes
    /// from directly; `None` lets the Reference CPU kernel run.
    fn run(
        &self,
        executor: &ReferenceCpuExecutor,
        invocation: &KernelInvocation,
    ) -> Option<Result<HostTensor, KernelError>>;
    /// A packed weight's logical values, for a kernel without a packed path.
    fn materialize(&self, id: &TensorResourceId) -> Option<HostTensor>;
    fn shape(&self, id: &TensorResourceId) -> Option<Vec<u64>>;
}

/// A dense weight whose every value is exactly a bfloat16 value, held as
/// those values' bfloat16 bits (`speed-up-dense-cpu-matmuls`): half the
/// bytes of f32, every value unchanged.
struct CompactWeight {
    rows: usize,
    cols: usize,
    bits: Vec<u16>,
}

impl CompactWeight {
    /// `tensor` compact, when it is a matrix of at least
    /// [`COMPACT_WEIGHT_MIN_ELEMENTS`] exact bfloat16 values.
    fn of(tensor: &HostTensor) -> Option<Self> {
        let (rows, cols) = tensor.rows_cols().ok()?;
        if tensor.shape.len() != 2 || tensor.data.len() < COMPACT_WEIGHT_MIN_ELEMENTS {
            return None;
        }
        let bits = tensor
            .data
            .iter()
            .map(|&value| dense::exact_bf16_bits(value))
            .collect::<Option<Vec<u16>>>()?;
        Some(Self {
            rows: rows as usize,
            cols: cols as usize,
            bits,
        })
    }

    fn expand(&self) -> HostTensor {
        let values: Vec<f32> = self
            .bits
            .iter()
            .map(|&bits| f32::from_bits(u32::from(bits) << 16))
            .collect();
        HostTensor::new([self.rows as u64, self.cols as u64], values)
            .expect("a compact weight's shape holds its values")
    }
}

struct NativeState {
    packed: Mutex<BTreeMap<TensorResourceId, Arc<PackedWeight>>>,
    compact: Mutex<BTreeMap<TensorResourceId, Arc<CompactWeight>>>,
    pool: pool::Pool,
    int8_activations: bool,
    /// Packed weights expanded to f32 for a caller or Kernel without a
    /// packed path: count and seconds (reported with [`PROFILE_ENV`]).
    materialized: Mutex<(u64, f64)>,
}

/// A dot product in eight independent lanes, so it vectorizes.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0_f32; 8];
    let (a8, a_rest) = a.as_chunks::<8>();
    let (b8, b_rest) = b.as_chunks::<8>();
    for (x, y) in a8.iter().zip(b8) {
        for i in 0..8 {
            lanes[i] += x[i] * y[i];
        }
    }
    let tail: f32 = a_rest.iter().zip(b_rest).map(|(x, y)| x * y).sum();
    lanes.iter().sum::<f32>() + tail
}

/// `*mut f32` shared across pool tasks that write disjoint elements.
struct SharedOut(*mut f32);
// SAFETY: every task writes a disjoint set of output elements.
unsafe impl Sync for SharedOut {}

impl NativeState {
    fn weight(&self, id: &TensorResourceId) -> Option<Arc<PackedWeight>> {
        self.packed.lock().unwrap().get(id).cloned()
    }

    fn compact(&self, id: &TensorResourceId) -> Option<Arc<CompactWeight>> {
        self.compact.lock().unwrap().get(id).cloned()
    }

    /// A matmul-family Kernel's sum or residual, after its product.
    fn finish_matmul(
        executor: &ReferenceCpuExecutor,
        invocation: &KernelInvocation,
        product: HostTensor,
    ) -> Result<HostTensor, KernelError> {
        match invocation.kernel.name.as_str() {
            "matmul-bias" => {
                add(&product, &executor.input_tensor(invocation, 2)?).map_err(KernelError::from)
            }
            "matmul-residual" => residual_add(&executor.input_tensor(invocation, 2)?, &product)
                .map_err(KernelError::from),
            _ => Ok(product),
        }
    }

    /// A matmul-family Kernel whose weight (input 1) is dense: compact or
    /// f32 (`speed-up-dense-cpu-matmuls`), across the pool. `None` hands
    /// any form it does not handle (a transposed `a`, mismatched shapes) to
    /// the Reference CPU kernel, which reports any error.
    fn dense_matmul(
        &self,
        executor: &ReferenceCpuExecutor,
        invocation: &KernelInvocation,
        id: &TensorResourceId,
    ) -> Option<Result<HostTensor, KernelError>> {
        if cfg!(target_arch = "wasm32")
            || ReferenceCpuExecutor::attribute_bool(&invocation.attributes, "transpose_a", false)
            || ReferenceCpuExecutor::require_f32_accumulation(&invocation.attributes).is_err()
        {
            return None;
        }
        let transpose_b =
            ReferenceCpuExecutor::attribute_bool(&invocation.attributes, "transpose_b", false);
        let a = match executor.input_tensor(invocation, 0) {
            Ok(a) => a,
            Err(error) => return Some(Err(error)),
        };
        let product = match self.compact(id) {
            Some(weight) => self.dense_product(
                &a,
                dense::Values::Bf16(&weight.bits),
                [weight.rows, weight.cols],
                transpose_b,
            )?,
            None => {
                let storage = executor.storage.lock().unwrap();
                let b = storage.get(id)?;
                let (rows, cols) = b.rows_cols().ok()?;
                if b.shape.len() != 2 {
                    return None;
                }
                self.dense_product(
                    &a,
                    dense::Values::F32(&b.data),
                    [rows as usize, cols as usize],
                    transpose_b,
                )?
            }
        };
        Some(product.and_then(|product| Self::finish_matmul(executor, invocation, product)))
    }

    /// `a x b` for a dense `b` of stored shape `[rows, cols]`: logically
    /// `[k, n]`, or `[n, k]` used transposed.
    fn dense_product(
        &self,
        a: &HostTensor,
        values: dense::Values<'_>,
        [rows, cols]: [usize; 2],
        transpose_b: bool,
    ) -> Option<Result<HostTensor, KernelError>> {
        let (m, k) = a.rows_cols().ok()?;
        let (m, k) = (m as usize, k as usize);
        if a.shape.len() != 2 {
            return None;
        }
        let (inner, n) = if transpose_b {
            (cols, rows)
        } else {
            (rows, cols)
        };
        if inner != k || m == 0 || n == 0 || k == 0 {
            return None;
        }
        let level = simd_level();
        let threads = self.pool.threads();
        let mut out = vec![0.0_f32; m * n];
        // An input-major weight narrower than `INPUT_MAJOR_COLUMNS` per
        // thread is shared out by blocks of whole rows (contiguous reads),
        // each task summing its block into a partial product: a block of a
        // few hundred columns per row would read short strided runs.
        const INPUT_MAJOR_COLUMNS: usize = 2048;
        if !transpose_b && n < threads * INPUT_MAJOR_COLUMNS && threads > 1 {
            let tasks = threads.min(k.div_ceil(64)).max(1);
            let block = k.div_ceil(tasks);
            let mut partials = vec![0.0_f32; tasks * m * n];
            let shared = SharedOut(partials.as_mut_ptr());
            let shared = &shared;
            self.pool.run(tasks, &|task| {
                let rows = (task * block).min(k)..((task + 1) * block).min(k);
                // SAFETY: partial `task` belongs to this task only.
                let partial =
                    unsafe { std::slice::from_raw_parts_mut(shared.0.add(task * m * n), m * n) };
                dense::input_major(level, &a.data, m, k, rows, values, n, 0, n, partial);
            });
            let (first, rest) = partials.split_at(m * n);
            out.copy_from_slice(first);
            for partial in rest.chunks_exact(m * n) {
                for (out, &value) in out.iter_mut().zip(partial) {
                    *out += value;
                }
            }
            return Some(HostTensor::new([m as u64, n as u64], out).map_err(KernelError::from));
        }
        let chunk = if transpose_b {
            n.div_ceil(threads * 4).max(16)
        } else {
            n.div_ceil(threads * 2).max(INPUT_MAJOR_COLUMNS)
        }
        .next_multiple_of(8);
        let tasks = n.div_ceil(chunk);
        let shared = SharedOut(out.as_mut_ptr());
        let shared = &shared;
        self.pool.run(tasks, &|task| {
            let first = task * chunk;
            let count = chunk.min(n - first);
            let mut local = vec![0.0_f32; m * count];
            if transpose_b {
                dense::output_major(level, &a.data, m, k, values, first, count, &mut local);
            } else {
                dense::input_major(
                    level,
                    &a.data,
                    m,
                    k,
                    0..k,
                    values,
                    n,
                    first,
                    count,
                    &mut local,
                );
            }
            for r in 0..m {
                for c in 0..count {
                    // SAFETY: column `first + c` belongs to this task only.
                    unsafe { *shared.0.add(r * n + first + c) = local[r * count + c] };
                }
            }
        });
        Some(HostTensor::new([m as u64, n as u64], out).map_err(KernelError::from))
    }

    /// The `embedding` Kernel over a compact table: each gathered row's
    /// values, exactly.
    fn compact_embedding(
        executor: &ReferenceCpuExecutor,
        invocation: &KernelInvocation,
        table: &CompactWeight,
    ) -> Result<HostTensor, KernelError> {
        let ids = executor.input_tensor(invocation, 1)?;
        let (vocab, dim) = (table.rows as u64, table.cols);
        let mut out = Vec::with_capacity(ids.data.len() * dim);
        for &raw_id in &ids.data {
            if raw_id < 0.0 || raw_id.fract() != 0.0 || raw_id as u64 >= vocab {
                return Err(KernelError::KernelExecutionFailed {
                    reason: format!("token id {raw_id} is not below the vocabulary size {vocab}"),
                });
            }
            let start = raw_id as usize * dim;
            out.extend(
                table.bits[start..start + dim]
                    .iter()
                    .map(|&bits| f32::from_bits(u32::from(bits) << 16)),
            );
        }
        HostTensor::new([ids.data.len() as u64, dim as u64], out).map_err(KernelError::from)
    }

    /// `a x W` for a packed `W` whose stored rows are the output columns.
    fn packed_matmul(
        &self,
        a: &HostTensor,
        weight: &PackedWeight,
    ) -> Result<HostTensor, KernelError> {
        let (m, k) = a.rows_cols().map_err(KernelError::from)?;
        let (m, k) = (m as usize, k as usize);
        let n = weight.stored_rows() as usize;
        if k != weight.stored_cols() as usize || !k.is_multiple_of(quant::RUN) {
            return Err(KernelError::KernelExecutionFailed {
                reason: format!(
                    "packed matmul: activations have {k} columns, the weight's stored rows {}",
                    weight.stored_cols()
                ),
            });
        }
        let quantized = self.int8_activations.then(|| quant::quantize_rows(&a.data));
        let reader = quant::RunReader::new(weight);
        let activations = quant::Activations {
            rows: &a.data,
            quantized: quantized.as_ref(),
            m,
            k,
        };
        let mut out = vec![0.0_f32; m * n];
        let chunk = n.div_ceil(self.pool.threads() * 4).max(8);
        let tasks = n.div_ceil(chunk);
        let shared = SharedOut(out.as_mut_ptr());
        let shared = &shared;
        self.pool.run(tasks, &|task| {
            let first = task * chunk;
            let count = chunk.min(n - first);
            let mut local = vec![0.0_f32; m * count];
            quant::matmul_columns(&reader, &activations, first, count, &mut local);
            for r in 0..m {
                for c in 0..count {
                    // SAFETY: column `first + c` belongs to this task only.
                    unsafe { *shared.0.add(r * n + first + c) = local[r * count + c] };
                }
            }
        });
        HostTensor::new([m as u64, n as u64], out).map_err(KernelError::from)
    }

    /// `a x W` for a packed `W` whose stored rows are the input rows.
    fn packed_axpy(
        &self,
        a: &HostTensor,
        weight: &PackedWeight,
    ) -> Result<HostTensor, KernelError> {
        let (m, k) = a.rows_cols().map_err(KernelError::from)?;
        let (m, k) = (m as usize, k as usize);
        let n = weight.stored_cols() as usize;
        if k != weight.stored_rows() as usize {
            return Err(KernelError::KernelExecutionFailed {
                reason: format!(
                    "packed matmul: activations have {k} columns, the weight {} stored rows",
                    weight.stored_rows()
                ),
            });
        }
        let reader = quant::RunReader::new(weight);
        let runs = n / quant::RUN;
        let chunk = runs.div_ceil(self.pool.threads() * 4).max(1);
        let tasks = runs.div_ceil(chunk);
        let mut out = vec![0.0_f32; m * n];
        let shared = SharedOut(out.as_mut_ptr());
        let shared = &shared;
        self.pool.run(tasks, &|task| {
            let first_run = task * chunk;
            let run_count = chunk.min(runs - first_run);
            let width = run_count * quant::RUN;
            let mut local = vec![0.0_f32; m * width];
            quant::axpy_columns(&reader, &a.data, m, k, first_run, run_count, &mut local);
            for r in 0..m {
                for c in 0..width {
                    // SAFETY: these output columns belong to this task only.
                    unsafe {
                        *shared.0.add(r * n + first_run * quant::RUN + c) = local[r * width + c]
                    };
                }
            }
        });
        HostTensor::new([m as u64, n as u64], out).map_err(KernelError::from)
    }

    /// A packed weight's logical f32 values, counted as a materialization.
    fn expand(&self, weight: &PackedWeight) -> HostTensor {
        let started = std::time::Instant::now();
        let tensor = weight.dequantize();
        let mut materialized = self.materialized.lock().unwrap();
        materialized.0 += 1;
        materialized.1 += started.elapsed().as_secs_f64();
        tensor
    }

    fn matmul(
        &self,
        executor: &ReferenceCpuExecutor,
        invocation: &KernelInvocation,
        weight: &PackedWeight,
    ) -> Result<HostTensor, KernelError> {
        let a = executor.input_tensor(invocation, 0)?;
        let transpose_a =
            ReferenceCpuExecutor::attribute_bool(&invocation.attributes, "transpose_a", false);
        let transpose_b =
            ReferenceCpuExecutor::attribute_bool(&invocation.attributes, "transpose_b", false);
        ReferenceCpuExecutor::require_f32_accumulation(&invocation.attributes)?;
        // The stored rows are the output columns when the logical `[k, n]`
        // weight is stored transposed, or an untransposed `[n, k]` one
        // (a tied embedding table) is used transposed.
        let product = if transpose_a {
            matmul(&a, &self.expand(weight), transpose_a, transpose_b).map_err(KernelError::from)?
        } else if weight.transposed() != transpose_b {
            self.packed_matmul(&a, weight)?
        } else {
            self.packed_axpy(&a, weight)?
        };
        Self::finish_matmul(executor, invocation, product)
    }

    fn f32_embedding(
        executor: &ReferenceCpuExecutor,
        invocation: &KernelInvocation,
        id: &TensorResourceId,
    ) -> Result<HostTensor, KernelError> {
        let ids = executor.input_tensor(invocation, 1)?;
        let storage = executor.storage.lock().unwrap();
        let table = storage
            .get(id)
            .ok_or_else(|| KernelError::KernelExecutionFailed {
                reason: format!("no materialized data for input resource {id}"),
            })?;
        let (vocab, dim) = table.rows_cols().map_err(KernelError::from)?;
        let dim = dim as usize;
        let mut out = Vec::with_capacity(ids.data.len() * dim);
        for &raw_id in &ids.data {
            if raw_id < 0.0 || raw_id.fract() != 0.0 || raw_id as u64 >= vocab {
                return Err(KernelError::KernelExecutionFailed {
                    reason: format!("token id {raw_id} is not below the vocabulary size {vocab}"),
                });
            }
            let start = raw_id as usize * dim;
            out.extend_from_slice(&table.data[start..start + dim]);
        }
        HostTensor::new([ids.data.len() as u64, dim as u64], out).map_err(KernelError::from)
    }

    /// The `attention` Kernel, multithreaded and reading K and V in place
    /// (the Reference CPU kernel copies both whole caches per call): every
    /// (head, query) pair is independent, so the pool shares them out.
    /// `None` hands any case it does not handle, or that the Reference CPU
    /// kernel rejects, to that kernel, which reports the error.
    fn attention(
        &self,
        executor: &ReferenceCpuExecutor,
        invocation: &KernelInvocation,
    ) -> Option<Result<HostTensor, KernelError>> {
        if cfg!(target_arch = "wasm32") {
            return None;
        }
        let attributes = &invocation.attributes;
        let head_count = ReferenceCpuExecutor::attribute_integer(attributes, "head_count")?;
        let head_dimension = ReferenceCpuExecutor::attribute_integer(attributes, "head_dimension")?;
        let kv_head_count = ReferenceCpuExecutor::attribute_integer(attributes, "kv_head_count")
            .unwrap_or(head_count);
        let window = ReferenceCpuExecutor::attribute_integer(attributes, "window_size");
        let causal = ReferenceCpuExecutor::attribute_bool(attributes, "causal", false);
        match attributes.get("attention_mask_kind") {
            None => {}
            Some(OperatorAttributeValue::String(kind))
                if (kind == "causal" && causal) || (kind == "bidirectional" && !causal) => {}
            Some(_) => return None,
        }
        // V rows of another width than Q/K rows, V heads wider than the
        // value, or latent K and V: the Reference CPU kernel.
        if ReferenceCpuExecutor::attribute_integer(attributes, "value_head_dimension")
            .is_some_and(|width| width != head_dimension)
            || attributes.contains_key("value_head_stride")
            || attributes.contains_key("latent_kv_width")
        {
            return None;
        }
        let score_scale = ReferenceCpuExecutor::attribute_float_opt(attributes, "scale");
        if score_scale.is_some_and(|scale| !scale.is_finite())
            || head_count == 0
            || head_dimension == 0
            || kv_head_count == 0
            || !head_count.is_multiple_of(kv_head_count)
            || window == Some(0)
            || (window.is_some() && !causal)
        {
            return None;
        }
        let q = match executor.input_tensor(invocation, 0) {
            Ok(q) => q,
            Err(error) => return Some(Err(error)),
        };
        let storage = executor.storage.lock().unwrap();
        let k = storage.get(&invocation.inputs.get(1)?.resource.id)?;
        let v = storage.get(&invocation.inputs.get(2)?.resource.id)?;
        let (seq_len, q_width) = q.rows_cols().ok()?;
        let (k_rows, kv_width) = k.rows_cols().ok()?;
        if v.shape != k.shape {
            return None;
        }
        let kv_len = match ReferenceCpuExecutor::attribute_integer(attributes, "key_count") {
            Some(count) if count <= k_rows => count,
            Some(_) => return None,
            None => k_rows,
        };
        if seq_len > kv_len
            || head_count * head_dimension != q_width
            || kv_head_count * head_dimension != kv_width
        {
            return None;
        }
        let (seq_len, kv_len) = (seq_len as usize, kv_len as usize);
        let (q_width, kv_width, dim) =
            (q_width as usize, kv_width as usize, head_dimension as usize);
        let group = (head_count / kv_head_count) as usize;
        let scale = score_scale.unwrap_or(1.0 / (dim as f32).sqrt());
        let offset = kv_len - seq_len;
        let pairs = head_count as usize * seq_len;
        let mut out = vec![0.0_f32; seq_len * q_width];
        let chunk = pairs.div_ceil(self.pool.threads() * 4).max(1);
        let shared = SharedOut(out.as_mut_ptr());
        let shared = &shared;
        let (q, k, v) = (&q.data, &k.data, &v.data);
        self.pool.run(pairs.div_ceil(chunk), &|task| {
            let mut weights = Vec::with_capacity(kv_len);
            let mut row = vec![0.0_f32; dim];
            for pair in task * chunk..((task + 1) * chunk).min(pairs) {
                let (head, query) = (pair / seq_len, pair % seq_len);
                let kv_offset = (head / group) * dim;
                let position = offset + query;
                let upper = if causal { position + 1 } else { kv_len };
                let lower = window
                    .map(|window| position.saturating_sub((window as usize).saturating_sub(1)))
                    .unwrap_or(0)
                    .min(upper);
                let query_row = &q[query * q_width + head * dim..][..dim];
                weights.clear();
                weights
                    .extend((lower..upper).map(|key| {
                        dot(query_row, &k[key * kv_width + kv_offset..][..dim]) * scale
                    }));
                let max = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0_f32;
                for weight in &mut weights {
                    *weight = (*weight - max).exp();
                    sum += *weight;
                }
                row.iter_mut().for_each(|value| *value = 0.0);
                for (key, &weight) in (lower..upper).zip(&weights) {
                    let value = &v[key * kv_width + kv_offset..][..dim];
                    let weight = weight / sum;
                    for (out, &x) in row.iter_mut().zip(value) {
                        *out += weight * x;
                    }
                }
                for (d, &value) in row.iter().enumerate() {
                    // SAFETY: element (query, head * dim + d) belongs to
                    // this pair only.
                    unsafe { *shared.0.add(query * q_width + head * dim + d) = value };
                }
            }
        });
        drop(storage);
        Some(HostTensor::new([seq_len as u64, q_width as u64], out).map_err(KernelError::from))
    }

    fn embedding(
        &self,
        executor: &ReferenceCpuExecutor,
        invocation: &KernelInvocation,
        table: &PackedWeight,
    ) -> Result<HostTensor, KernelError> {
        let ids = executor.input_tensor(invocation, 1)?;
        let [vocab, dim] = table.logical_shape();
        let dim = dim as usize;
        let reader = quant::RunReader::new(table);
        let mut out = vec![0.0_f32; ids.data.len() * dim];
        for (&raw_id, row) in ids.data.iter().zip(out.chunks_exact_mut(dim)) {
            if raw_id < 0.0 || raw_id.fract() != 0.0 || raw_id as u64 >= vocab {
                return Err(KernelError::KernelExecutionFailed {
                    reason: format!("token id {raw_id} is not below the vocabulary size {vocab}"),
                });
            }
            let token = raw_id as usize;
            if table.transposed() {
                // Stored `[dim][vocab]`: the token's value in every stored row.
                for (j, value) in row.iter_mut().enumerate() {
                    *value = quant::element(&reader, j, token);
                }
            } else {
                quant::dequantize_row(&reader, token, dim, row);
            }
        }
        HostTensor::new([ids.data.len() as u64, dim as u64], out).map_err(KernelError::from)
    }
}

impl PackedWeights for NativeState {
    fn run(
        &self,
        executor: &ReferenceCpuExecutor,
        invocation: &KernelInvocation,
    ) -> Option<Result<HostTensor, KernelError>> {
        match invocation.kernel.name.as_str() {
            "matmul" | "matmul-bias" | "matmul-residual" => {
                let id = &invocation.inputs.get(1)?.resource.id;
                match self.weight(id) {
                    Some(weight) => Some(self.matmul(executor, invocation, &weight)),
                    None => self.dense_matmul(executor, invocation, id),
                }
            }
            "attention" => self.attention(executor, invocation),
            "embedding" => {
                let id = &invocation.inputs.first()?.resource.id;
                let rows = match (self.weight(id), self.compact(id)) {
                    (Some(table), _) => self.embedding(executor, invocation, &table),
                    (None, Some(table)) => Self::compact_embedding(executor, invocation, &table),
                    // An f32 table (one this Provider received dequantized):
                    // gather its rows in place instead of the Reference
                    // CPU kernel's copy of the whole table per lookup.
                    (None, None) => Self::f32_embedding(executor, invocation, id),
                };
                // `scale` multiplies every gathered row.
                let scale =
                    ReferenceCpuExecutor::attribute_float_opt(&invocation.attributes, "scale");
                Some(rows.and_then(|rows| match scale {
                    Some(factor) => {
                        operator_semantics::scale(&rows, factor).map_err(KernelError::from)
                    }
                    None => Ok(rows),
                }))
            }
            _ => None,
        }
    }

    fn materialize(&self, id: &TensorResourceId) -> Option<HostTensor> {
        if let Some(weight) = self.weight(id) {
            return Some(self.expand(&weight));
        }
        let weight = self.compact(id)?;
        let started = std::time::Instant::now();
        let tensor = weight.expand();
        let mut materialized = self.materialized.lock().unwrap();
        materialized.0 += 1;
        materialized.1 += started.elapsed().as_secs_f64();
        Some(tensor)
    }

    fn shape(&self, id: &TensorResourceId) -> Option<Vec<u64>> {
        if let Some(weight) = self.weight(id) {
            return Some(weight.logical_shape().to_vec());
        }
        self.compact(id)
            .map(|weight| vec![weight.rows as u64, weight.cols as u64])
    }
}

/// The optimized CPU Provider's executor: a [`ReferenceCpuExecutor`]
/// reporting this Provider's identity, with packed weights and their
/// kernels.
pub struct NativeCpuExecutor {
    inner: Arc<ReferenceCpuExecutor>,
    state: Arc<NativeState>,
    packed_weights: bool,
    /// Whether exact bfloat16 dense weights are stored compact.
    compact_weights: bool,
    /// Whether steps are captured and replayed ([`capture`]).
    step_capture: bool,
    capture: Mutex<capture::CaptureState>,
    /// Per-Kernel invocation counts and seconds, when [`PROFILE_ENV`] is set.
    profile: Option<Mutex<BTreeMap<String, (u64, f64)>>>,
}

impl Drop for NativeCpuExecutor {
    fn drop(&mut self) {
        let Some(profile) = &self.profile else {
            return;
        };
        let (count, seconds) = *self.state.materialized.lock().unwrap();
        eprintln!(
            "[cpu-native profile] packed weights expanded to f32: {count}, {:.1} ms",
            seconds * 1e3
        );
        let profile = profile.lock().unwrap();
        let mut rows: Vec<_> = profile.iter().collect();
        rows.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
        for (name, (count, seconds)) in rows {
            eprintln!(
                "[cpu-native profile] {name}: {count} calls, {:.1} ms total, {:.3} ms each",
                seconds * 1e3,
                seconds * 1e3 / *count as f64
            );
        }
    }
}

impl Default for NativeCpuExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeCpuExecutor {
    /// An executor configured from the environment controls.
    pub fn new() -> Self {
        let threads = std::env::var(THREADS_ENV)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&threads| threads > 0)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|count| count.get())
                    .unwrap_or(1)
            });
        let mut executor = Self::with_settings(
            std::env::var(PACKED_WEIGHTS_ENV).as_deref() != Ok("0"),
            std::env::var(INT8_ACTIVATIONS_ENV).as_deref() != Ok("0"),
            threads,
        );
        executor.compact_weights = std::env::var(COMPACT_WEIGHTS_ENV).as_deref() != Ok("0");
        executor.step_capture = std::env::var(STEP_CAPTURE_ENV).as_deref() != Ok("0");
        executor
    }

    /// An executor with explicit controls (tests, embedders).
    pub fn with_settings(packed_weights: bool, int8_activations: bool, threads: usize) -> Self {
        let state = Arc::new(NativeState {
            packed: Mutex::new(BTreeMap::new()),
            compact: Mutex::new(BTreeMap::new()),
            pool: pool::Pool::new(threads),
            int8_activations,
            materialized: Mutex::new((0, 0.0)),
        });
        let inner = ReferenceCpuExecutor {
            provider_name: NATIVE_CPU_PROVIDER_NAME,
            packed: Some(state.clone()),
            ..ReferenceCpuExecutor::default()
        };
        Self {
            inner: Arc::new(inner),
            state,
            packed_weights,
            compact_weights: true,
            step_capture: true,
            capture: Mutex::new(capture::CaptureState::default()),
            profile: (std::env::var(PROFILE_ENV).as_deref() == Ok("1"))
                .then(|| Mutex::new(BTreeMap::new())),
        }
    }

    /// The wrapped executor (Reference CPU kernels, storage, observations).
    pub fn reference(&self) -> &ReferenceCpuExecutor {
        &self.inner
    }

    pub fn threads(&self) -> usize {
        self.state.pool.threads()
    }

    /// Fails the step capture in progress, if any: `what` writes host data
    /// a replay would not write again.
    fn poison_capture(&self, what: &str) {
        self.capture.lock().unwrap().poison(what);
    }

    /// Adds `seconds` of Kernel `name` to the profile, when profiling.
    fn profile(&self, name: &str, started: Option<std::time::Instant>) {
        if let (Some(profile), Some(started)) = (&self.profile, started) {
            let mut profile = profile.lock().unwrap();
            let entry = profile.entry(name.to_string()).or_default();
            entry.0 += 1;
            entry.1 += started.elapsed().as_secs_f64();
        }
    }

    /// Whether `id` is held packed, or compact.
    pub fn is_packed(&self, id: &TensorResourceId) -> bool {
        self.state.weight(id).is_some() || self.is_compact(id)
    }

    /// Whether `id` is held as compact bfloat16 bits.
    pub fn is_compact(&self, id: &TensorResourceId) -> bool {
        self.state.compact(id).is_some()
    }

    /// Forgets a packed or compact weight `id`, which is being replaced.
    fn forget_packed(&self, id: &TensorResourceId) -> bool {
        let packed = self.state.packed.lock().unwrap().remove(id).is_some();
        let compact = self.state.compact.lock().unwrap().remove(id).is_some();
        packed || compact
    }
}

impl ProviderExecutionApi for NativeCpuExecutor {
    fn submit(
        &self,
        request: ProviderExecutionRequest,
    ) -> Result<ProviderExecutionHandle, ProviderExecutionError> {
        ProviderExecutionApi::submit(&*self.inner, request)
    }

    fn status(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionStatus, ProviderExecutionError> {
        ProviderExecutionApi::status(&*self.inner, handle)
    }

    fn cancel(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderCancellationOutcome, ProviderExecutionError> {
        ProviderExecutionApi::cancel(&*self.inner, handle)
    }

    fn complete(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionResult, ProviderExecutionError> {
        ProviderExecutionApi::complete(&*self.inner, handle)
    }

    fn release(&self, handle: ProviderExecutionHandle) -> Result<(), ProviderExecutionError> {
        ProviderExecutionApi::release(&*self.inner, handle)
    }

    fn submit_kernel(
        &self,
        advertisement: &KernelAdvertisement,
        operator: &OperatorSpec,
        invocation: &KernelInvocation,
        memory: &mut MemoryManager,
    ) -> Result<ProviderExecutionHandle, ProviderExecutionError> {
        let started = self.profile.as_ref().map(|_| std::time::Instant::now());
        let handle = ProviderExecutionApi::submit_kernel(
            &*self.inner,
            advertisement,
            operator,
            invocation,
            memory,
        );
        self.profile(&invocation.kernel.name, started);
        let mut state = self.capture.lock().unwrap();
        if let Some(recording) = &mut state.recording {
            match &handle {
                Ok(_) => recording.invocations.push(invocation.clone()),
                Err(_) => {
                    recording
                        .failure
                        .get_or_insert_with(|| "a submission failed".into());
                }
            }
        }
        drop(state);
        handle
    }

    fn complete_kernel(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<KernelResult, ProviderExecutionError> {
        ProviderExecutionApi::complete_kernel(&*self.inner, handle)
    }

    fn write_tensor(
        &self,
        id: TensorResourceId,
        tensor: HostTensor,
    ) -> Result<(), ProviderExecutionError> {
        self.poison_capture("a host tensor write");
        self.forget_packed(&id);
        ProviderExecutionApi::write_tensor(&*self.inner, id, tensor)
    }

    /// A weight whose every value is exactly a bfloat16 value is stored
    /// as those values' bits (`speed-up-dense-cpu-matmuls`); any other one
    /// as f32.
    fn write_weight_tensor(
        &self,
        id: TensorResourceId,
        tensor: HostTensor,
    ) -> Result<(), ProviderExecutionError> {
        self.poison_capture("a weight write");
        let compact = self
            .compact_weights
            .then(|| CompactWeight::of(&tensor))
            .flatten();
        let Some(compact) = compact else {
            return self.write_tensor(id, tensor);
        };
        self.forget_packed(&id);
        self.inner.storage.lock().unwrap().remove(&id);
        self.state
            .compact
            .lock()
            .unwrap()
            .insert(id, Arc::new(compact));
        Ok(())
    }

    fn write_packed_weight_tensor(
        &self,
        id: TensorResourceId,
        weight: PackedWeight,
    ) -> Result<PackedWeightWrite, ProviderExecutionError> {
        self.poison_capture("a packed weight write");
        if !self.packed_weights {
            return Ok(PackedWeightWrite::Declined);
        }
        self.inner.storage.lock().unwrap().remove(&id);
        self.state
            .packed
            .lock()
            .unwrap()
            .insert(id, Arc::new(weight));
        Ok(PackedWeightWrite::Stored)
    }

    fn read_tensor(&self, id: &TensorResourceId) -> Option<HostTensor> {
        ProviderExecutionApi::read_tensor(&*self.inner, id)
    }

    fn release_tensor(&self, id: &TensorResourceId) -> Result<bool, ProviderExecutionError> {
        let packed = self.forget_packed(id);
        Ok(ProviderExecutionApi::release_tensor(&*self.inner, id)? || packed)
    }

    fn release_admitted_tensor(
        &self,
        memory: &mut MemoryManager,
        id: &TensorResourceId,
    ) -> Result<bool, ProviderExecutionError> {
        let packed = self.forget_packed(id);
        Ok(ProviderExecutionApi::release_admitted_tensor(&*self.inner, memory, id)? || packed)
    }

    fn write_tensor_admitted(
        &self,
        memory: &mut MemoryManager,
        resource_id: TensorResourceId,
        tensor: HostTensor,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<(), MemoryError> {
        self.poison_capture("a host tensor write");
        // A packed weight is replaced only once its replacement is admitted
        // and stored: a denied admission leaves it resident.
        ProviderExecutionApi::write_tensor_admitted(
            &*self.inner,
            memory,
            resource_id.clone(),
            tensor,
            class,
            owner,
        )?;
        self.forget_packed(&resource_id);
        Ok(())
    }

    /// Like the CUDA Provider, never copies a resource out to answer this
    /// (the Runtime reads every node's inputs through it): resident
    /// resources are `Opaque`, and callers that need host values read them
    /// through [`Self::read_tensor`].
    fn read_tensor_value(&self, id: &TensorResourceId) -> Option<TensorValue> {
        (self.is_packed(id) || self.inner.storage.lock().unwrap().contains_key(id))
            .then_some(TensorValue::Opaque)
    }

    fn resident_shape(&self, id: &TensorResourceId) -> Option<Vec<u64>> {
        ProviderExecutionApi::resident_shape(&*self.inner, id)
    }

    fn write_tensor_value(
        &self,
        id: TensorResourceId,
        value: TensorValue,
    ) -> Result<(), ProviderExecutionError> {
        // Writing back an `Opaque` value (what `read_tensor_value` returns)
        // stores nothing, so a packed weight stays resident.
        let replaces = !matches!(value, TensorValue::Opaque);
        if replaces {
            self.poison_capture("a host tensor write");
        }
        ProviderExecutionApi::write_tensor_value(&*self.inner, id.clone(), value)?;
        if replaces {
            self.forget_packed(&id);
        }
        Ok(())
    }

    fn write_tensor_value_admitted(
        &self,
        memory: &mut MemoryManager,
        resource_id: TensorResourceId,
        value: TensorValue,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<(), TensorValueAdmissionError> {
        let replaces = !matches!(value, TensorValue::Opaque);
        if replaces {
            self.poison_capture("a host tensor write");
        }
        ProviderExecutionApi::write_tensor_value_admitted(
            &*self.inner,
            memory,
            resource_id.clone(),
            value,
            class,
            owner,
        )?;
        if replaces {
            self.forget_packed(&resource_id);
        }
        Ok(())
    }

    fn copy_tensor_admitted(
        &self,
        memory: &mut MemoryManager,
        from: &TensorResourceId,
        to: TensorResourceId,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<MemoryAllocationId, TensorValueAdmissionError> {
        self.poison_capture("a tensor copy");
        let allocation = ProviderExecutionApi::copy_tensor_admitted(
            &*self.inner,
            memory,
            from,
            to.clone(),
            class,
            owner,
        )?;
        // The copy replaces whatever `to` held, a packed weight included.
        self.forget_packed(&to);
        Ok(allocation)
    }

    fn write_zeroed_kv_store(
        &self,
        id: &TensorResourceId,
        shape: &[u64],
        precision: KvStorePrecision,
        head_dimension: u64,
    ) -> Result<bool, ProviderExecutionError> {
        self.poison_capture("a KV store reset");
        // The store replaces whatever `id` held, a packed weight included.
        self.forget_packed(id);
        Ok(self
            .inner
            .write_zeroed_kv_store(id, shape, precision, head_dimension))
    }

    fn write_zeroed_kv_store_with_exact_head(
        &self,
        id: &TensorResourceId,
        shape: &[u64],
        precision: KvStorePrecision,
        head_dimension: u64,
        exact_rows: u64,
    ) -> Result<bool, ProviderExecutionError> {
        self.poison_capture("a KV store reset");
        // The store replaces whatever `id` held, a packed weight included.
        self.forget_packed(id);
        Ok(self.inner.write_zeroed_kv_store_with_exact_head(
            id,
            shape,
            precision,
            head_dimension,
            exact_rows,
        ))
    }

    fn allocate_workspace(
        &self,
        memory: &mut MemoryManager,
        size_bytes: u64,
    ) -> Result<MemoryAllocationId, MemoryError> {
        ProviderExecutionApi::allocate_workspace(&*self.inner, memory, size_bytes)
    }

    fn observations(&self) -> Vec<KernelObservation> {
        ProviderExecutionApi::observations(&*self.inner)
    }

    fn supports_step_capture(&self) -> bool {
        self.step_capture && !cfg!(target_arch = "wasm32")
    }

    fn begin_step_capture(
        &self,
        request: &magnetar_provider_api::StepCaptureRequest,
    ) -> Result<(), ProviderExecutionError> {
        let mut state = self.capture.lock().unwrap();
        if !self.supports_step_capture() || state.recording.is_some() {
            return Err(ProviderExecutionError::new(
                ProviderExecutionErrorCode::UnsupportedOperation,
                ProviderExecutionPhase::Submit,
                ProviderBinding::new(NATIVE_CPU_PROVIDER_NAME),
                None,
                "step capture is off or already in progress",
            ));
        }
        state.recording = Some(capture::Recording {
            advancing: request.advancing_attributes.clone(),
            ..capture::Recording::default()
        });
        Ok(())
    }

    fn end_step_capture(
        &self,
    ) -> Result<magnetar_provider_api::StepCaptureId, ProviderExecutionError> {
        let mut state = self.capture.lock().unwrap();
        let failed = |reason: String| {
            ProviderExecutionError::new(
                ProviderExecutionErrorCode::UnsupportedOperation,
                ProviderExecutionPhase::Complete,
                ProviderBinding::new(NATIVE_CPU_PROVIDER_NAME),
                None,
                reason,
            )
        };
        let recording = state
            .recording
            .take()
            .ok_or_else(|| failed("no step capture in progress".into()))?;
        if let Some(reason) = recording.failure {
            return Err(failed(reason));
        }
        let id = magnetar_provider_api::StepCaptureId::new(format!("cpu-step:{}", state.next));
        state.next += 1;
        state.captures.insert(id.clone(), Arc::new(recording));
        Ok(id)
    }

    fn abort_step_capture(&self) {
        self.capture.lock().unwrap().recording = None;
    }

    fn replay_step(
        &self,
        id: &magnetar_provider_api::StepCaptureId,
        offset: u64,
        inputs: Vec<(TensorResourceId, HostTensor)>,
    ) -> Result<(), magnetar_provider_api::StepReplayError> {
        let recording = self
            .capture
            .lock()
            .unwrap()
            .captures
            .get(id)
            .cloned()
            .ok_or_else(|| magnetar_provider_api::StepReplayError::Invalidated {
                reason: format!("no capture {}", id.as_str()),
            })?;
        // Every attribute advanced first: an offset that does not fit runs
        // nothing.
        let invocations = recording
            .invocations
            .iter()
            .map(|invocation| capture::advanced(invocation, &recording.advancing, offset))
            .collect::<Result<Vec<_>, _>>()?;
        for (id, tensor) in inputs {
            self.forget_packed(&id);
            self.inner.write_tensor(id, tensor);
        }
        for invocation in &invocations {
            let started = self.profile.as_ref().map(|_| std::time::Instant::now());
            let ran = self.inner.run_invocation(invocation);
            self.profile(&invocation.kernel.name, started);
            if let Err(error) = ran {
                return Err(capture::replay_failed(format!(
                    "replaying {}: {error:?}",
                    invocation.kernel.name
                )));
            }
        }
        Ok(())
    }

    fn release_step_capture(&self, id: &magnetar_provider_api::StepCaptureId) {
        self.capture.lock().unwrap().captures.remove(id);
    }

    /// The highest-scoring column of `logits`' row `row` outside
    /// `excluded`, each score divided by `temperature` unless it is 0,
    /// read in place.
    fn select_greedy_column(
        &self,
        logits: &TensorResourceId,
        row: u64,
        excluded: &[std::ops::Range<u64>],
        temperature: f32,
    ) -> Option<u64> {
        let storage = self.inner.storage.lock().unwrap();
        let tensor = storage.get(logits)?;
        let columns = *tensor.shape.last()? as usize;
        let start = usize::try_from(row).ok()?.checked_mul(columns)?;
        let scores = tensor.data.get(start..start.checked_add(columns)?)?;
        let mut best: Option<(f32, u64)> = None;
        for (column, &score) in scores.iter().enumerate() {
            let column = column as u64;
            if !score.is_finite() || excluded.iter().any(|range| range.contains(&column)) {
                continue;
            }
            let score = if temperature == 0.0 {
                score
            } else {
                score / temperature
            };
            if best.is_none_or(|(value, _)| score > value) {
                best = Some((score, column));
            }
        }
        best.map(|(_, column)| column)
    }
}

pub fn native_cpu_provider_metadata() -> ProviderMetadata {
    ProviderMetadata::new(
        NATIVE_CPU_PROVIDER_NAME,
        NATIVE_CPU_PROVIDER_VERSION,
        REFERENCE_CPU_PROVIDER_VENDOR,
        "Optimized host CPU implementation: packed quantized weights, int8 activations, \
         vector kernels and worker threads",
    )
}

pub fn native_cpu_device() -> DeviceDescriptor {
    let reference = reference_cpu_device();
    let mut metadata = DeviceMetadata::new(
        DeviceId::new(NATIVE_CPU_DEVICE_ID),
        "Optimized CPU",
        DeviceType::Cpu,
        NATIVE_CPU_PROVIDER_NAME,
    );
    let base = reference.metadata();
    metadata.vendor = base.vendor.clone();
    metadata.architecture = base.architecture.clone();
    metadata.compute_units = base.compute_units;
    metadata.execution_capabilities = base.execution_capabilities.clone();
    metadata.dtype_support = base.dtype_support.clone();
    metadata.layout_support = base.layout_support.clone();
    metadata.memory_class_support = base.memory_class_support.clone();
    metadata.execution_limits = base.execution_limits;
    metadata.pressure = ProviderPressureLevel::Low;
    DeviceDescriptor::new(metadata)
}

/// The Reference CPU kernel set under this Provider's identity: packed
/// weights reach the matmul and embedding kernels through their resource
/// ids, with no change to the operators advertised.
pub fn native_cpu_kernel_advertisements() -> Vec<KernelAdvertisement> {
    let family = if simd_level() > SimdLevel::Scalar {
        KernelImplementationFamily::CpuVector
    } else {
        KernelImplementationFamily::CpuScalar
    };
    reference_cpu_kernel_advertisements()
        .into_iter()
        .map(|mut advertisement| {
            advertisement.id.provider = ProviderBinding::new(NATIVE_CPU_PROVIDER_NAME);
            advertisement.id.family = family;
            advertisement.devices = [DeviceBinding::new(DeviceId::new(NATIVE_CPU_DEVICE_ID))]
                .into_iter()
                .collect();
            advertisement
        })
        .collect()
}

/// The optimized CPU Provider.
pub struct NativeCpuProvider {
    device: Arc<DeviceDescriptor>,
    executor: Arc<NativeCpuExecutor>,
    pressure: Mutex<ProviderPressureLevel>,
}

impl Default for NativeCpuProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeCpuProvider {
    pub fn new() -> Self {
        Self::with_executor(NativeCpuExecutor::new())
    }

    pub fn with_executor(executor: NativeCpuExecutor) -> Self {
        Self {
            device: Arc::new(native_cpu_device()),
            executor: Arc::new(executor),
            pressure: Mutex::new(ProviderPressureLevel::Low),
        }
    }

    pub fn executor(&self) -> Arc<NativeCpuExecutor> {
        self.executor.clone()
    }

    pub fn report_pressure(&self, level: ProviderPressureLevel) {
        *self.pressure.lock().unwrap() = level;
    }
}

impl Provider for NativeCpuProvider {
    fn metadata(&self) -> ProviderMetadata {
        native_cpu_provider_metadata()
    }

    fn register(&self, _registry: &mut ProviderRegistry) -> Result<(), ProviderError> {
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
        native_cpu_kernel_advertisements()
    }

    fn initialize(&self) -> Result<(), ProviderError> {
        self.executor.inner.observe(
            KernelObservation::new(KernelObservationKind::ProviderRegistered)
                .with_redacted_metadata("provider", NATIVE_CPU_PROVIDER_NAME),
        );
        self.executor.inner.observe(
            KernelObservation::new(KernelObservationKind::DeviceDetected)
                .with_redacted_metadata("device", NATIVE_CPU_DEVICE_ID),
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
