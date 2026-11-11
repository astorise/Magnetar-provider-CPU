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
//! - [`SIMD_ENV`] `=scalar` disables the vector kernels.

mod pool;
pub mod quant;

use super::*;
use magnetar_runtime::packed_weight::{PackedWeight, PackedWeightWrite};
use std::sync::OnceLock;

/// Stable, package-qualified optimized CPU Provider identity.
pub use magnetar_runtime::reference_cpu::NATIVE_CPU_PROVIDER_NAME;
pub const NATIVE_CPU_DEVICE_ID: &str = "cpu-native:host:0";
pub const NATIVE_CPU_PROVIDER_VERSION: &str = "0.1.0";
pub const PACKED_WEIGHTS_ENV: &str = "MAGNETAR_CPU_PACKED_WEIGHTS";
pub const INT8_ACTIVATIONS_ENV: &str = "MAGNETAR_CPU_INT8_ACTIVATIONS";
pub const THREADS_ENV: &str = "MAGNETAR_CPU_THREADS";
pub const SIMD_ENV: &str = "MAGNETAR_CPU_SIMD";
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
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
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

struct NativeState {
    packed: Mutex<BTreeMap<TensorResourceId, Arc<PackedWeight>>>,
    pool: pool::Pool,
    int8_activations: bool,
    /// Packed weights expanded to f32 for a caller or Kernel without a
    /// packed path: count and seconds (reported with [`PROFILE_ENV`]).
    materialized: Mutex<(u64, f64)>,
}

/// `*mut f32` shared across pool tasks that write disjoint elements.
struct SharedOut(*mut f32);
// SAFETY: every task writes a disjoint set of output elements.
unsafe impl Sync for SharedOut {}

impl NativeState {
    fn weight(&self, id: &TensorResourceId) -> Option<Arc<PackedWeight>> {
        self.packed.lock().unwrap().get(id).cloned()
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
        match invocation.kernel.name.as_str() {
            "matmul-bias" => {
                add(&product, &executor.input_tensor(invocation, 2)?).map_err(KernelError::from)
            }
            "matmul-residual" => residual_add(&executor.input_tensor(invocation, 2)?, &product)
                .map_err(KernelError::from),
            _ => Ok(product),
        }
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
                let weight = self.weight(&invocation.inputs.get(1)?.resource.id)?;
                Some(self.matmul(executor, invocation, &weight))
            }
            "embedding" => {
                let table = self.weight(&invocation.inputs.first()?.resource.id)?;
                Some(self.embedding(executor, invocation, &table))
            }
            _ => None,
        }
    }

    fn materialize(&self, id: &TensorResourceId) -> Option<HostTensor> {
        self.weight(id).map(|weight| self.expand(&weight))
    }

    fn shape(&self, id: &TensorResourceId) -> Option<Vec<u64>> {
        self.weight(id)
            .map(|weight| weight.logical_shape().to_vec())
    }
}

/// The optimized CPU Provider's executor: a [`ReferenceCpuExecutor`]
/// reporting this Provider's identity, with packed weights and their
/// kernels.
pub struct NativeCpuExecutor {
    inner: Arc<ReferenceCpuExecutor>,
    state: Arc<NativeState>,
    packed_weights: bool,
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
        Self::with_settings(
            std::env::var(PACKED_WEIGHTS_ENV).as_deref() != Ok("0"),
            std::env::var(INT8_ACTIVATIONS_ENV).as_deref() != Ok("0"),
            threads,
        )
    }

    /// An executor with explicit controls (tests, embedders).
    pub fn with_settings(packed_weights: bool, int8_activations: bool, threads: usize) -> Self {
        let state = Arc::new(NativeState {
            packed: Mutex::new(BTreeMap::new()),
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

    /// Whether `id` is held packed.
    pub fn is_packed(&self, id: &TensorResourceId) -> bool {
        self.state.weight(id).is_some()
    }

    fn forget_packed(&self, id: &TensorResourceId) -> bool {
        self.state.packed.lock().unwrap().remove(id).is_some()
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
        if let (Some(profile), Some(started)) = (&self.profile, started) {
            let mut profile = profile.lock().unwrap();
            let entry = profile.entry(invocation.kernel.name.clone()).or_default();
            entry.0 += 1;
            entry.1 += started.elapsed().as_secs_f64();
        }
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
        self.forget_packed(&id);
        ProviderExecutionApi::write_tensor(&*self.inner, id, tensor)
    }

    fn write_packed_weight_tensor(
        &self,
        id: TensorResourceId,
        weight: PackedWeight,
    ) -> Result<PackedWeightWrite, ProviderExecutionError> {
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
        self.forget_packed(&resource_id);
        ProviderExecutionApi::write_tensor_admitted(
            &*self.inner,
            memory,
            resource_id,
            tensor,
            class,
            owner,
        )
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
        self.forget_packed(&id);
        ProviderExecutionApi::write_tensor_value(&*self.inner, id, value)
    }

    fn write_tensor_value_admitted(
        &self,
        memory: &mut MemoryManager,
        resource_id: TensorResourceId,
        value: TensorValue,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<(), TensorValueAdmissionError> {
        self.forget_packed(&resource_id);
        ProviderExecutionApi::write_tensor_value_admitted(
            &*self.inner,
            memory,
            resource_id,
            value,
            class,
            owner,
        )
    }

    fn copy_tensor_admitted(
        &self,
        memory: &mut MemoryManager,
        from: &TensorResourceId,
        to: TensorResourceId,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<MemoryAllocationId, TensorValueAdmissionError> {
        ProviderExecutionApi::copy_tensor_admitted(&*self.inner, memory, from, to, class, owner)
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
