//! Tests of the optimized CPU Provider against the Reference CPU kernels.

use super::*;
use magnetar_provider_api::packed_weight::PackedWeightFormat;

const FORMATS: [PackedWeightFormat; 14] = [
    PackedWeightFormat::Q8_0,
    PackedWeightFormat::Q4K,
    PackedWeightFormat::Q6K,
    PackedWeightFormat::Q2K,
    PackedWeightFormat::Q3K,
    PackedWeightFormat::Iq4Nl,
    PackedWeightFormat::Iq4Xs,
    PackedWeightFormat::Iq2Xxs,
    PackedWeightFormat::Iq2Xs,
    PackedWeightFormat::Iq2S,
    PackedWeightFormat::Iq3Xxs,
    PackedWeightFormat::Iq3S,
    PackedWeightFormat::Q5K,
    PackedWeightFormat::Q4_0,
];

/// Byte offsets of each block's `f16` scale fields.
fn half_offsets(format: PackedWeightFormat) -> &'static [usize] {
    match format {
        PackedWeightFormat::Q4K | PackedWeightFormat::Q5K => &[0, 2],
        PackedWeightFormat::Q6K => &[208],
        PackedWeightFormat::Q2K => &[80, 82],
        PackedWeightFormat::Q3K => &[108],
        _ => &[0],
    }
}

struct Rng(u32);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
}

/// A random packed matrix: random bytes, small positive f16 scales.
fn packed(
    format: PackedWeightFormat,
    rows: u64,
    cols: u64,
    transposed: bool,
    seed: u32,
) -> PackedWeight {
    let mut rng = Rng(seed.wrapping_mul(2_654_435_761).wrapping_add(1));
    let block_bytes = format.block_bytes() as usize;
    let blocks = (rows * cols / format.block_elements()) as usize;
    let mut bytes = Vec::with_capacity(blocks * block_bytes);
    for _ in 0..blocks {
        let mut block: Vec<u8> = (0..block_bytes).map(|_| (rng.next() % 256) as u8).collect();
        for &at in half_offsets(format) {
            let exponent = 3 + (rng.next() % 5) as u16;
            let bits = (exponent << 10) | (rng.next() % 1024) as u16;
            block[at..at + 2].copy_from_slice(&bits.to_le_bytes());
        }
        bytes.extend(block);
    }
    PackedWeight::new(format, rows, cols, transposed, bytes).unwrap()
}

fn activations(rows: u64, cols: u64, seed: u32) -> HostTensor {
    let mut rng = Rng(seed.wrapping_add(7));
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| (rng.next() % 2001) as f32 / 1000.0 - 1.0)
        .collect();
    HostTensor::new([rows, cols], data).unwrap()
}

/// The activations as int8 activations see them.
fn int8_rounded(a: &HostTensor) -> HostTensor {
    let quantized = quant::quantize_rows(&a.data);
    let data: Vec<f32> = quantized
        .values
        .as_chunks::<{ quant::RUN }>()
        .0
        .iter()
        .zip(&quantized.scales)
        .flat_map(|(block, &d)| block.iter().map(move |&q| f32::from(q) * d))
        .collect();
    HostTensor::new(a.shape.clone(), data).unwrap()
}

fn assert_close(actual: &HostTensor, expected: &HostTensor, what: &str) {
    assert_eq!(actual.shape, expected.shape, "{what}");
    let scale = expected
        .data
        .iter()
        .fold(1.0_f32, |max, e| max.max(e.abs()));
    for (index, (a, e)) in actual.data.iter().zip(&expected.data).enumerate() {
        assert!(
            (a - e).abs() <= 2e-5 * scale,
            "{what}: index {index}: {a} vs {e} (scale {scale})"
        );
    }
}

fn state(int8: bool, threads: usize) -> NativeState {
    NativeState {
        packed: Mutex::new(BTreeMap::new()),
        compact: Mutex::new(BTreeMap::new()),
        padding_rows: std::sync::atomic::AtomicUsize::new(0),
        pool: pool::Pool::new(threads),
        decode_pool: None,
        int8_activations: int8,
        materialized: Mutex::new((0, 0.0)),
    }
}

/// Every format's runs decode to the Runtime's dequantization.
#[test]
fn runs_decode_like_the_runtime() {
    for format in FORMATS {
        let weight = packed(format, 3, 512, true, 11);
        let expected = weight.dequantize_stored();
        let reader = quant::RunReader::new(&weight);
        let mut row = vec![0.0_f32; 512];
        for r in 0..3 {
            quant::dequantize_row(&reader, r, 512, &mut row);
            for (c, (&a, &e)) in row.iter().zip(&expected[r * 512..]).enumerate() {
                assert!(
                    (a - e).abs() <= 1e-6 * e.abs().max(1e-3),
                    "{format:?} row {r} col {c}: {a} vs {e}"
                );
            }
        }
    }
}

/// Packed matmuls match the Reference CPU on the int8-rounded activations,
/// and on the f32 activations without int8 activations, for every format,
/// one and several rows, one and several threads.
#[test]
fn packed_matmuls_match_the_reference_cpu() {
    let (k, n) = (512_u64, 37_u64);
    for format in FORMATS {
        let weight = packed(format, n, k, true, 21);
        let dense = weight.dequantize();
        for (int8, threads) in [(true, 1), (true, 4), (false, 3)] {
            let state = state(int8, threads);
            for m in [1_u64, 3, 9] {
                let a = activations(m, k, m as u32);
                let seen = if int8 { int8_rounded(&a) } else { a.clone() };
                let expected = matmul(&seen, &dense, false, false).unwrap();
                let actual = state.packed_matmul(&a, &weight).unwrap();
                assert_close(&actual, &expected, &format!("{format:?} int8 {int8} m {m}"));
            }
        }
    }
}

/// 32-value block formats whose row holds an odd number of runs take the
/// paired kernels' single-run tail.
#[test]
fn odd_run_counts_match_the_reference_cpu() {
    let (k, n) = (96_u64, 5_u64);
    for format in [
        PackedWeightFormat::Q8_0,
        PackedWeightFormat::Iq4Nl,
        PackedWeightFormat::Q4_0,
    ] {
        let weight = packed(format, n, k, true, 23);
        let dense = weight.dequantize();
        let state = state(true, 1);
        let a = activations(2, k, 7);
        let expected = matmul(&int8_rounded(&a), &dense, false, false).unwrap();
        let actual = state.packed_matmul(&a, &weight).unwrap();
        assert_close(&actual, &expected, &format!("{format:?} odd runs"));
    }
}

/// A packed untransposed table serves embedding lookups, and its tied use
/// as an output head (`transpose_b`) reads it packed.
#[test]
fn packed_tables_serve_lookups_and_tied_heads() {
    let table = packed(PackedWeightFormat::Q8_0, 50, 64, false, 31);
    let dense = table.dequantize();
    let reader = quant::RunReader::new(&table);
    let mut row = vec![0.0_f32; 64];
    quant::dequantize_row(&reader, 7, 64, &mut row);
    let expected = embedding_lookup(&dense, &HostTensor::new([1], vec![7.0]).unwrap()).unwrap();
    assert_close(&HostTensor::new([1, 64], row).unwrap(), &expected, "row 7");
    let state = state(false, 2);
    let a = activations(2, 64, 3);
    let expected = matmul(&a, &dense, false, true).unwrap();
    let actual = state.packed_matmul(&a, &table).unwrap();
    assert_close(&actual, &expected, "tied head");
}

/// A weight whose stored rows are its input rows (an untransposed `[k,
/// n]` weight, or a tied table stored transposed) multiplies through the
/// axpy path, for every format.
#[test]
fn input_major_weights_multiply_packed() {
    for format in FORMATS {
        let weight = packed(format, 64, 256, false, 61);
        let dense = weight.dequantize();
        let state = state(false, 3);
        for m in [1_u64, 4] {
            let a = activations(m, 64, 62);
            let expected = matmul(&a, &dense, false, false).unwrap();
            let actual = state.packed_axpy(&a, &weight).unwrap();
            assert_close(&actual, &expected, &format!("{format:?} axpy m {m}"));
        }
    }
}

/// The AVX2 half-dot products equal the scalar ones, including -128.
#[test]
fn vector_dots_equal_scalar_dots() {
    let mut rng = Rng(5);
    for _ in 0..1000 {
        let mut q = [0_i8; quant::RUN];
        let mut a = [0_i8; quant::RUN];
        for i in 0..quant::RUN {
            q[i] = (rng.next() % 256) as u8 as i8;
            a[i] = (rng.next() % 256) as u8 as i8;
        }
        q[0] = -128;
        let scalar = quant::dot_halves_scalar(&q, &a);
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected.
            assert_eq!(unsafe { quant::dot_halves_avx2(&q, &a) }, scalar);
        }
        let _ = scalar;
    }
}

/// The pool runs every task once and returns after the last.
#[test]
fn pool_runs_every_task_once() {
    let pool = pool::Pool::new(4);
    for tasks in [0, 1, 3, 100] {
        let counts: Vec<AtomicU64> = (0..tasks).map(|_| AtomicU64::new(0)).collect();
        pool.run(tasks, &|i| {
            counts[i].fetch_add(1, Ordering::Relaxed);
        });
        assert!(
            counts
                .iter()
                .all(|count| count.load(Ordering::Relaxed) == 1)
        );
    }
}

/// The executor stores packed weights (or declines them under the
/// control), reads them back dequantized and releases them.
#[test]
fn executor_stores_packed_weights() {
    let id = TensorResourceId::new("w");
    let weight = packed(PackedWeightFormat::Q4K, 4, 256, true, 41);
    let executor = NativeCpuExecutor::with_settings(true, true, 2);
    assert_eq!(
        executor
            .write_packed_weight_tensor(id.clone(), weight.clone())
            .unwrap(),
        PackedWeightWrite::Stored
    );
    assert!(executor.is_packed(&id));
    assert_eq!(executor.read_tensor(&id).unwrap(), weight.dequantize());
    assert_eq!(executor.resident_shape(&id).unwrap(), vec![256, 4]);
    assert!(executor.release_tensor(&id).unwrap());
    assert!(!executor.is_packed(&id));
    let declining = NativeCpuExecutor::with_settings(false, true, 1);
    assert_eq!(
        declining.write_packed_weight_tensor(id, weight).unwrap(),
        PackedWeightWrite::Declined
    );
}

/// Writing back the `Opaque` value a packed weight reads as keeps it, and
/// a copy onto a packed weight's identity replaces it for execution too.
#[test]
fn packed_weights_survive_opaque_writes_and_yield_to_copies() {
    let id = TensorResourceId::new("w");
    let weight = packed(PackedWeightFormat::Q8_0, 2, 32, true, 43);
    let executor = NativeCpuExecutor::with_settings(true, true, 1);
    executor
        .write_packed_weight_tensor(id.clone(), weight)
        .unwrap();
    let value = executor.read_tensor_value(&id).unwrap();
    executor.write_tensor_value(id.clone(), value).unwrap();
    assert!(executor.is_packed(&id));
    let mut memory = MemoryManager::default();
    let owner = || MemoryAllocationOwner::Session("test-session".into());
    let source = TensorResourceId::new("source");
    let replacement = HostTensor::new(vec![32, 2], vec![0.5; 64]).unwrap();
    executor
        .write_tensor_admitted(
            &mut memory,
            source.clone(),
            replacement.clone(),
            MemoryAllocationClass::Tensor,
            owner(),
        )
        .unwrap();
    executor
        .copy_tensor_admitted(
            &mut memory,
            &source,
            id.clone(),
            MemoryAllocationClass::Tensor,
            owner(),
        )
        .unwrap();
    assert!(!executor.is_packed(&id));
    assert_eq!(executor.read_tensor(&id).unwrap(), replacement);
}

/// The Provider has its own identity, Device and advertisements.
#[test]
fn provider_identity_is_distinct() {
    let provider =
        NativeCpuProvider::with_executor(NativeCpuExecutor::with_settings(true, true, 1));
    assert_eq!(provider.metadata().name, NATIVE_CPU_PROVIDER_NAME);
    assert_ne!(provider.metadata().name, REFERENCE_CPU_PROVIDER_NAME);
    let advertisements = provider.kernel_advertisements();
    assert!(
        advertisements
            .iter()
            .any(|advertisement| advertisement.id.name == "matmul")
    );
    assert!(advertisements.iter().all(|advertisement| {
        advertisement.id.provider == ProviderBinding::new(NATIVE_CPU_PROVIDER_NAME)
    }));
}

/// Decode-step bandwidth on Mistral 7B shapes per format (diagnostic: run
/// with `--ignored --nocapture`).
#[test]
#[ignore = "timing diagnostic"]
fn packed_matmul_bandwidth() {
    let threads = std::thread::available_parallelism().map_or(1, |count| count.get());
    let state = state(true, threads);
    println!("{threads} threads, {:?}", simd_level());
    for format in [
        PackedWeightFormat::Q8_0,
        PackedWeightFormat::Q4K,
        PackedWeightFormat::Q6K,
        PackedWeightFormat::Iq4Xs,
        PackedWeightFormat::Q2K,
        PackedWeightFormat::Q3K,
        PackedWeightFormat::Q5K,
        PackedWeightFormat::Iq4Nl,
        PackedWeightFormat::Q4_0,
        PackedWeightFormat::Iq2Xxs,
        PackedWeightFormat::Iq2Xs,
        PackedWeightFormat::Iq2S,
        PackedWeightFormat::Iq3Xxs,
        PackedWeightFormat::Iq3S,
    ] {
        for (k, n) in [(4096_u64, 14_336_u64), (14_336, 4096)] {
            let weight = packed(format, n, k, true, 51);
            let a = activations(1, k, 52);
            state.packed_matmul(&a, &weight).unwrap();
            let repeats = 10;
            let start = std::time::Instant::now();
            for _ in 0..repeats {
                state.packed_matmul(&a, &weight).unwrap();
            }
            let seconds = start.elapsed().as_secs_f64() / f64::from(repeats);
            let bytes = weight.bytes().len() as f64;
            println!(
                "{format:?} [1, {k}] x [{k}, {n}]: {:.2} ms ({:.1} GB/s over {:.1} MB)",
                seconds * 1e3,
                bytes / seconds / 1e9,
                bytes / 1e6
            );
        }
    }
}

/// An f32 embedding table (a k-quant table expanded at load) is gathered
/// in place, equal to the Reference CPU lookup.
#[test]
fn f32_tables_are_gathered_in_place() {
    let executor = NativeCpuExecutor::with_settings(true, true, 1);
    let table = activations(10, 64, 71);
    let id = TensorResourceId::new("table");
    executor.reference().write_tensor(id.clone(), table.clone());
    let ids = HostTensor::new([3], vec![2.0, 9.0, 0.0]).unwrap();
    let ids_id = TensorResourceId::new("ids");
    executor
        .reference()
        .write_tensor(ids_id.clone(), ids.clone());
    let invocation = invocation_reading("embedding", &[&id, &ids_id]);
    let actual = NativeState::f32_embedding(executor.reference(), &invocation, &id).unwrap();
    assert_eq!(actual, embedding_lookup(&table, &ids).unwrap());
}

/// A `KernelInvocation` of `kernel` reading `inputs`, for direct kernel
/// calls.
fn invocation_reading(kernel: &str, inputs: &[&TensorResourceId]) -> KernelInvocation {
    use magnetar_kernel::{kernel::KernelInvocationId, kernel::KernelResource};
    let advertisement = native_cpu_kernel_advertisements()
        .into_iter()
        .find(|advertisement| advertisement.id.name == kernel)
        .unwrap();
    let mut invocation = KernelInvocation::new(
        KernelInvocationId::new("test"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(NATIVE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    );
    for id in inputs {
        invocation = invocation.with_input(KernelResource::new(
            TensorResourceDescriptor::new(
                (*id).clone(),
                TensorDescriptor::materialized(
                    ShapeDescriptor::new(vec![1]),
                    DTypeDescriptor::portable(ComputeDType::Float32),
                ),
                ResourceAffinity::new(FallbackClass::Transparent),
            ),
            KernelMemoryClass::Host,
        ));
    }
    invocation
}

/// The SSE unpacking of a `Q3_K` super-block's sixteen scales matches the
/// Runtime's scalar one.
#[test]
#[cfg(target_arch = "x86_64")]
fn q3k_scales_unpack_like_the_runtime() {
    if simd_level() < SimdLevel::Avx2 {
        return;
    }
    let mut block = [0u8; 110];
    for (i, b) in block.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(37).wrapping_add(11);
    }
    let reference: Vec<f32> = magnetar_provider_api::q3_k_scales(&block[96..108])
        .iter()
        .map(|&s| f32::from(s - 32))
        .collect();
    // SAFETY: AVX2 was detected above.
    let simd = unsafe { quant::q3k_scales_avx2(&block, 1.0) };
    assert_eq!(simd.to_vec(), reference);
}

/// (queries, cached rows, key_count, window, causal).
type AttentionCase = (u64, u64, Option<u64>, Option<u64>, bool);

/// The multithreaded attention matches the Reference CPU kernel for
/// multi-head and grouped-query attention, causal prefill, a decode step
/// reading only `key_count` cached rows, a sliding window and bidirectional
/// attention, on one and several threads.
#[test]
fn attention_matches_the_reference_cpu() {
    use magnetar_core::operator::OperatorAttributeValue as Value;
    let (heads, kv_heads, dim) = (4_u64, 2_u64, 16_u64);
    let cases: [AttentionCase; 5] = [
        // (queries, cached rows, key_count, window, causal)
        (6, 6, None, None, true),
        (1, 9, Some(7), None, true),
        (5, 5, None, Some(3), true),
        (4, 4, None, None, false),
        (1, 7, None, None, true),
    ];
    for (threads, (queries, rows, key_count, window, causal)) in [1_usize, 4]
        .into_iter()
        .flat_map(|t| cases.into_iter().map(move |c| (t, c)))
    {
        let executor = NativeCpuExecutor::with_settings(true, true, threads);
        let q = activations(queries, heads * dim, 61);
        let k = activations(rows, kv_heads * dim, 62);
        let v = activations(rows, kv_heads * dim, 63);
        let ids = ["q", "k", "v"].map(TensorResourceId::new);
        for (id, tensor) in ids.iter().zip([&q, &k, &v]) {
            executor
                .reference()
                .write_tensor(id.clone(), tensor.clone());
        }
        let mut attributes = BTreeMap::new();
        attributes.insert("head_count".into(), Value::Integer(heads as i64));
        attributes.insert("head_dimension".into(), Value::Integer(dim as i64));
        attributes.insert("kv_head_count".into(), Value::Integer(kv_heads as i64));
        attributes.insert("causal".into(), Value::Boolean(causal));
        if let Some(count) = key_count {
            attributes.insert("key_count".into(), Value::Integer(count as i64));
        }
        if let Some(window) = window {
            attributes.insert("window_size".into(), Value::Integer(window as i64));
        }
        let invocation = invocation_reading("attention", &[&ids[0], &ids[1], &ids[2]])
            .with_attributes(attributes);
        let actual = executor
            .state
            .attention(executor.reference(), &invocation)
            .expect("handled natively")
            .unwrap();
        let (k, v) = match key_count {
            Some(count) => (
                leading_rows(&k, count).unwrap(),
                leading_rows(&v, count).unwrap(),
            ),
            None => (k, v),
        };
        let expected = attention(&q, &k, &v, heads, dim, Some(kv_heads), window, causal).unwrap();
        assert_close(
            &actual,
            &expected,
            &format!("{threads} threads, {queries} queries over {rows} rows"),
        );
    }
}

/// The optimized CPU Provider's own attention and embedding honour the
/// optional forms they read: an attention `scale` replaces
/// `1/sqrt(head_dimension)`, V rows of another width than Q/K rows are left
/// to the Reference CPU kernel, and an embedding `scale` multiplies the
/// gathered rows.
#[test]
fn attention_and_embedding_honour_their_optional_forms() {
    use magnetar_core::operator::OperatorAttributeValue as Value;
    let (heads, dim) = (2_u64, 8_u64);
    let executor = NativeCpuExecutor::with_settings(true, true, 2);
    let q = activations(3, heads * dim, 81);
    let k = activations(3, heads * dim, 82);
    let v = activations(3, heads * dim, 83);
    let ids = ["q", "k", "v"].map(TensorResourceId::new);
    for (id, tensor) in ids.iter().zip([&q, &k, &v]) {
        executor
            .reference()
            .write_tensor(id.clone(), tensor.clone());
    }
    let attributes = BTreeMap::from([
        ("head_count".to_string(), Value::Integer(heads as i64)),
        ("head_dimension".to_string(), Value::Integer(dim as i64)),
        ("causal".to_string(), Value::Boolean(true)),
        ("scale".to_string(), Value::Float(1.0)),
    ]);
    let invocation = invocation_reading("attention", &[&ids[0], &ids[1], &ids[2]])
        .with_attributes(attributes.clone());
    let actual = executor
        .state
        .attention(executor.reference(), &invocation)
        .expect("handled natively")
        .unwrap();
    let expected = attention_scaled(&q, &k, &v, heads, dim, None, None, true, Some(1.0)).unwrap();
    assert_close(&actual, &expected, "attention with scale 1");
    let unscaled = attention(&q, &k, &v, heads, dim, None, None, true).unwrap();
    assert!(
        actual
            .data
            .iter()
            .zip(&unscaled.data)
            .any(|(a, b)| (a - b).abs() > 1e-3),
        "the scale changes the result"
    );

    let mut wider_values = attributes.clone();
    wider_values.insert("value_head_dimension".into(), Value::Integer(4));
    let invocation =
        invocation_reading("attention", &[&ids[0], &ids[1], &ids[2]]).with_attributes(wider_values);
    assert!(
        executor
            .state
            .attention(executor.reference(), &invocation)
            .is_none(),
        "another V width runs the Reference CPU kernel"
    );
    let mut strided_values = attributes;
    strided_values.insert("value_head_stride".into(), Value::Integer(dim as i64));
    let invocation = invocation_reading("attention", &[&ids[0], &ids[1], &ids[2]])
        .with_attributes(strided_values);
    assert!(
        executor
            .state
            .attention(executor.reference(), &invocation)
            .is_none(),
        "V heads read with a stride run the Reference CPU kernel"
    );

    let table = activations(10, 64, 84);
    let table_id = TensorResourceId::new("table");
    executor
        .reference()
        .write_tensor(table_id.clone(), table.clone());
    let lookup = HostTensor::new([2], vec![3.0, 7.0]).unwrap();
    let lookup_id = TensorResourceId::new("ids");
    executor
        .reference()
        .write_tensor(lookup_id.clone(), lookup.clone());
    let invocation = invocation_reading("embedding", &[&table_id, &lookup_id])
        .with_attributes(BTreeMap::from([("scale".to_string(), Value::Float(2.5))]));
    let actual = executor
        .state
        .run(executor.reference(), &invocation)
        .expect("handled natively")
        .unwrap();
    let expected = embedding_lookup(&table, &lookup).unwrap();
    assert_eq!(
        actual.data,
        expected
            .data
            .iter()
            .map(|value| value * 2.5)
            .collect::<Vec<_>>()
    );
}

/// Decode-step attention on Mistral 7B shapes (32 heads, 8 KV heads of 128,
/// a 2048-row cache holding `key_count` keys) against the Reference CPU
/// kernel (diagnostic: run with `--ignored --nocapture`).
#[test]
#[ignore = "timing diagnostic"]
fn attention_decode_time() {
    use magnetar_core::operator::OperatorAttributeValue as Value;
    let threads = std::thread::available_parallelism().map_or(1, |count| count.get());
    let executor = NativeCpuExecutor::with_settings(true, true, threads);
    let (heads, kv_heads, dim, rows) = (32_u64, 8_u64, 128_u64, 2048_u64);
    let ids = ["q", "k", "v"].map(TensorResourceId::new);
    let tensors = [
        activations(1, heads * dim, 1),
        activations(rows, kv_heads * dim, 2),
        activations(rows, kv_heads * dim, 3),
    ];
    for (id, tensor) in ids.iter().zip(&tensors) {
        executor
            .reference()
            .write_tensor(id.clone(), tensor.clone());
    }
    for key_count in [64_u64, 512, 2048] {
        let mut attributes = BTreeMap::new();
        attributes.insert("head_count".into(), Value::Integer(heads as i64));
        attributes.insert("head_dimension".into(), Value::Integer(dim as i64));
        attributes.insert("kv_head_count".into(), Value::Integer(kv_heads as i64));
        attributes.insert("causal".into(), Value::Boolean(true));
        attributes.insert("key_count".into(), Value::Integer(key_count as i64));
        let invocation = invocation_reading("attention", &[&ids[0], &ids[1], &ids[2]])
            .with_attributes(attributes);
        let time = |run: &dyn Fn()| {
            run();
            let start = std::time::Instant::now();
            for _ in 0..20 {
                run();
            }
            start.elapsed().as_secs_f64() * 1e3 / 20.0
        };
        let native = time(&|| {
            executor
                .state
                .attention(executor.reference(), &invocation)
                .unwrap()
                .unwrap();
        });
        // What the Reference CPU kernel does: copy the inputs, then attend.
        let reference = time(&|| {
            let read = |i| executor.reference().input_tensor(&invocation, i).unwrap();
            let (q, k, v) = (read(0), read(1), read(2));
            let k = leading_rows(&k, key_count).unwrap();
            let v = leading_rows(&v, key_count).unwrap();
            attention(&q, &k, &v, heads, dim, Some(kv_heads), None, true).unwrap();
        });
        println!(
            "{key_count} keys: native {native:.3} ms ({threads} threads), reference {reference:.3} ms"
        );
    }
}

/// A `rows x cols` matrix of exact bfloat16 values in [-2, 2].
fn bf16_values(rows: u64, cols: u64, seed: u32) -> HostTensor {
    let data: Vec<f32> = activations(rows, cols, seed)
        .data
        .iter()
        .map(|&value| f32::from_bits((value * 2.0).to_bits() & 0xffff_0000))
        .collect();
    HostTensor::new([rows, cols], data).unwrap()
}

/// `speed-up-dense-cpu-matmuls`: dense products against compact and f32
/// weights, used transposed or not, match the Reference CPU matmul, with
/// and without the vector kernels, for decode and prefill row counts and
/// inner sizes off a multiple of the vector width.
#[test]
fn dense_matmuls_match_the_reference_cpu() {
    for threads in [1, 3] {
        let state = state(false, threads);
        for (m, k, n) in [
            (1, 64, 37),
            (1, 37, 41),
            (1, 300, 70),
            (3, 70, 200),
            (8, 96, 33),
            (5, 9, 8),
        ] {
            for transpose_b in [false, true] {
                let a = activations(m, k, 81 + m as u32);
                let shape = if transpose_b { [n, k] } else { [k, n] };
                let b = bf16_values(shape[0], shape[1], 83 + n as u32);
                let expected = matmul(&a, &b, false, transpose_b).unwrap();
                let bits: Vec<u16> = b.data.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
                for values in [dense::Values::F32(&b.data), dense::Values::Bf16(&bits)] {
                    let actual = state
                        .dense_product(
                            &a,
                            values,
                            [shape[0] as usize, shape[1] as usize],
                            transpose_b,
                        )
                        .expect("handled natively")
                        .unwrap();
                    assert_close(
                        &actual,
                        &expected,
                        &format!("{m}x{k}x{n}, transposed {transpose_b}, {threads} thread(s)"),
                    );
                    for (level, what) in [(SimdLevel::Scalar, "scalar"), (simd_level(), "simd")] {
                        let count = n as usize;
                        let mut out = vec![0.0_f32; m as usize * count];
                        if transpose_b {
                            dense::output_major(
                                level, &a.data, m as usize, k as usize, values, 0, count, &mut out,
                            );
                        } else {
                            dense::input_major(
                                level,
                                &a.data,
                                m as usize,
                                k as usize,
                                0..k as usize,
                                values,
                                count,
                                0,
                                count,
                                &mut out,
                            );
                        }
                        assert_close(
                            &HostTensor::new([m, n], out).unwrap(),
                            &expected,
                            &format!("{m}x{k}x{n}, transposed {transpose_b}, {what}"),
                        );
                    }
                }
            }
        }
    }
}

/// A Model Instance weight of exact bfloat16 values is stored compact and
/// reads back unchanged; an inexact or small one, or any with compact
/// weights off, stays f32; a later write replaces a compact weight.
#[test]
fn exact_bfloat16_weights_are_stored_compact() {
    let id = TensorResourceId::new("w");
    let weight = bf16_values(256, 256, 91);
    let executor = NativeCpuExecutor::with_settings(true, true, 2);
    executor
        .write_weight_tensor(id.clone(), weight.clone())
        .unwrap();
    assert!(executor.is_compact(&id));
    assert!(
        !executor
            .reference()
            .storage
            .lock()
            .unwrap()
            .contains_key(&id)
    );
    assert_eq!(executor.read_tensor(&id).unwrap(), weight);
    assert_eq!(executor.resident_shape(&id).unwrap(), vec![256, 256]);
    assert!(matches!(
        executor.read_tensor_value(&id),
        Some(TensorValue::Opaque)
    ));
    // Writing back the `Opaque` value keeps it; a new value replaces it.
    executor
        .write_tensor_value(id.clone(), TensorValue::Opaque)
        .unwrap();
    assert!(executor.is_compact(&id));
    let replacement = activations(256, 256, 92);
    executor
        .write_tensor(id.clone(), replacement.clone())
        .unwrap();
    assert!(!executor.is_compact(&id));
    assert_eq!(executor.read_tensor(&id).unwrap(), replacement);

    let inexact = TensorResourceId::new("inexact");
    executor
        .write_weight_tensor(inexact.clone(), activations(256, 256, 93))
        .unwrap();
    assert!(!executor.is_compact(&inexact));
    let small = TensorResourceId::new("small");
    executor
        .write_weight_tensor(small.clone(), bf16_values(4, 64, 94))
        .unwrap();
    assert!(!executor.is_compact(&small));
    let mut off = NativeCpuExecutor::with_settings(true, true, 1);
    off.compact_weights = false;
    off.write_weight_tensor(id.clone(), weight).unwrap();
    assert!(!off.is_compact(&id));
    assert!(executor.release_tensor(&inexact).unwrap());
}

/// The matmul family and the embedding lookup read compact weights in
/// place, with the Reference CPU kernels' results: a weight with more rows
/// than columns (held transposed) and one with fewer, each used transposed
/// and not.
#[test]
fn matmul_kernels_and_lookups_read_compact_weights() {
    use magnetar_core::operator::OperatorAttributeValue as Value;
    for (rows, cols) in [(640_u64, 128_u64), (128, 640)] {
        for transpose_b in [true, false] {
            for m in [1_u64, 3] {
                compact_weight_reads(rows, cols, transpose_b, m);
            }
        }
    }

    fn compact_weight_reads(rows: u64, cols: u64, transpose_b: bool, m: u64) {
        let what = format!("[{rows}, {cols}], transposed {transpose_b}, {m} row(s)");
        let executor = NativeCpuExecutor::with_settings(true, true, 2);
        let (k, n) = if transpose_b {
            (cols, rows)
        } else {
            (rows, cols)
        };
        let weight = bf16_values(rows, cols, 101);
        let ids = ["a", "w", "bias", "residual", "ids"].map(TensorResourceId::new);
        executor
            .write_weight_tensor(ids[1].clone(), weight.clone())
            .unwrap();
        assert!(executor.is_compact(&ids[1]));
        assert_eq!(executor.read_tensor(&ids[1]).unwrap(), weight, "{what}");
        assert_eq!(executor.resident_shape(&ids[1]).unwrap(), vec![rows, cols]);
        let a = activations(m, k, 102);
        let bias = activations(1, n, 103);
        let bias = HostTensor::new([n], bias.data).unwrap();
        let residual = activations(m, n, 104);
        for (id, tensor) in [(&ids[0], &a), (&ids[2], &bias), (&ids[3], &residual)] {
            executor
                .reference()
                .write_tensor(id.clone(), tensor.clone());
        }
        let mut attributes = BTreeMap::new();
        attributes.insert("transpose_b".into(), Value::Boolean(transpose_b));
        let product = matmul(&a, &weight, false, transpose_b).unwrap();
        for (kernel, extra, expected) in [
            ("matmul", None, product.clone()),
            ("matmul-bias", Some(&ids[2]), add(&product, &bias).unwrap()),
            (
                "matmul-residual",
                Some(&ids[3]),
                residual_add(&residual, &product).unwrap(),
            ),
        ] {
            let mut inputs = vec![&ids[0], &ids[1]];
            inputs.extend(extra);
            let invocation =
                invocation_reading(kernel, &inputs).with_attributes(attributes.clone());
            let actual = executor
                .state
                .run(executor.reference(), &invocation)
                .expect("handled natively")
                .unwrap();
            assert_close(&actual, &expected, &format!("{kernel}, {what}"));
        }
        // The weight as an embedding table: rows gathered exactly.
        let tokens = HostTensor::new([3], vec![5.0, 0.0, (rows - 1) as f32]).unwrap();
        executor
            .reference()
            .write_tensor(ids[4].clone(), tokens.clone());
        let invocation = invocation_reading("embedding", &[&ids[1], &ids[4]]);
        let actual = executor
            .state
            .run(executor.reference(), &invocation)
            .expect("handled natively")
            .unwrap();
        assert_eq!(
            actual,
            embedding_lookup(&weight, &tokens).unwrap(),
            "{what}"
        );
        // Nothing but the read-back above was expanded to f32.
        assert_eq!(executor.state.materialized.lock().unwrap().0, 1, "{what}");
    }
}

/// Diagnostic: a decode-step dense matmul's time and weight bandwidth,
/// compact and f32, output-major (a transposed checkpoint weight) and
/// input-major.
#[test]
#[ignore = "timing diagnostic"]
fn dense_matmul_bandwidth() {
    let threads = std::thread::available_parallelism().map_or(1, |count| count.get());
    let state = state(false, threads);
    println!("{threads} threads, {:?}", simd_level());
    let rows = std::env::var("MAGNETAR_CPU_BENCH_ROWS")
        .ok()
        .and_then(|rows| rows.parse::<u64>().ok())
        .unwrap_or(1);
    for (k, n) in [(896_u64, 4864_u64), (4864, 896), (896, 151_936)] {
        let a = activations(rows, k, 111);
        for transpose_b in [true, false] {
            let shape = if transpose_b { [n, k] } else { [k, n] };
            let b = bf16_values(shape[0], shape[1], 112);
            let bits: Vec<u16> = b.data.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
            for (values, bytes, what) in [
                (dense::Values::F32(&b.data), b.data.len() * 4, "f32"),
                (dense::Values::Bf16(&bits), bits.len() * 2, "bf16"),
            ] {
                let shape = [shape[0] as usize, shape[1] as usize];
                state.dense_product(&a, values, shape, transpose_b);
                let repeats = 20;
                let start = std::time::Instant::now();
                for _ in 0..repeats {
                    state.dense_product(&a, values, shape, transpose_b);
                }
                let seconds = start.elapsed().as_secs_f64() / f64::from(repeats);
                println!(
                    "{what} [{rows}, {k}] x [{k}, {n}] transposed {transpose_b}: {:.3} ms ({:.1} GB/s)",
                    seconds * 1e3,
                    bytes as f64 / seconds / 1e9
                );
            }
        }
    }
}

/// Diagnostic: what one pool run of trivial tasks costs (the fixed cost
/// every pooled Kernel pays), and a decode-sized matmul against weights
/// streamed from memory rather than cache.
#[test]
#[ignore = "timing diagnostic"]
fn pool_and_streamed_matmul_costs() {
    let threads = std::thread::available_parallelism().map_or(1, |count| count.get());
    let state = state(false, threads);
    let repeats = 2000;
    let start = std::time::Instant::now();
    for _ in 0..repeats {
        state.pool.run(threads * 4, &|_| {});
    }
    println!(
        "pool run of {} empty tasks: {:.1} us",
        threads * 4,
        start.elapsed().as_secs_f64() / f64::from(repeats) * 1e6
    );
    // Distinct bfloat16 weights of each decode shape of a Qwen2.5-0.5B
    // layer (about 400 MB of each): each read once per pass, as a decode
    // step reads its layers', in both stored layouts.
    let rows = std::env::var("MAGNETAR_CPU_BENCH_ROWS")
        .ok()
        .and_then(|rows| rows.parse::<u64>().ok())
        .unwrap_or(1);
    for (k, n) in [(896_u64, 4864_u64), (4864, 896), (896, 896), (896, 128)] {
        let a = activations(rows, k, 121);
        let count = (400_000_000 / (n * k * 2)).clamp(4, 256);
        let weights: Vec<Vec<u16>> = (0..count)
            .map(|seed| {
                bf16_values(n, k, 122 + seed as u32)
                    .data
                    .iter()
                    .map(|v| (v.to_bits() >> 16) as u16)
                    .collect()
            })
            .collect();
        for transpose_b in [true, false] {
            let shape = if transpose_b {
                [n as usize, k as usize]
            } else {
                [k as usize, n as usize]
            };
            let mut best = f64::MAX;
            for _ in 0..3 {
                let start = std::time::Instant::now();
                for bits in &weights {
                    state.dense_product(&a, dense::Values::Bf16(bits), shape, transpose_b);
                }
                best = best.min(start.elapsed().as_secs_f64() / count as f64);
            }
            println!(
                "streamed bf16 [{rows}, {k}] x [{k}, {n}] {}: {:.3} ms ({:.1} GB/s)",
                if transpose_b {
                    "output-major"
                } else {
                    "input-major"
                },
                best * 1e3,
                (n * k * 2) as f64 / best / 1e9
            );
        }
    }
}

/// Diagnostic: llama.cpp's pattern for a one-row product, beside
/// `pool_and_streamed_matmul_costs`: every weight stored output-major,
/// each task a contiguous range of output rows, one dot product at a time
/// (each row read in order). Run on 4 and 2 tasks (ggml's default thread
/// count is the physical cores), so each runner's CPU says whether this
/// pattern streams faster than the kernels the Provider uses.
#[test]
#[ignore = "diagnostic: streams about 400 MB per shape"]
fn row_streamed_matmul_costs() {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_rows(
        a: &[f32],
        bits: &[u16],
        k: usize,
        rows: std::ops::Range<usize>,
        out: &mut [f32],
    ) {
        use std::arch::x86_64::*;
        for (value, row) in out.iter_mut().zip(rows) {
            let w = &bits[row * k..(row + 1) * k];
            let mut acc = [_mm256_setzero_ps(); 4];
            let mut i = 0;
            unsafe {
                while i + 32 <= k {
                    for (j, acc) in acc.iter_mut().enumerate() {
                        let half = _mm_loadu_si128(w.as_ptr().add(i + 8 * j) as *const __m128i);
                        let weights = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(
                            _mm256_cvtepu16_epi32(half),
                        ));
                        let x = _mm256_loadu_ps(a.as_ptr().add(i + 8 * j));
                        *acc = _mm256_fmadd_ps(x, weights, *acc);
                    }
                    i += 32;
                }
                let sum =
                    _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
                let mut lanes = [0.0_f32; 8];
                _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
                let mut total: f32 = lanes.iter().sum();
                for t in i..k {
                    total += a[t] * f32::from_bits(u32::from(w[t]) << 16);
                }
                *value = total;
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        let threads = std::thread::available_parallelism().map_or(1, |count| count.get());
        let state = state(false, threads);
        for (k, n) in [(896_usize, 4864_usize), (4864, 896), (896, 896), (896, 128)] {
            let a = activations(1, k as u64, 121);
            let count = (400_000_000 / (n * k * 2)).clamp(4, 256);
            let weights: Vec<Vec<u16>> = (0..count)
                .map(|seed| {
                    bf16_values(n as u64, k as u64, 122 + seed as u32)
                        .data
                        .iter()
                        .map(|v| (v.to_bits() >> 16) as u16)
                        .collect()
                })
                .collect();
            for tasks in [threads, 2.min(threads)] {
                let mut out = vec![0.0_f32; n];
                let mut best = f64::MAX;
                for _ in 0..3 {
                    let start = std::time::Instant::now();
                    for bits in &weights {
                        let shared = SharedOut(out.as_mut_ptr());
                        let shared = &shared;
                        state.pool.run(tasks, &|task| {
                            let rows = task * n / tasks..(task + 1) * n / tasks;
                            // SAFETY: rows `rows` of the output belong to this task only.
                            let slice = unsafe {
                                std::slice::from_raw_parts_mut(shared.0.add(rows.start), rows.len())
                            };
                            // SAFETY: AVX2 and FMA were detected above.
                            unsafe { dot_rows(&a.data, bits, k, rows, slice) };
                        });
                    }
                    best = best.min(start.elapsed().as_secs_f64() / count as f64);
                }
                std::hint::black_box(&out);
                println!(
                    "streamed bf16 [1, {k}] x [{k}, {n}] row-at-a-time, {tasks} tasks: {:.3} ms ({:.1} GB/s)",
                    best * 1e3,
                    (n * k * 2) as f64 / best / 1e9
                );
            }
        }
    }
}

/// Diagnostic: the matmuls of a Qwen2.5-0.5B decode step in their order,
/// on distinct bfloat16 weights in their stored layouts (24 layers and a
/// tied LM head, about 1 GB), so each reads its weight from memory as in a
/// real step: one product at a time (the gated MLP's two then their
/// activation), against the query, key and value projections in one pool
/// run and the gated MLP's two products in another, without and with the
/// one-row kernels reading the next rows ahead, alternating step by step.
/// Prints each class's time and the steps' totals.
#[test]
#[ignore = "diagnostic: allocates about 1 GB"]
fn decode_step_matmul_costs() {
    let threads = std::thread::available_parallelism().map_or(1, |count| count.get());
    let state = state(false, threads);
    let mut seed = 0x9e37_79b9_u32;
    let mut weight = |rows: usize, cols: usize| -> CompactWeight {
        let bits = (0..rows * cols)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                // A bfloat16 of magnitude below 2, either sign.
                0x3c00 | (seed as u16 & 0x83ff)
            })
            .collect();
        // Stored as `CompactWeight::of` stores it: transposed when it has
        // more rows than columns.
        CompactWeight {
            rows,
            cols,
            transposed: rows > cols,
            bits,
        }
    };
    struct Layer {
        q: CompactWeight,
        k: CompactWeight,
        v: CompactWeight,
        o: CompactWeight,
        gate: CompactWeight,
        up: CompactWeight,
        down: CompactWeight,
    }
    // Logical `[k, n]` weights.
    let layers: Vec<Layer> = (0..24)
        .map(|_| Layer {
            q: weight(896, 896),
            k: weight(896, 128),
            v: weight(896, 128),
            o: weight(896, 896),
            gate: weight(896, 4864),
            up: weight(896, 4864),
            down: weight(4864, 896),
        })
        .collect();
    let head = weight(896, 151936);
    let narrow = activations(1, 896, 301);
    let wide = activations(1, 4864, 302);
    fn plan<'w>(state: &NativeState, a: &HostTensor, w: &'w CompactWeight) -> DenseProduct<'w> {
        state
            .dense_plan(
                a,
                dense::Values::Bf16(&w.bits),
                w.stored_shape(),
                w.transposed,
            )
            .unwrap()
    }
    let plan = |a: &HostTensor, w| plan(&state, a, w);
    let product = |a: &HostTensor, w: &CompactWeight| {
        state
            .dense_product(
                a,
                dense::Values::Bf16(&w.bits),
                w.stored_shape(),
                w.transposed,
            )
            .unwrap()
            .unwrap()
    };
    let mut totals: BTreeMap<String, f64> = BTreeMap::new();
    let mut add = |key: String, begin: std::time::Instant, measured: bool| {
        if measured {
            *totals.entry(key).or_default() += begin.elapsed().as_secs_f64();
        }
    };
    let steps = 16;
    for step in 0..steps {
        for (variant, grouped, ahead) in [
            ("apart", false, false),
            ("grouped", true, false),
            ("grouped+prefetch", true, true),
        ] {
            dense::SOFTWARE_PREFETCH.store(ahead, std::sync::atomic::Ordering::Relaxed);
            let measured = step >= 2;
            let start = std::time::Instant::now();
            for layer in &layers {
                let begin = std::time::Instant::now();
                if grouped {
                    let products = [
                        plan(&narrow, &layer.q),
                        plan(&narrow, &layer.k),
                        plan(&narrow, &layer.v),
                    ];
                    let computed = state.run_products(&narrow.data, &products);
                    for (computed, product) in computed.into_iter().zip(&products) {
                        std::hint::black_box(computed.into_sum(product.m * product.n));
                    }
                } else {
                    for w in [&layer.q, &layer.k, &layer.v] {
                        std::hint::black_box(product(&narrow, w));
                    }
                }
                add(format!("{variant} qkv"), begin, measured);
                let begin = std::time::Instant::now();
                std::hint::black_box(product(&narrow, &layer.o));
                add(format!("{variant} o"), begin, measured);
                let begin = std::time::Instant::now();
                if grouped {
                    let products = [plan(&narrow, &layer.gate), plan(&narrow, &layer.up)];
                    let computed = state.run_products(&narrow.data, &products);
                    let len = products[0].n;
                    let out: Vec<f32> = (0..len)
                        .map(|i| silu_times_value(computed[0].get(len, i), computed[1].get(len, i)))
                        .collect();
                    std::hint::black_box(out);
                } else {
                    let gate = product(&narrow, &layer.gate);
                    let up = product(&narrow, &layer.up);
                    std::hint::black_box(silu_times(gate, &up).unwrap());
                }
                add(format!("{variant} gate-up"), begin, measured);
                let begin = std::time::Instant::now();
                std::hint::black_box(product(&wide, &layer.down));
                add(format!("{variant} down"), begin, measured);
            }
            let begin = std::time::Instant::now();
            std::hint::black_box(product(&narrow, &head));
            add(format!("{variant} lm-head"), begin, measured);
            add(format!("{variant} step"), start, measured);
        }
    }
    dense::SOFTWARE_PREFETCH.store(true, std::sync::atomic::Ordering::Relaxed);
    let measured = (steps - 2) as f64;
    for (key, seconds) in &totals {
        println!(
            "decode step {key}: {:.3} ms per step",
            seconds / measured * 1e3
        );
    }
}

/// A replayed step runs dense products of one input (a layer's query,
/// key and value projections) in one pool run, moving a later one ahead
/// of invocations that neither write what it reads nor read or write what
/// it writes: each output is the one running the invocations one by one
/// gives, bit for bit, and a product writing what the group reads runs
/// after it.
#[test]
fn replayed_product_groups_match_products_run_one_by_one() {
    use magnetar_kernel::{kernel::KernelInvocationId, kernel::KernelResource};
    use magnetar_provider_api::StepCaptureRequest;
    let advertisements = native_cpu_kernel_advertisements();
    let catalog = magnetar_core::operator::initial_operator_catalog();
    let find = |name: &str| {
        advertisements
            .iter()
            .find(|advertisement| advertisement.id.name == name)
            .unwrap()
            .clone()
    };
    let resource = |id: &str, shape: Vec<u64>| {
        KernelResource::new(
            TensorResourceDescriptor::new(
                TensorResourceId::new(id),
                TensorDescriptor::materialized(
                    ShapeDescriptor::new(shape),
                    DTypeDescriptor::portable(ComputeDType::Float32),
                ),
                ResourceAffinity::new(FallbackClass::Transparent),
            ),
            KernelMemoryClass::Host,
        )
    };
    let id = |name: &str| TensorResourceId::new(name);
    // Every weight past `COMPACT_WEIGHT_MIN_ELEMENTS`.
    let k = 1024_u64;
    for m in [1_u64, 3] {
        // (kernel, output, input, input width, outputs, third input): a
        // query projection held input-major, a key projection held
        // transposed (more rows than columns), a product of the query
        // (another input), a value projection that moves ahead of it, then
        // a product that overwrites the shared input.
        let steps = [
            ("matmul-bias", "q", "x", k, 1280_u64, Some("q_bias")),
            ("matmul-bias", "k", "x", k, 64, Some("k_bias")),
            ("matmul", "qq", "q", 1280, 64, None),
            ("matmul", "v", "x", k, 64, None),
            ("matmul-residual", "x", "x", k, k, Some("x")),
        ];
        let invocations: Vec<KernelInvocation> = steps
            .iter()
            .map(|&(kernel, output, input, width, n, third)| {
                let advertisement = find(kernel);
                let mut invocation = KernelInvocation::new(
                    KernelInvocationId::new(output),
                    advertisement.implemented_operator.clone(),
                    advertisement.id.clone(),
                    ProviderBinding::new(NATIVE_CPU_PROVIDER_NAME),
                    ResourceAffinity::new(FallbackClass::Transparent),
                )
                .with_input(resource(input, vec![m, width]))
                .with_input(resource(&format!("w_{output}"), vec![width, n]));
                if let Some(third) = third {
                    invocation = invocation
                        .with_input(resource(third, vec![if third == "x" { m } else { 1 }, n]));
                }
                invocation.with_output(resource(output, vec![m, n]))
            })
            .collect();
        let setup = |executor: &NativeCpuExecutor, x: &HostTensor| {
            for (seed, &(_, output, _, width, n, third)) in steps.iter().enumerate() {
                let weight = id(&format!("w_{output}"));
                executor
                    .write_weight_tensor(weight.clone(), bf16_values(width, n, 140 + seed as u32))
                    .unwrap();
                assert!(executor.is_compact(&weight));
                if let Some(third) = third.filter(|third| *third != "x") {
                    executor
                        .write_tensor(id(third), activations(1, n, 150 + seed as u32))
                        .unwrap();
                }
            }
            executor.write_tensor(id("x"), x.clone()).unwrap();
        };
        let run = |executor: &NativeCpuExecutor| {
            let mut memory = MemoryManager::default();
            for invocation in &invocations {
                let advertisement = find(&invocation.kernel.name);
                let operator = catalog.get(&advertisement.implemented_operator).unwrap();
                let handle = executor
                    .submit_kernel(&advertisement, operator, invocation, &mut memory)
                    .unwrap();
                assert_eq!(
                    executor.complete_kernel(&handle).unwrap().status,
                    KernelResultStatus::Succeeded
                );
            }
        };
        let (x0, x1) = (activations(m, k, 160), activations(m, k, 161));
        let replayed = NativeCpuExecutor::with_settings(true, true, 3);
        setup(&replayed, &x0);
        let references: Vec<&KernelInvocation> = invocations.iter().collect();
        let schedule: Vec<Vec<usize>> = replayed
            .state
            .schedule(&references)
            .into_iter()
            .map(|step| step.invocations)
            .collect();
        assert_eq!(schedule, [vec![0, 1, 3], vec![2], vec![4]]);
        replayed
            .begin_step_capture(&StepCaptureRequest {
                advancing_attributes: Vec::new(),
            })
            .unwrap();
        run(&replayed);
        let capture = replayed.end_step_capture().unwrap();
        replayed
            .replay_step(&capture, 0, vec![(id("x"), x1.clone())])
            .unwrap();
        let one_by_one = NativeCpuExecutor::with_settings(true, true, 3);
        setup(&one_by_one, &x1);
        run(&one_by_one);
        for output in ["q", "k", "qq", "v", "x"] {
            let (got, want) = (
                replayed.read_tensor(&id(output)).unwrap(),
                one_by_one.read_tensor(&id(output)).unwrap(),
            );
            assert_eq!(got.shape, want.shape, "{output}, {m} row(s)");
            assert!(
                got.data
                    .iter()
                    .zip(&want.data)
                    .all(|(x, y)| x.to_bits() == y.to_bits()),
                "{output}, {m} row(s): the grouped replay's values differ"
            );
        }
    }
}

/// `capture-cpu-steps`: a captured step ran as it was captured, and a
/// replay writes its inputs and runs its invocations again with every
/// advancing attribute increased by the offset; an offset off a store
/// fails the replay, a host write during a capture fails the capture, a
/// released capture no longer replays, and capture can be turned off.
#[test]
fn captured_steps_replay_with_advanced_attributes() {
    use magnetar_core::operator::OperatorAttributeValue as Value;
    use magnetar_kernel::{kernel::KernelInvocationId, kernel::KernelResource};
    use magnetar_provider_api::{StepCaptureRequest, StepReplayError};
    let executor = NativeCpuExecutor::with_settings(true, true, 2);
    assert!(executor.supports_step_capture());
    let catalog = magnetar_core::operator::initial_operator_catalog();
    let advertisements = native_cpu_kernel_advertisements();
    let find = |name: &str| {
        advertisements
            .iter()
            .find(|advertisement| advertisement.id.name == name)
            .unwrap()
            .clone()
    };
    let resource = |id: &str, shape: Vec<u64>| {
        KernelResource::new(
            TensorResourceDescriptor::new(
                TensorResourceId::new(id),
                TensorDescriptor::materialized(
                    ShapeDescriptor::new(shape),
                    DTypeDescriptor::portable(ComputeDType::Float32),
                ),
                ResourceAffinity::new(FallbackClass::Transparent),
            ),
            KernelMemoryClass::Host,
        )
    };
    let invocation = |advertisement: &KernelAdvertisement, id: &str| {
        KernelInvocation::new(
            KernelInvocationId::new(id),
            advertisement.implemented_operator.clone(),
            advertisement.id.clone(),
            ProviderBinding::new(NATIVE_CPU_PROVIDER_NAME),
            ResourceAffinity::new(FallbackClass::Transparent),
        )
    };
    let (k, n, rows) = (16_u64, 8_u64, 6_u64);
    let w = activations(k, n, 131);
    let x0 = activations(1, k, 132);
    let id = |name: &str| TensorResourceId::new(name);
    executor.write_tensor(id("w"), w.clone()).unwrap();
    executor
        .write_tensor(
            id("kv"),
            HostTensor::new([rows, n], vec![0.0; (rows * n) as usize]).unwrap(),
        )
        .unwrap();
    executor.write_tensor(id("x"), x0.clone()).unwrap();
    let (matmul_kernel, kv_write) = (find("matmul"), find("kv-write"));
    let project = invocation(&matmul_kernel, "project")
        .with_input(resource("x", vec![1, k]))
        .with_input(resource("w", vec![k, n]))
        .with_output(resource("h", vec![1, n]));
    let write = invocation(&kv_write, "write")
        .with_input(resource("kv", vec![rows, n]))
        .with_input(resource("h", vec![1, n]))
        .with_output(resource("kv", vec![rows, n]))
        .with_attributes(BTreeMap::from([(
            "row_offset".to_string(),
            Value::Integer(1),
        )]));
    let mut memory = MemoryManager::default();
    let request = StepCaptureRequest {
        advancing_attributes: vec!["row_offset".into()],
    };
    executor.begin_step_capture(&request).unwrap();
    for (advertisement, invocation) in [(&matmul_kernel, &project), (&kv_write, &write)] {
        let operator = catalog.get(&advertisement.implemented_operator).unwrap();
        let handle = executor
            .submit_kernel(advertisement, operator, invocation, &mut memory)
            .unwrap();
        assert_eq!(
            executor.complete_kernel(&handle).unwrap().status,
            KernelResultStatus::Succeeded
        );
    }
    let capture = executor.end_step_capture().unwrap();
    let row = |store: &HostTensor, r: u64| {
        HostTensor::new(
            [1, n],
            store.data[(r * n) as usize..((r + 1) * n) as usize].to_vec(),
        )
        .unwrap()
    };
    let x1 = activations(1, k, 133);
    executor
        .replay_step(&capture, 3, vec![(id("x"), x1.clone())])
        .unwrap();
    let kv = executor.read_tensor(&id("kv")).unwrap();
    assert_close(
        &row(&kv, 1),
        &matmul(&x0, &w, false, false).unwrap(),
        "captured row",
    );
    assert_close(
        &row(&kv, 4),
        &matmul(&x1, &w, false, false).unwrap(),
        "replayed row",
    );
    assert!(row(&kv, 2).data.iter().all(|&value| value == 0.0));
    assert!(matches!(
        executor.replay_step(&capture, 5, Vec::new()),
        Err(StepReplayError::Failed(_))
    ));
    executor.begin_step_capture(&request).unwrap();
    executor.write_tensor(id("x"), x1).unwrap();
    assert!(executor.end_step_capture().is_err());
    executor.release_step_capture(&capture);
    assert!(matches!(
        executor.replay_step(&capture, 0, Vec::new()),
        Err(StepReplayError::Invalidated { .. })
    ));
    let mut off = NativeCpuExecutor::with_settings(true, true, 1);
    off.step_capture = false;
    assert!(!off.supports_step_capture());
    assert!(off.begin_step_capture(&request).is_err());
}

/// Greedy selection on the Provider reads the logits row in place and
/// honours exclusions and the temperature, like host sampling.
#[test]
fn greedy_columns_are_selected_in_place() {
    let executor = NativeCpuExecutor::with_settings(true, true, 1);
    let logits = TensorResourceId::new("logits");
    let values = vec![0.5, 9.0, 1.0, 3.0, f32::NAN, 0.0, 2.0, 7.0];
    executor
        .write_tensor(logits.clone(), HostTensor::new([2, 4], values).unwrap())
        .unwrap();
    assert_eq!(executor.select_greedy_column(&logits, 0, &[], 0.0), Some(1));
    assert_eq!(
        executor.select_greedy_column(&logits, 0, std::slice::from_ref(&(1..2)), 0.5),
        Some(3)
    );
    assert_eq!(executor.select_greedy_column(&logits, 1, &[], 0.0), Some(3));
    assert_eq!(executor.select_greedy_column(&logits, 2, &[], 0.0), None);
}

/// While a bucketed prefill replays, row matmuls compute only the rows
/// before its padding (the padding rows are zero); a one-row product, or
/// one no longer than the padding, is computed whole.
#[test]
fn replayed_prefills_skip_their_padding_rows() {
    use magnetar_core::operator::OperatorAttributeValue as Value;
    let executor = NativeCpuExecutor::with_settings(true, true, 2);
    let (k, n) = (128_u64, 512_u64);
    let weight = bf16_values(n, k, 141);
    let ids = ["a", "w"].map(TensorResourceId::new);
    executor
        .write_weight_tensor(ids[1].clone(), weight.clone())
        .unwrap();
    let attributes = BTreeMap::from([("transpose_b".to_string(), Value::Boolean(true))]);
    let invocation = invocation_reading("matmul", &[&ids[0], &ids[1]]).with_attributes(attributes);
    for (rows, padding, live) in [(8_u64, 3_usize, 5_u64), (1, 3, 1), (3, 3, 3)] {
        let a = activations(rows, k, 142);
        executor.reference().write_tensor(ids[0].clone(), a.clone());
        executor
            .state
            .padding_rows
            .store(padding, std::sync::atomic::Ordering::Relaxed);
        let actual = executor
            .state
            .run(executor.reference(), &invocation)
            .expect("handled natively")
            .unwrap();
        executor
            .state
            .padding_rows
            .store(0, std::sync::atomic::Ordering::Relaxed);
        let expected = matmul(&a, &weight, false, true).unwrap();
        assert_eq!(actual.shape, expected.shape);
        let split = (live * n) as usize;
        assert_close(
            &HostTensor::new([live, n], actual.data[..split].to_vec()).unwrap(),
            &HostTensor::new([live, n], expected.data[..split].to_vec()).unwrap(),
            &format!("{rows} rows, padding {padding}"),
        );
        assert!(actual.data[split..].iter().all(|&value| value == 0.0));
    }
}

/// Both vector widths of the dense kernels (AVX2, and AVX-512F where the
/// host has it) match the Reference CPU matmul, for decode and prefill row
/// counts, inner and outer sizes off a multiple of either width, both
/// layouts and both weight encodings.
#[test]
#[cfg(target_arch = "x86_64")]
fn both_vector_widths_match_the_reference_cpu() {
    if simd_level() != SimdLevel::Avx2 {
        return;
    }
    let wide = std::is_x86_feature_detected!("avx512f");
    for (m, k, n) in [
        (1, 64, 48),
        (1, 200, 37),
        (1, 37, 41),
        (1, 300, 70),
        (3, 70, 200),
        (8, 136, 40),
        (5, 9, 17),
    ] {
        for transpose_b in [false, true] {
            let a = activations(m, k, 151 + m as u32);
            let shape = if transpose_b { [n, k] } else { [k, n] };
            let b = bf16_values(shape[0], shape[1], 152 + n as u32);
            let expected = matmul(&a, &b, false, transpose_b).unwrap();
            let bits: Vec<u16> = b.data.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
            let (m, k, n) = (m as usize, k as usize, n as usize);
            for values in [dense::Values::F32(&b.data), dense::Values::Bf16(&bits)] {
                for width in [256, 512].into_iter().filter(|&width| width == 256 || wide) {
                    let mut out = vec![0.0_f32; m * n];
                    // SAFETY: AVX2 and FMA are present (`simd_level`), and
                    // AVX-512F where width 512 runs.
                    unsafe {
                        match (width, transpose_b) {
                            (256, true) => {
                                dense::avx2::output_major(&a.data, m, k, values, 0, n, &mut out)
                            }
                            (256, false) => dense::avx2::input_major(
                                &a.data,
                                m,
                                k,
                                0..k,
                                values,
                                n,
                                0,
                                n,
                                &mut out,
                            ),
                            (_, true) => {
                                dense::avx512::output_major(&a.data, m, k, values, 0, n, &mut out)
                            }
                            (_, false) => dense::avx512::input_major(
                                &a.data,
                                m,
                                k,
                                0..k,
                                values,
                                n,
                                0,
                                n,
                                &mut out,
                            ),
                        }
                    }
                    assert_close(
                        &HostTensor::new([m as u64, n as u64], out).unwrap(),
                        &expected,
                        &format!("{m}x{k}x{n}, transposed {transpose_b}, width {width}"),
                    );
                }
            }
        }
    }
}

/// Diagnostic: what one invocation of a decode step's small Kernels costs
/// through the executor (input reads, the Kernel, the output write), next
/// to the same Kernel's arithmetic alone.
#[test]
#[ignore = "timing diagnostic"]
fn small_kernel_invocation_costs() {
    use magnetar_core::operator::OperatorAttributeValue as Value;
    let executor = NativeCpuExecutor::new();
    let ids = ["gate", "up", "x", "norm", "w", "bias", "out"].map(TensorResourceId::new);
    let gate = activations(1, 4864, 201);
    let up = activations(1, 4864, 202);
    let x = activations(1, 896, 203);
    let norm = activations(1, 896, 204);
    let norm = HostTensor::new([896], norm.data).unwrap();
    let bias = HostTensor::new([128], activations(1, 128, 206).data).unwrap();
    for (id, tensor) in [
        (&ids[0], &gate),
        (&ids[1], &up),
        (&ids[2], &x),
        (&ids[3], &norm),
        (&ids[5], &bias),
    ] {
        executor
            .reference()
            .write_tensor(id.clone(), tensor.clone());
    }
    executor
        .write_weight_tensor(ids[4].clone(), bf16_values(896, 128, 205))
        .unwrap();
    let time = |what: &str, invocation: &KernelInvocation| {
        for _ in 0..200 {
            executor.reference().run_invocation(invocation).unwrap();
        }
        let repeats = 5000;
        let start = std::time::Instant::now();
        for _ in 0..repeats {
            executor.reference().run_invocation(invocation).unwrap();
        }
        println!(
            "{what}: {:.1} us per invocation",
            start.elapsed().as_secs_f64() / f64::from(repeats) * 1e6
        );
    };
    let with_output = |invocation: KernelInvocation| invocation_writing(invocation, &ids[6]);
    time(
        "silu-mul [1, 4864]",
        &with_output(invocation_reading("silu-mul", &[&ids[0], &ids[1]])),
    );
    let mut attributes = BTreeMap::new();
    attributes.insert("epsilon".into(), Value::Float(1e-6));
    time(
        "rmsnorm [1, 896]",
        &with_output(
            invocation_reading("rmsnorm", &[&ids[2], &ids[3]]).with_attributes(attributes),
        ),
    );
    time(
        "matmul-bias [1, 896] x [896, 128]",
        &with_output(invocation_reading(
            "matmul-bias",
            &[&ids[2], &ids[4], &ids[5]],
        )),
    );
    // The arithmetic alone.
    let repeats = 5000;
    let start = std::time::Instant::now();
    for _ in 0..repeats {
        std::hint::black_box(crate::mul(&crate::silu(&gate), &up).unwrap());
    }
    println!(
        "silu then mul alone: {:.1} us",
        start.elapsed().as_secs_f64() / f64::from(repeats) * 1e6
    );
    let start = std::time::Instant::now();
    for _ in 0..repeats {
        std::hint::black_box(executor.reference().read_tensor(&ids[0]).unwrap());
    }
    println!(
        "one [1, 4864] read: {:.1} us",
        start.elapsed().as_secs_f64() / f64::from(repeats) * 1e6
    );
}

/// `invocation` writing its output to `id`.
fn invocation_writing(invocation: KernelInvocation, id: &TensorResourceId) -> KernelInvocation {
    use magnetar_kernel::kernel::KernelResource;
    invocation.with_output(KernelResource::new(
        TensorResourceDescriptor::new(
            id.clone(),
            TensorDescriptor::materialized(
                ShapeDescriptor::new(vec![1]),
                DTypeDescriptor::portable(ComputeDType::Float32),
            ),
            ResourceAffinity::new(FallbackClass::Transparent),
        ),
        KernelMemoryClass::Host,
    ))
}

/// `gated-matmul` against compact, f32 and packed weights, for one and
/// several rows, is the Reference CPU `silu(a x gate) * (a x up)`, and the
/// optimized CPU Provider advertises it, so the Runtime fuses it there.
#[test]
fn gated_matmuls_match_the_reference_cpu() {
    assert!(
        native_cpu_kernel_advertisements()
            .iter()
            .any(|advertisement| advertisement.id.name == "gated-matmul")
    );
    // An input-major weight (shared out by row blocks), then one held
    // transposed (by column blocks).
    for (k, n, m, kind) in [(256_u64, 640_u64), (640, 256)]
        .into_iter()
        .flat_map(|(k, n)| {
            [1_u64, 5].into_iter().flat_map(move |m| {
                ["compact", "f32", "packed"]
                    .into_iter()
                    .map(move |kind| (k, n, m, kind))
            })
        })
    {
        {
            let executor = NativeCpuExecutor::with_settings(true, false, 2);
            let ids = ["a", "gate", "up", "out"].map(TensorResourceId::new);
            let (gate, up) = match kind {
                "packed" => {
                    let gate = packed(PackedWeightFormat::Q8_0, n, k, true, 301);
                    let up = packed(PackedWeightFormat::Q8_0, n, k, true, 302);
                    let dense = (gate.dequantize(), up.dequantize());
                    executor
                        .write_packed_weight_tensor(ids[1].clone(), gate)
                        .unwrap();
                    executor
                        .write_packed_weight_tensor(ids[2].clone(), up)
                        .unwrap();
                    dense
                }
                _ => {
                    let (gate, up) = if kind == "compact" {
                        (bf16_values(k, n, 303), bf16_values(k, n, 304))
                    } else {
                        (activations(k, n, 303), activations(k, n, 304))
                    };
                    executor
                        .write_weight_tensor(ids[1].clone(), gate.clone())
                        .unwrap();
                    executor
                        .write_weight_tensor(ids[2].clone(), up.clone())
                        .unwrap();
                    assert_eq!(executor.is_compact(&ids[1]), kind == "compact");
                    (gate, up)
                }
            };
            let a = activations(m, k, 305);
            executor.reference().write_tensor(ids[0].clone(), a.clone());
            let expected = crate::mul(
                &crate::silu(&matmul(&a, &gate, false, false).unwrap()),
                &matmul(&a, &up, false, false).unwrap(),
            )
            .unwrap();
            let invocation = invocation_writing(
                invocation_reading("gated-matmul", &[&ids[0], &ids[1], &ids[2]]),
                &ids[3],
            );
            let actual = executor
                .state
                .run(executor.reference(), &invocation)
                .expect("handled natively")
                .unwrap();
            assert_close(
                &actual,
                &expected,
                &format!("{kind}, {m} row(s), [{k}, {n}]"),
            );
            if kind == "compact" {
                // One pool run for both products: the values of the two
                // products run apart, then `silu_times`, bit for bit.
                let product = |weight: &TensorResourceId| {
                    let invocation = invocation_writing(
                        invocation_reading("matmul", &[&ids[0], weight]),
                        &ids[3],
                    );
                    executor
                        .state
                        .run(executor.reference(), &invocation)
                        .unwrap()
                        .unwrap()
                };
                let apart = silu_times(product(&ids[1]), &product(&ids[2])).unwrap();
                assert_eq!(actual.shape, apart.shape);
                assert!(
                    actual
                        .data
                        .iter()
                        .zip(&apart.data)
                        .all(|(x, y)| x.to_bits() == y.to_bits()),
                    "{m} row(s), [{k}, {n}]: the fused run's values differ"
                );
            }
        }
    }
}

/// The greedy column scan picks what a one-by-one scan keeping the first
/// strictly higher score picks: ties, non-finite scores, excluded ranges
/// (overlapping, unsorted, past the end), temperatures that overflow.
#[test]
fn greedy_columns_match_a_one_by_one_scan() {
    fn one_by_one(
        scores: &[f32],
        excluded: &[std::ops::Range<u64>],
        temperature: f32,
    ) -> Option<u64> {
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
    let mut rng = Rng(77);
    for case in 0..400 {
        let len = 1 + (rng.next() % 3000) as usize;
        let levels = 1 + rng.next() % 50;
        let scores: Vec<f32> = (0..len)
            .map(|_| match rng.next() % 40 {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => f32::NEG_INFINITY,
                3 => f32::MAX,
                4 => -f32::MAX,
                _ => (rng.next() % levels) as f32 - 20.0,
            })
            .collect();
        let excluded: Vec<std::ops::Range<u64>> = (0..rng.next() % 4)
            .map(|_| {
                let start = u64::from(rng.next()) % (len as u64 + 10);
                start..start + u64::from(rng.next()) % 700
            })
            .collect();
        for temperature in [0.0_f32, 0.7, 1.0, 1e-38, -1.0, f32::NAN] {
            assert_eq!(
                greedy_column(&scores, &excluded, temperature),
                one_by_one(&scores, &excluded, temperature),
                "case {case}, temperature {temperature}"
            );
        }
    }
    assert_eq!(greedy_column(&[1.0, 2.0], &[0..1, 1..5], 0.0), None);
    assert_eq!(greedy_column(&[], &[], 0.0), None);
    assert_eq!(greedy_column(&[f32::NAN, f32::INFINITY], &[], 0.0), None);
}

/// `MAGNETAR_CPU_DECODE_THREADS`: one-row Kernels run on the decode pool,
/// Kernels of several rows on the main pool, both with the Reference
/// values.
#[test]
fn one_row_kernels_run_on_the_decode_pool() {
    let executor = NativeCpuExecutor::with_pools(true, false, 4, Some(2));
    assert_eq!(executor.state.pool_for(1).threads(), 2);
    assert_eq!(executor.state.pool_for(3).threads(), 4);
    assert_eq!(
        NativeCpuExecutor::with_settings(true, false, 4)
            .state
            .pool_for(1)
            .threads(),
        4
    );
    let (k, n) = (256_u64, 640_u64);
    let ids = ["a", "gate", "up", "out"].map(TensorResourceId::new);
    let (gate, up) = (bf16_values(k, n, 311), bf16_values(k, n, 312));
    executor
        .write_weight_tensor(ids[1].clone(), gate.clone())
        .unwrap();
    executor
        .write_weight_tensor(ids[2].clone(), up.clone())
        .unwrap();
    for m in [1_u64, 3] {
        let a = activations(m, k, 313);
        executor.reference().write_tensor(ids[0].clone(), a.clone());
        let product = executor
            .state
            .run(
                executor.reference(),
                &invocation_writing(invocation_reading("matmul", &[&ids[0], &ids[1]]), &ids[3]),
            )
            .expect("handled natively")
            .unwrap();
        assert_close(
            &product,
            &matmul(&a, &gate, false, false).unwrap(),
            &format!("matmul, {m} row(s)"),
        );
        let gated = executor
            .state
            .run(
                executor.reference(),
                &invocation_writing(
                    invocation_reading("gated-matmul", &[&ids[0], &ids[1], &ids[2]]),
                    &ids[3],
                ),
            )
            .expect("handled natively")
            .unwrap();
        let expected = crate::mul(
            &crate::silu(&matmul(&a, &gate, false, false).unwrap()),
            &matmul(&a, &up, false, false).unwrap(),
        )
        .unwrap();
        assert_close(&gated, &expected, &format!("gated-matmul, {m} row(s)"));
    }
}

#[test]
fn cpu_lists_parse_as_linux_writes_them() {
    let list =
        |text| super::topology::cpu_list(text).map(|cpus| cpus.into_iter().collect::<Vec<_>>());
    assert_eq!(list("0-3"), Some(vec![0, 1, 2, 3]));
    assert_eq!(list("0,2"), Some(vec![0, 2]));
    assert_eq!(list("0-1,4,6-7"), Some(vec![0, 1, 4, 6, 7]));
    assert_eq!(list(""), Some(vec![]));
    assert_eq!(list("3-1"), None);
    assert_eq!(list("a"), None);
}

/// One thread per physical core: at least one, never more than the
/// logical CPUs.
#[test]
fn one_thread_per_core_never_exceeds_the_logical_cpus() {
    let logical = std::thread::available_parallelism().map_or(1, |count| count.get());
    let threads = super::topology::one_per_core();
    assert!((1..=logical).contains(&threads), "{threads} of {logical}");
}
