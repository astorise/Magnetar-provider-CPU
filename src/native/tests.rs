//! Tests of the optimized CPU Provider against the Reference CPU kernels.

use super::*;
use magnetar_runtime::packed_weight::PackedWeightFormat;

const FORMATS: [PackedWeightFormat; 12] = [
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
];

/// Byte offsets of each block's `f16` scale fields.
fn half_offsets(format: PackedWeightFormat) -> &'static [usize] {
    match format {
        PackedWeightFormat::Q4K => &[0, 2],
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
        pool: pool::Pool::new(threads),
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
    for format in [PackedWeightFormat::Q8_0, PackedWeightFormat::Iq4Nl] {
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
        PackedWeightFormat::Iq4Nl,
        PackedWeightFormat::Iq2Xs,
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
    use magnetar_runtime::kernel::{KernelInvocationId, KernelResource};
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
