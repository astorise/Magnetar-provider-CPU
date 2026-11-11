//! Packed-weight matmul kernels of the optimized CPU Provider
//! (`execute-quantized-weights-on-cpu`).
//!
//! Every packed format is read as runs of 32 values, the checkpoint's own
//! bytes decoded per run into signed 8-bit values `q` and, per 16-value
//! half, a scale `s` and a min `m` (`value = s * q - m`). A matmul
//! quantizes its activation rows per 32-value block (`d = max |x| / 127`,
//! round to nearest even, as the CUDA Provider does) and accumulates
//! integer half-dot products, then `d (s0 dot0 + s1 dot1 - m0 sum0 - m1
//! sum1)`. Without int8 activations the same runs are applied to the f32
//! activations.

use magnetar_runtime::model_loading::{
    IQ4_NL_VALUES, f16_to_f32, iq_grids, iq_signs, iq4_xs_scales, q3_k_scales,
};
use magnetar_runtime::packed_weight::{PackedWeight, PackedWeightFormat};

/// Values per run.
pub const RUN: usize = 32;

/// A decoded run: its 32 signed values and, per 16-value half, the scale
/// and min that turn them into weight values.
#[derive(Clone, Copy, Debug, Default)]
pub struct Run {
    pub q: [i8; RUN],
    pub scale: [f32; 2],
    pub min: [f32; 2],
}

/// Every `f16` bit pattern's exact `f32` value (256 KiB, built once): a
/// table lookup per block scale instead of the bit-by-bit conversion.
fn half_table() -> &'static [f32] {
    static TABLE: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| (0..=u16::MAX).map(f16_to_f32).collect())
}

#[inline(always)]
fn half(bytes: &[u8], at: usize) -> f32 {
    half_table()[usize::from(u16::from_le_bytes([bytes[at], bytes[at + 1]]))]
}

fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}

/// `ggml-quants.c`'s `get_scale_min_k4`: sub-block `j`'s 6-bit scale and
/// min in a `Q4_K` super-block's twelve scale bytes.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Eight grid values with their signs applied.
fn signed_grid(grid: u64, signs: u8, out: &mut [i8]) {
    for (j, value) in out.iter_mut().enumerate().take(8) {
        let magnitude = ((grid >> (8 * j)) & 0xFF) as i8;
        *value = if (signs >> j) & 1 == 1 {
            -magnitude
        } else {
            magnitude
        };
    }
}

/// Every packed format, indexed by the code the kernels are specialized on.
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

fn format_code(format: PackedWeightFormat) -> u8 {
    FORMATS
        .iter()
        .position(|&known| known == format)
        .expect("every packed format has a code") as u8
}

/// Decodes run `run` (counted from 0 within its block) of one block;
/// inlined into kernels specialized on a constant `format`, so the format
/// dispatch folds away.
#[inline(always)]
fn decode(format: PackedWeightFormat, block: &[u8], run: usize, out: &mut Run) {
    match format {
        PackedWeightFormat::Q8_0 => {
            let d = half(block, 0);
            for (q, &byte) in out.q.iter_mut().zip(&block[2..34]) {
                *q = byte as i8;
            }
            out.scale = [d, d];
            out.min = [0.0, 0.0];
        }
        PackedWeightFormat::Q4K => {
            let d = half(block, 0);
            let dmin = half(block, 2);
            let (sc, m) = scale_min_k4(run, &block[4..16]);
            let chunk = run / 2;
            let shift = 4 * (run % 2);
            for (q, &byte) in out.q.iter_mut().zip(&block[16 + 32 * chunk..][..32]) {
                *q = ((byte >> shift) & 0xF) as i8;
            }
            out.scale = [d * f32::from(sc); 2];
            out.min = [dmin * f32::from(m); 2];
        }
        PackedWeightFormat::Q6K => {
            let d = half(block, 208);
            let h = run / 4;
            let g = run % 4;
            let low = &block[64 * h + 32 * (g % 2)..][..32];
            let high = &block[128 + 32 * h..][..32];
            let (low_shift, high_shift) = (4 * (g / 2), 2 * g);
            for ((q, &low), &high) in out.q.iter_mut().zip(low).zip(high) {
                *q = (((low >> low_shift) & 0xF) | (((high >> high_shift) & 3) << 4)) as i8 - 32;
            }
            let sc = |i: usize| d * f32::from(block[192 + 8 * h + 2 * g + i] as i8);
            out.scale = [sc(0), sc(1)];
            out.min = [0.0, 0.0];
        }
        PackedWeightFormat::Q2K => {
            let d = half(block, 80);
            let dmin = half(block, 82);
            let h = run / 4;
            let j = run % 4;
            for (q, &byte) in out.q.iter_mut().zip(&block[16 + 32 * h..][..32]) {
                *q = ((byte >> (2 * j)) & 3) as i8;
            }
            for p in 0..2 {
                let sc = block[8 * h + 2 * j + p];
                out.scale[p] = d * f32::from(sc & 0xF);
                out.min[p] = dmin * f32::from(sc >> 4);
            }
        }
        PackedWeightFormat::Q3K => {
            let d = half(block, 108);
            let scales = q3_k_scales(&block[96..108]);
            let h = run / 4;
            let j = run % 4;
            let quants = &block[32 + 32 * h..][..32];
            let mask = &block[..32];
            let bit = 4 * h + j;
            for ((q, &byte), &high) in out.q.iter_mut().zip(quants).zip(mask) {
                *q = (((byte >> (2 * j)) & 3) | (((high >> bit) & 1) << 2)) as i8 - 4;
            }
            for p in 0..2 {
                out.scale[p] = d * f32::from(scales[8 * h + 2 * j + p] - 32);
            }
            out.min = [0.0, 0.0];
        }
        PackedWeightFormat::Iq4Nl => {
            let d = half(block, 0);
            for l in 0..16 {
                let byte = block[2 + l];
                out.q[l] = IQ4_NL_VALUES[usize::from(byte & 0xF)];
                out.q[l + 16] = IQ4_NL_VALUES[usize::from(byte >> 4)];
            }
            out.scale = [d, d];
            out.min = [0.0, 0.0];
        }
        PackedWeightFormat::Iq4Xs => {
            let d = half(block, 0);
            let scale = d * f32::from(iq4_xs_scales(&block[2..8])[run]);
            for l in 0..16 {
                let byte = block[8 + 16 * run + l];
                out.q[l] = IQ4_NL_VALUES[usize::from(byte & 0xF)];
                out.q[l + 16] = IQ4_NL_VALUES[usize::from(byte >> 4)];
            }
            out.scale = [scale, scale];
            out.min = [0.0, 0.0];
        }
        PackedWeightFormat::Iq2Xxs => {
            let d = half(block, 0);
            let grids = word(block, 2 + 8 * run);
            let aux = word(block, 6 + 8 * run);
            for l in 0..4 {
                let grid = iq_grids::IQ2XXS_GRID[((grids >> (8 * l)) & 0xFF) as usize];
                signed_grid(grid, iq_signs(aux >> (7 * l)), &mut out.q[8 * l..]);
            }
            let scale = d * (0.5 + (aux >> 28) as f32) * 0.25;
            out.scale = [scale, scale];
            out.min = [0.0, 0.0];
        }
        PackedWeightFormat::Iq2Xs => {
            let d = half(block, 0);
            for l in 0..4 {
                let at = 2 + 2 * (4 * run + l);
                let code = u32::from(u16::from_le_bytes([block[at], block[at + 1]]));
                let grid = iq_grids::IQ2XS_GRID[(code & 511) as usize];
                signed_grid(grid, iq_signs(code >> 9), &mut out.q[8 * l..]);
            }
            for p in 0..2 {
                let sc = (block[66 + run] >> (4 * p)) & 0xF;
                out.scale[p] = d * (0.5 + f32::from(sc)) * 0.25;
            }
            out.min = [0.0, 0.0];
        }
        PackedWeightFormat::Iq2S => {
            let d = half(block, 0);
            for l in 0..4 {
                let index = usize::from(block[2 + 4 * run + l])
                    | ((usize::from(block[66 + run]) >> (2 * l)) & 3) << 8;
                signed_grid(
                    iq_grids::IQ2S_GRID[index],
                    block[34 + 4 * run + l],
                    &mut out.q[8 * l..],
                );
            }
            for p in 0..2 {
                let sc = (block[74 + run] >> (4 * p)) & 0xF;
                out.scale[p] = d * (0.5 + f32::from(sc)) * 0.25;
            }
            out.min = [0.0, 0.0];
        }
        PackedWeightFormat::Iq3Xxs => {
            let d = half(block, 0);
            let aux = word(block, 66 + 4 * run);
            for l in 0..4 {
                let low = iq_grids::IQ3XXS_GRID[usize::from(block[2 + 8 * run + 2 * l])];
                let high = iq_grids::IQ3XXS_GRID[usize::from(block[3 + 8 * run + 2 * l])];
                let grid = u64::from(low) | (u64::from(high) << 32);
                signed_grid(grid, iq_signs(aux >> (7 * l)), &mut out.q[8 * l..]);
            }
            let scale = d * (0.5 + (aux >> 28) as f32) * 0.5;
            out.scale = [scale, scale];
            out.min = [0.0, 0.0];
        }
        PackedWeightFormat::Iq3S => {
            let d = half(block, 0);
            let index = |i: usize| {
                usize::from(block[2 + 8 * run + i]) | ((usize::from(block[66 + run]) >> i) & 1) << 8
            };
            for l in 0..4 {
                let low = iq_grids::IQ3S_GRID[index(2 * l)];
                let high = iq_grids::IQ3S_GRID[index(2 * l + 1)];
                let grid = u64::from(low) | (u64::from(high) << 32);
                signed_grid(grid, block[74 + 4 * run + l], &mut out.q[8 * l..]);
            }
            let sc = (block[106 + run / 2] >> (4 * (run % 2))) & 0xF;
            let scale = d * f32::from(1 + 2 * sc);
            out.scale = [scale, scale];
            out.min = [0.0, 0.0];
        }
    }
}

/// A packed weight read as runs: stored row `r` (one output column of a
/// projection) holds `stored_cols / 32` runs.
pub struct RunReader<'a> {
    format: PackedWeightFormat,
    code: u8,
    bytes: &'a [u8],
    row_bytes: usize,
    block_bytes: usize,
    runs_per_block: usize,
}

impl<'a> RunReader<'a> {
    pub fn new(weight: &'a PackedWeight) -> Self {
        let format = weight.format();
        let block_elements = format.block_elements() as usize;
        let block_bytes = format.block_bytes() as usize;
        Self {
            format,
            code: format_code(format),
            bytes: weight.bytes(),
            row_bytes: weight.stored_cols() as usize / block_elements * block_bytes,
            block_bytes,
            runs_per_block: block_elements / RUN,
        }
    }

    /// Decodes run `run` of stored row `row`.
    pub fn read(&self, row: usize, run: usize, out: &mut Run) {
        let block = run / self.runs_per_block;
        let start = row * self.row_bytes + block * self.block_bytes;
        decode(
            self.format,
            &self.bytes[start..start + self.block_bytes],
            run % self.runs_per_block,
            out,
        );
    }

    /// The block holding run `run` of stored row `row`, and the run's
    /// index within it.
    #[inline(always)]
    fn block(&self, row: usize, run: usize) -> (&[u8], usize) {
        let block = run / self.runs_per_block;
        let start = row * self.row_bytes + block * self.block_bytes;
        (
            &self.bytes[start..start + self.block_bytes],
            run % self.runs_per_block,
        )
    }

    /// [`Self::read`] for a reader of format code `F`.
    #[inline(always)]
    fn read_as<const F: u8>(&self, row: usize, run: usize, out: &mut Run) {
        let (block, run) = self.block(row, run);
        decode(FORMATS[F as usize], block, run, out);
    }
}

/// Activation rows quantized per 32-value block: values, one scale per
/// block and each 16-value half's sum.
pub struct QuantizedRows {
    pub values: Vec<i8>,
    pub scales: Vec<f32>,
    pub sums: Vec<[i32; 2]>,
}

/// Quantizes `rows` (row-major, each `k` long, `k` a multiple of 32) as
/// the CUDA Provider's `quantize_activations_q8_kernel` does.
pub fn quantize_rows(rows: &[f32]) -> QuantizedRows {
    let blocks = rows.len() / RUN;
    let mut quantized = QuantizedRows {
        values: vec![0; rows.len()],
        scales: Vec::with_capacity(blocks),
        sums: Vec::with_capacity(blocks),
    };
    for (block, out) in rows
        .as_chunks::<RUN>()
        .0
        .iter()
        .zip(quantized.values.as_chunks_mut::<RUN>().0.iter_mut())
    {
        let amax = block.iter().fold(0.0_f32, |max, x| max.max(x.abs()));
        let d = amax / 127.0;
        let mut sums = [0_i32; 2];
        for (i, (x, q)) in block.iter().zip(out.iter_mut()).enumerate() {
            *q = if amax == 0.0 {
                0
            } else {
                (x / d).round_ties_even() as i8
            };
            sums[i / 16] += i32::from(*q);
        }
        quantized.scales.push(d);
        quantized.sums.push(sums);
    }
    quantized
}

/// The two 16-value integer dot products of a run with its activations
/// (the scalar oracle).
pub fn dot_halves_scalar(q: &[i8; RUN], a: &[i8]) -> [i32; 2] {
    let mut dots = [0_i32; 2];
    for (i, (&w, &x)) in q.iter().zip(a).enumerate() {
        dots[i / 16] += i32::from(w) * i32::from(x);
    }
    dots
}

/// [`dot_halves_scalar`] with AVX2: both operands widened to 16 bits (exact for
/// any signed byte, including -128) and multiplied pairwise with
/// `vpmaddwd`.
#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn dot_halves_avx2(q: &[i8; RUN], a: &[i8]) -> [i32; 2] {
    use std::arch::x86_64::*;
    debug_assert!(a.len() >= RUN);
    // SAFETY: both slices hold at least 32 bytes; unaligned loads.
    unsafe {
        let w = _mm256_loadu_si256(q.as_ptr().cast());
        let x = _mm256_loadu_si256(a.as_ptr().cast());
        let w0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w));
        let w1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1));
        let x0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x));
        let x1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x, 1));
        let p0 = _mm256_madd_epi16(w0, x0);
        let p1 = _mm256_madd_epi16(w1, x1);
        let sum = |v: __m256i| {
            let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
            _mm_cvtsi128_si32(s)
        };
        [sum(p0), sum(p1)]
    }
}

/// A matmul's activations: `m` rows of `k` f32 values and, with int8
/// activations, their quantized form.
pub struct Activations<'a> {
    pub rows: &'a [f32],
    pub quantized: Option<&'a QuantizedRows>,
    pub m: usize,
    pub k: usize,
}

/// `out[r][c - first]` for columns `first..first + count` of `a x W`,
/// `W` packed with its stored rows the output columns: int8 activations
/// when quantized, else the f32 rows.
pub fn matmul_columns(
    reader: &RunReader<'_>,
    activations: &Activations<'_>,
    first: usize,
    count: usize,
    out: &mut [f32],
) {
    macro_rules! dispatch {
        ($($code:literal),*) => {
            match reader.code {
                $($code => columns::<$code>(reader, activations, first, count, out),)*
                _ => unreachable!("format codes are below 12"),
            }
        };
    }
    dispatch!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
}

/// [`matmul_columns`] for format code `F`.
fn columns<const F: u8>(
    reader: &RunReader<'_>,
    activations: &Activations<'_>,
    first: usize,
    count: usize,
    out: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    if let Some(quantized) = activations.quantized
        && super::simd_level() >= super::SimdLevel::Avx2
    {
        // SAFETY: AVX2 and FMA were detected at run time (`simd_level`).
        unsafe {
            columns_avx2::<F>(
                reader,
                quantized,
                activations.m,
                activations.k,
                first,
                count,
                out,
            )
        };
        return;
    }
    let Activations {
        rows: a,
        quantized,
        m,
        k,
    } = *activations;
    let runs = k / RUN;
    let mut run = Run::default();
    let mut acc = vec![0.0_f32; m];
    for c in 0..count {
        acc.iter_mut().for_each(|value| *value = 0.0);
        for u in 0..runs {
            reader.read_as::<F>(first + c, u, &mut run);
            for (r, value) in acc.iter_mut().enumerate() {
                let block = r * runs + u;
                match quantized {
                    Some(quantized) => {
                        let dots = dot_halves_scalar(&run.q, &quantized.values[block * RUN..]);
                        let sums = quantized.sums[block];
                        let d = quantized.scales[block];
                        *value += d
                            * (run.scale[0] * dots[0] as f32 + run.scale[1] * dots[1] as f32
                                - run.min[0] * sums[0] as f32
                                - run.min[1] * sums[1] as f32);
                    }
                    None => {
                        let x = &a[block * RUN..block * RUN + RUN];
                        for (i, (&q, &x)) in run.q.iter().zip(x).enumerate() {
                            let p = i / 16;
                            *value += (run.scale[p] * f32::from(q) - run.min[p]) * x;
                        }
                    }
                }
            }
        }
        for (r, value) in acc.iter().enumerate() {
            out[r * count + c] = *value;
        }
    }
}

/// [`columns`] with int8 activations on AVX2 and FMA: each run's values are
/// widened once and multiplied with every row's activations by
/// `vpmaddwd`; the products stay in eight f32 lanes per row, scaled by the
/// run's half scales, and are summed horizontally once per column.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn columns_avx2<const F: u8>(
    reader: &RunReader<'_>,
    quantized: &QuantizedRows,
    m: usize,
    k: usize,
    first: usize,
    count: usize,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;
    let runs = k / RUN;
    let mut run = Run::default();
    let mut acc = vec![_mm256_setzero_ps(); m];
    let mut mins = vec![0.0_f32; m];
    for c in 0..count {
        acc.iter_mut()
            .for_each(|lanes| *lanes = _mm256_setzero_ps());
        mins.iter_mut().for_each(|value| *value = 0.0);
        for u in 0..runs {
            // SAFETY: AVX2 is enabled for this function.
            let w = unsafe { vector_run::<F>(reader, first + c, u, &mut run) };
            let has_min = run.min[0] != 0.0 || run.min[1] != 0.0;
            let w0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w));
            let w1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1));
            for r in 0..m {
                let block = r * runs + u;
                // SAFETY: each quantized block holds 32 values.
                let x = unsafe {
                    _mm256_loadu_si256(quantized.values.as_ptr().add(block * RUN).cast())
                };
                let x0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x));
                let x1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x, 1));
                let d = quantized.scales[block];
                let p0 = _mm256_cvtepi32_ps(_mm256_madd_epi16(w0, x0));
                let p1 = _mm256_cvtepi32_ps(_mm256_madd_epi16(w1, x1));
                acc[r] = _mm256_fmadd_ps(p0, _mm256_set1_ps(d * run.scale[0]), acc[r]);
                acc[r] = _mm256_fmadd_ps(p1, _mm256_set1_ps(d * run.scale[1]), acc[r]);
                if has_min {
                    let sums = quantized.sums[block];
                    mins[r] += d * (run.min[0] * sums[0] as f32 + run.min[1] * sums[1] as f32);
                }
            }
        }
        for r in 0..m {
            let lanes = acc[r];
            let half = _mm_add_ps(
                _mm256_castps256_ps128(lanes),
                _mm256_extractf128_ps(lanes, 1),
            );
            let half = _mm_add_ps(half, _mm_movehl_ps(half, half));
            let half = _mm_add_ss(half, _mm_shuffle_ps(half, half, 0b01));
            out[r * count + c] = _mm_cvtss_f32(half) - mins[r];
        }
    }
}

/// A run's 32 signed values as one AVX2 register, its scales and mins in
/// `run`: decoded straight from the block bytes for `Q8_0` (a load),
/// `Q4_K` (a shift and a mask) and the IQ4 formats (`vpshufb` through the
/// non-linear table), through [`decode`] for the others.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn vector_run<const F: u8>(
    reader: &RunReader<'_>,
    row: usize,
    index: usize,
    run: &mut Run,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let (block, local) = reader.block(row, index);
    let nibbles = |bytes: __m128i| {
        let low = _mm_and_si128(bytes, _mm_set1_epi8(0xF));
        let high = _mm_and_si128(_mm_srli_epi16(bytes, 4), _mm_set1_epi8(0xF));
        _mm256_set_m128i(high, low)
    };
    let iq4 = |indices: __m256i| {
        // SAFETY: the table holds 16 bytes.
        let table = unsafe { _mm_loadu_si128(IQ4_NL_VALUES.as_ptr().cast()) };
        _mm256_shuffle_epi8(_mm256_set_m128i(table, table), indices)
    };
    match FORMATS[F as usize] {
        PackedWeightFormat::Q8_0 => {
            let d = half(block, 0);
            run.scale = [d, d];
            run.min = [0.0, 0.0];
            // SAFETY: a Q8_0 block holds 32 quants after its scale.
            unsafe { _mm256_loadu_si256(block.as_ptr().add(2).cast()) }
        }
        PackedWeightFormat::Q4K => {
            let d = half(block, 0);
            let dmin = half(block, 2);
            let (sc, m) = scale_min_k4(local, &block[4..16]);
            run.scale = [d * f32::from(sc); 2];
            run.min = [dmin * f32::from(m); 2];
            // SAFETY: each Q4_K chunk holds 32 quant bytes.
            let bytes =
                unsafe { _mm256_loadu_si256(block.as_ptr().add(16 + 32 * (local / 2)).cast()) };
            let bytes = if local % 2 == 0 {
                bytes
            } else {
                _mm256_srli_epi16(bytes, 4)
            };
            _mm256_and_si256(bytes, _mm256_set1_epi8(0xF))
        }
        PackedWeightFormat::Iq4Nl => {
            let d = half(block, 0);
            run.scale = [d, d];
            run.min = [0.0, 0.0];
            // SAFETY: an IQ4_NL block holds 16 index bytes after its scale.
            iq4(nibbles(unsafe {
                _mm_loadu_si128(block.as_ptr().add(2).cast())
            }))
        }
        PackedWeightFormat::Iq4Xs => {
            let scale = half(block, 0) * f32::from(iq4_xs_scales(&block[2..8])[local]);
            run.scale = [scale, scale];
            run.min = [0.0, 0.0];
            // SAFETY: each IQ4_XS run holds 16 index bytes.
            iq4(nibbles(unsafe {
                _mm_loadu_si128(block.as_ptr().add(8 + 16 * local).cast())
            }))
        }
        format => {
            decode(format, block, local, run);
            // SAFETY: `run.q` holds 32 bytes.
            unsafe { _mm256_loadu_si256(run.q.as_ptr().cast()) }
        }
    }
}

/// `out[r][c - 32 first_run]` for the output columns of runs
/// `first_run..first_run + run_count` of `a x W`, `W` packed with its stored
/// rows the *input* rows (an untransposed `[k, n]` weight, or a tied
/// embedding table read transposed): each stored row `j` adds `a[r][j]`
/// times its decoded values. f32 activations; every weight value is
/// decoded once.
pub fn axpy_columns(
    reader: &RunReader<'_>,
    a: &[f32],
    m: usize,
    k: usize,
    first_run: usize,
    run_count: usize,
    out: &mut [f32],
) {
    let width = run_count * RUN;
    let mut run = Run::default();
    let mut values = [0.0_f32; RUN];
    for j in 0..k {
        for u in 0..run_count {
            reader.read(j, first_run + u, &mut run);
            for (i, value) in values.iter_mut().enumerate() {
                let p = i / 16;
                *value = run.scale[p] * f32::from(run.q[i]) - run.min[p];
            }
            for r in 0..m {
                let weight = a[r * k + j];
                if weight == 0.0 {
                    continue;
                }
                let row = &mut out[r * width + u * RUN..r * width + u * RUN + RUN];
                for (o, &value) in row.iter_mut().zip(&values) {
                    *o += weight * value;
                }
            }
        }
    }
}

/// Element `index` of stored row `row`.
pub fn element(reader: &RunReader<'_>, row: usize, index: usize) -> f32 {
    let mut run = Run::default();
    reader.read(row, index / RUN, &mut run);
    let i = index % RUN;
    let p = i / 16;
    run.scale[p] * f32::from(run.q[i]) - run.min[p]
}

/// Stored row `row` of a packed weight as f32 values (an embedding row of
/// an untransposed table).
pub fn dequantize_row(reader: &RunReader<'_>, row: usize, cols: usize, out: &mut [f32]) {
    let mut run = Run::default();
    for u in 0..cols / RUN {
        reader.read(row, u, &mut run);
        for (i, value) in out[u * RUN..u * RUN + RUN].iter_mut().enumerate() {
            let p = i / 16;
            *value = run.scale[p] * f32::from(run.q[i]) - run.min[p];
        }
    }
}
