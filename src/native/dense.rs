//! Dense matmuls of the optimized CPU Provider (`speed-up-dense-cpu-matmuls`):
//! `a x W` for an f32 activation matrix `a` (`m x k`) and a dense weight
//! `W` held as f32 values or as compact bfloat16 bits (a weight whose every
//! value is exactly a bfloat16 value, as every weight of a bfloat16
//! checkpoint is: half the bytes, every value unchanged).
//!
//! Each kernel computes a range of output columns, so the pool shares a
//! matmul out by column. Products are accumulated in f32 with fused
//! multiply-adds: values agree with the Reference CPU kernel's to rounding
//! (the summation order differs), not bit for bit.

use super::SimdLevel;

/// A dense weight's values, in their stored order.
#[derive(Clone, Copy)]
pub(crate) enum Values<'a> {
    F32(&'a [f32]),
    /// The upper 16 bits of each f32 value.
    Bf16(&'a [u16]),
}

impl Values<'_> {
    #[inline]
    fn get(&self, index: usize) -> f32 {
        match self {
            Values::F32(values) => values[index],
            Values::Bf16(bits) => f32::from_bits(u32::from(bits[index]) << 16),
        }
    }
}

/// Whether the dense kernels of several rows (a prefill, compute-bound)
/// use AVX-512F (sixteen lanes) where the host has it, rather than the
/// AVX2 baseline; `MAGNETAR_CPU_SIMD=avx2` keeps them on AVX2. A single
/// row (a decode step) is bound by memory bandwidth and stays on AVX2:
/// locally, the wider kernel was slower there (8.2 against 6.7 ms for the
/// LM head) and faster with 8 rows (12.2 against 21.1 ms).
#[cfg(target_arch = "x86_64")]
pub(crate) fn avx512() -> bool {
    static WIDE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *WIDE.get_or_init(|| {
        std::env::var(super::SIMD_ENV).as_deref() != Ok("avx2")
            && std::is_x86_feature_detected!("avx512f")
    })
}

/// The bfloat16 bits of `value`, when converting them back gives `value`.
pub(crate) fn exact_bf16_bits(value: f32) -> Option<u16> {
    let bits = value.to_bits();
    (bits & 0xffff == 0).then_some((bits >> 16) as u16)
}

/// `out[r * count + c] = sum_i a[r * k + i] * W[first + c][i]` for a weight
/// stored output-major (`n` rows of `k` values: a `[k, n]` weight used
/// transposed), `out` holding `m x count` values.
#[allow(clippy::too_many_arguments)]
pub(crate) fn output_major(
    level: SimdLevel,
    a: &[f32],
    m: usize,
    k: usize,
    weight: Values<'_>,
    first: usize,
    count: usize,
    out: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    if level == SimdLevel::Avx2 {
        // SAFETY: `SimdLevel::Avx2` is only reported when AVX2 and FMA are
        // available, and `avx512` only when AVX-512F is.
        unsafe {
            if m > 1 && avx512() {
                avx512::output_major(a, m, k, weight, first, count, out);
            } else {
                avx2::output_major(a, m, k, weight, first, count, out);
            }
        }
        return;
    }
    let _ = level;
    for c in 0..count {
        let row = (first + c) * k;
        for r in 0..m {
            let a_row = &a[r * k..(r + 1) * k];
            let mut lanes = [0.0_f32; 8];
            let mut i = 0;
            while i + 8 <= k {
                for lane in 0..8 {
                    lanes[lane] = a_row[i + lane].mul_add(weight.get(row + i + lane), lanes[lane]);
                }
                i += 8;
            }
            let mut sum: f32 = lanes.iter().sum();
            while i < k {
                sum = a_row[i].mul_add(weight.get(row + i), sum);
                i += 1;
            }
            out[r * count + c] = sum;
        }
    }
}

/// `out[r * count + c] = sum_{i in rows} a[r * k + i] * W[i][first + c]`
/// for a weight stored input-major (`k` rows of `n` values), `out` holding
/// `m x count` values: the whole product for `rows` `0..k`, a partial sum
/// over a block of rows otherwise.
#[allow(clippy::too_many_arguments)]
pub(crate) fn input_major(
    level: SimdLevel,
    a: &[f32],
    m: usize,
    k: usize,
    rows: std::ops::Range<usize>,
    weight: Values<'_>,
    n: usize,
    first: usize,
    count: usize,
    out: &mut [f32],
) {
    out[..m * count].iter_mut().for_each(|value| *value = 0.0);
    #[cfg(target_arch = "x86_64")]
    if level == SimdLevel::Avx2 {
        // SAFETY: as in `output_major`.
        unsafe {
            if m > 1 && avx512() {
                avx512::input_major(a, m, k, rows, weight, n, first, count, out);
            } else {
                avx2::input_major(a, m, k, rows, weight, n, first, count, out);
            }
        }
        return;
    }
    let _ = level;
    for i in rows {
        let row = i * n + first;
        for r in 0..m {
            let scale = a[r * k + i];
            let out_row = &mut out[r * count..(r + 1) * count];
            for (c, value) in out_row.iter_mut().enumerate() {
                *value = scale.mul_add(weight.get(row + c), *value);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(in crate::native) mod avx2 {
    use super::Values;
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn sum8(v: __m256) -> f32 {
        let pair = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let quad = _mm_add_ps(pair, _mm_movehl_ps(pair, pair));
        let single = _mm_add_ss(quad, _mm_shuffle_ps(quad, quad, 1));
        _mm_cvtss_f32(single)
    }

    /// Eight f32 weight values from `index`.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn load_f32(values: &[f32], index: usize) -> __m256 {
        unsafe { _mm256_loadu_ps(values.as_ptr().add(index)) }
    }

    /// Eight bfloat16 weight values from `index`, as f32.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn load_bf16(bits: &[u16], index: usize) -> __m256 {
        unsafe {
            let halves = _mm_loadu_si128(bits.as_ptr().add(index) as *const __m128i);
            _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(halves), 16))
        }
    }

    /// The kernels for one weight encoding, each a loop with one load
    /// function (no per-load dispatch).
    macro_rules! kernels {
        ($output_major:ident, $input_major:ident, $element:ty, $load:ident, $get:expr) => {
            /// Rows of `a` taken four at a time (one with four independent
            /// accumulators): each weight vector loaded once serves them.
            #[target_feature(enable = "avx2,fma")]
            #[allow(clippy::too_many_arguments)]
            unsafe fn $output_major(
                a: &[f32],
                m: usize,
                k: usize,
                weight: &[$element],
                first: usize,
                count: usize,
                out: &mut [f32],
            ) {
                let get = $get;
                let whole = k - k % 8;
                if m == 1 {
                    let wide = k - k % 32;
                    for c in 0..count {
                        let row = (first + c) * k;
                        let mut acc = [_mm256_setzero_ps(); 4];
                        let mut i = 0;
                        while i < wide {
                            for (j, acc) in acc.iter_mut().enumerate() {
                                let at = i + 8 * j;
                                unsafe {
                                    let w = $load(weight, row + at);
                                    let x = _mm256_loadu_ps(a.as_ptr().add(at));
                                    *acc = _mm256_fmadd_ps(x, w, *acc);
                                }
                            }
                            i += 32;
                        }
                        while i < whole {
                            unsafe {
                                let w = $load(weight, row + i);
                                let x = _mm256_loadu_ps(a.as_ptr().add(i));
                                acc[0] = _mm256_fmadd_ps(x, w, acc[0]);
                            }
                            i += 8;
                        }
                        let pair = _mm256_add_ps(
                            _mm256_add_ps(acc[0], acc[1]),
                            _mm256_add_ps(acc[2], acc[3]),
                        );
                        let mut sum = sum8(pair);
                        for t in whole..k {
                            sum = a[t].mul_add(get(weight, row + t), sum);
                        }
                        out[c] = sum;
                    }
                    return;
                }
                for c in 0..count {
                    let row = (first + c) * k;
                    let mut r = 0;
                    while r < m {
                        let rows = (m - r).min(4);
                        let mut acc = [_mm256_setzero_ps(); 4];
                        let mut i = 0;
                        while i < whole {
                            let w = unsafe { $load(weight, row + i) };
                            for (j, acc) in acc.iter_mut().enumerate().take(rows) {
                                let x = unsafe { _mm256_loadu_ps(a.as_ptr().add((r + j) * k + i)) };
                                *acc = _mm256_fmadd_ps(x, w, *acc);
                            }
                            i += 8;
                        }
                        for (j, acc) in acc.iter().enumerate().take(rows) {
                            let mut sum = sum8(*acc);
                            for t in whole..k {
                                sum = a[(r + j) * k + t].mul_add(get(weight, row + t), sum);
                            }
                            out[(r + j) * count + c] = sum;
                        }
                        r += rows;
                    }
                }
            }

            #[target_feature(enable = "avx2,fma")]
            #[allow(clippy::too_many_arguments)]
            unsafe fn $input_major(
                a: &[f32],
                m: usize,
                k: usize,
                rows: std::ops::Range<usize>,
                weight: &[$element],
                n: usize,
                first: usize,
                count: usize,
                out: &mut [f32],
            ) {
                let get = $get;
                let whole = count - count % 8;
                // Rows four at a time: one read and write of the output
                // block serves four weight rows.
                let mut i = rows.start;
                while i + 4 <= rows.end {
                    let w = [
                        i * n + first,
                        (i + 1) * n + first,
                        (i + 2) * n + first,
                        (i + 3) * n + first,
                    ];
                    if m == 1 {
                        // One row (a decode step): its four scales once, then
                        // each block of eight columns read and written once.
                        let x = &a[i..i + 4];
                        let s = [
                            _mm256_set1_ps(x[0]),
                            _mm256_set1_ps(x[1]),
                            _mm256_set1_ps(x[2]),
                            _mm256_set1_ps(x[3]),
                        ];
                        let mut c = 0;
                        while c < whole {
                            unsafe {
                                let at = out.as_mut_ptr().add(c);
                                let mut acc = _mm256_loadu_ps(at);
                                for j in 0..4 {
                                    acc = _mm256_fmadd_ps(s[j], $load(weight, w[j] + c), acc);
                                }
                                _mm256_storeu_ps(at, acc);
                            }
                            c += 8;
                        }
                    } else {
                        // Several rows (a prefill): each block of eight
                        // columns of the four weight rows is loaded (and
                        // widened) once and serves every row of `a`.
                        let mut c = 0;
                        while c < whole {
                            let wv = unsafe {
                                [
                                    $load(weight, w[0] + c),
                                    $load(weight, w[1] + c),
                                    $load(weight, w[2] + c),
                                    $load(weight, w[3] + c),
                                ]
                            };
                            for r in 0..m {
                                unsafe {
                                    let x = a.as_ptr().add(r * k + i);
                                    let at = out.as_mut_ptr().add(r * count + c);
                                    let mut acc = _mm256_loadu_ps(at);
                                    for j in 0..4 {
                                        acc = _mm256_fmadd_ps(
                                            _mm256_broadcast_ss(&*x.add(j)),
                                            wv[j],
                                            acc,
                                        );
                                    }
                                    _mm256_storeu_ps(at, acc);
                                }
                            }
                            c += 8;
                        }
                    }
                    for r in 0..m {
                        let x = &a[r * k + i..r * k + i + 4];
                        let out_row = r * count;
                        for c in whole..count {
                            let mut value = out[out_row + c];
                            for j in 0..4 {
                                value = x[j].mul_add(get(weight, w[j] + c), value);
                            }
                            out[out_row + c] = value;
                        }
                    }
                    i += 4;
                }
                for i in i..rows.end {
                    let row = i * n + first;
                    for r in 0..m {
                        let scale = a[r * k + i];
                        let s = _mm256_set1_ps(scale);
                        let out_row = r * count;
                        let mut c = 0;
                        while c < whole {
                            unsafe {
                                let w = $load(weight, row + c);
                                let at = out.as_mut_ptr().add(out_row + c);
                                _mm256_storeu_ps(at, _mm256_fmadd_ps(s, w, _mm256_loadu_ps(at)));
                            }
                            c += 8;
                        }
                        for c in whole..count {
                            out[out_row + c] =
                                scale.mul_add(get(weight, row + c), out[out_row + c]);
                        }
                    }
                }
            }
        };
    }

    kernels!(
        output_major_f32,
        input_major_f32,
        f32,
        load_f32,
        |values: &[f32], index: usize| values[index]
    );
    kernels!(
        output_major_bf16,
        input_major_bf16,
        u16,
        load_bf16,
        |bits: &[u16], index: usize| f32::from_bits(u32::from(bits[index]) << 16)
    );

    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::native) unsafe fn output_major(
        a: &[f32],
        m: usize,
        k: usize,
        weight: Values<'_>,
        first: usize,
        count: usize,
        out: &mut [f32],
    ) {
        unsafe {
            match weight {
                Values::F32(values) => output_major_f32(a, m, k, values, first, count, out),
                Values::Bf16(bits) => output_major_bf16(a, m, k, bits, first, count, out),
            }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::native) unsafe fn input_major(
        a: &[f32],
        m: usize,
        k: usize,
        rows: std::ops::Range<usize>,
        weight: Values<'_>,
        n: usize,
        first: usize,
        count: usize,
        out: &mut [f32],
    ) {
        unsafe {
            match weight {
                Values::F32(values) => input_major_f32(a, m, k, rows, values, n, first, count, out),
                Values::Bf16(bits) => input_major_bf16(a, m, k, rows, bits, n, first, count, out),
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(in crate::native) mod avx512 {
    use super::Values;
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx512f")]
    fn sum16(v: __m512) -> f32 {
        _mm512_reduce_add_ps(v)
    }

    /// Sixteen f32 weight values from `index`.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn load_f32(values: &[f32], index: usize) -> __m512 {
        unsafe { _mm512_loadu_ps(values.as_ptr().add(index)) }
    }

    /// Sixteen bfloat16 weight values from `index`, as f32.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn load_bf16(bits: &[u16], index: usize) -> __m512 {
        unsafe {
            let halves = _mm256_loadu_si256(bits.as_ptr().add(index) as *const __m256i);
            _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(halves)))
        }
    }

    /// The kernels for one weight encoding, each a loop with one load
    /// function (no per-load dispatch).
    macro_rules! kernels {
        ($output_major:ident, $input_major:ident, $element:ty, $load:ident, $get:expr) => {
            /// Rows of `a` taken four at a time (one with four independent
            /// accumulators): each weight vector loaded once serves them.
            #[target_feature(enable = "avx512f")]
            #[allow(clippy::too_many_arguments)]
            unsafe fn $output_major(
                a: &[f32],
                m: usize,
                k: usize,
                weight: &[$element],
                first: usize,
                count: usize,
                out: &mut [f32],
            ) {
                let get = $get;
                let whole = k - k % 16;
                if m == 1 {
                    let wide = k - k % 64;
                    for c in 0..count {
                        let row = (first + c) * k;
                        let mut acc = [_mm512_setzero_ps(); 4];
                        let mut i = 0;
                        while i < wide {
                            for (j, acc) in acc.iter_mut().enumerate() {
                                let at = i + 16 * j;
                                unsafe {
                                    let w = $load(weight, row + at);
                                    let x = _mm512_loadu_ps(a.as_ptr().add(at));
                                    *acc = _mm512_fmadd_ps(x, w, *acc);
                                }
                            }
                            i += 64;
                        }
                        while i < whole {
                            unsafe {
                                let w = $load(weight, row + i);
                                let x = _mm512_loadu_ps(a.as_ptr().add(i));
                                acc[0] = _mm512_fmadd_ps(x, w, acc[0]);
                            }
                            i += 16;
                        }
                        let pair = _mm512_add_ps(
                            _mm512_add_ps(acc[0], acc[1]),
                            _mm512_add_ps(acc[2], acc[3]),
                        );
                        let mut sum = sum16(pair);
                        for t in whole..k {
                            sum = a[t].mul_add(get(weight, row + t), sum);
                        }
                        out[c] = sum;
                    }
                    return;
                }
                for c in 0..count {
                    let row = (first + c) * k;
                    let mut r = 0;
                    while r < m {
                        let rows = (m - r).min(4);
                        let mut acc = [_mm512_setzero_ps(); 4];
                        let mut i = 0;
                        while i < whole {
                            let w = unsafe { $load(weight, row + i) };
                            for (j, acc) in acc.iter_mut().enumerate().take(rows) {
                                let x = unsafe { _mm512_loadu_ps(a.as_ptr().add((r + j) * k + i)) };
                                *acc = _mm512_fmadd_ps(x, w, *acc);
                            }
                            i += 16;
                        }
                        for (j, acc) in acc.iter().enumerate().take(rows) {
                            let mut sum = sum16(*acc);
                            for t in whole..k {
                                sum = a[(r + j) * k + t].mul_add(get(weight, row + t), sum);
                            }
                            out[(r + j) * count + c] = sum;
                        }
                        r += rows;
                    }
                }
            }

            #[target_feature(enable = "avx512f")]
            #[allow(clippy::too_many_arguments)]
            unsafe fn $input_major(
                a: &[f32],
                m: usize,
                k: usize,
                rows: std::ops::Range<usize>,
                weight: &[$element],
                n: usize,
                first: usize,
                count: usize,
                out: &mut [f32],
            ) {
                let get = $get;
                let whole = count - count % 16;
                // Rows four at a time: one read and write of the output
                // block serves four weight rows.
                let mut i = rows.start;
                while i + 4 <= rows.end {
                    let w = [
                        i * n + first,
                        (i + 1) * n + first,
                        (i + 2) * n + first,
                        (i + 3) * n + first,
                    ];
                    if m == 1 {
                        // One row (a decode step): its four scales once, then
                        // each block of sixteen columns read and written once.
                        let x = &a[i..i + 4];
                        let s = [
                            _mm512_set1_ps(x[0]),
                            _mm512_set1_ps(x[1]),
                            _mm512_set1_ps(x[2]),
                            _mm512_set1_ps(x[3]),
                        ];
                        let mut c = 0;
                        while c < whole {
                            unsafe {
                                let at = out.as_mut_ptr().add(c);
                                let mut acc = _mm512_loadu_ps(at);
                                for j in 0..4 {
                                    acc = _mm512_fmadd_ps(s[j], $load(weight, w[j] + c), acc);
                                }
                                _mm512_storeu_ps(at, acc);
                            }
                            c += 16;
                        }
                    } else {
                        // Several rows (a prefill): each block of sixteen
                        // columns of the four weight rows is loaded (and
                        // widened) once and serves every row of `a`.
                        let mut c = 0;
                        while c < whole {
                            let wv = unsafe {
                                [
                                    $load(weight, w[0] + c),
                                    $load(weight, w[1] + c),
                                    $load(weight, w[2] + c),
                                    $load(weight, w[3] + c),
                                ]
                            };
                            for r in 0..m {
                                unsafe {
                                    let x = a.as_ptr().add(r * k + i);
                                    let at = out.as_mut_ptr().add(r * count + c);
                                    let mut acc = _mm512_loadu_ps(at);
                                    for j in 0..4 {
                                        acc =
                                            _mm512_fmadd_ps(_mm512_set1_ps(*x.add(j)), wv[j], acc);
                                    }
                                    _mm512_storeu_ps(at, acc);
                                }
                            }
                            c += 16;
                        }
                    }
                    for r in 0..m {
                        let x = &a[r * k + i..r * k + i + 4];
                        let out_row = r * count;
                        for c in whole..count {
                            let mut value = out[out_row + c];
                            for j in 0..4 {
                                value = x[j].mul_add(get(weight, w[j] + c), value);
                            }
                            out[out_row + c] = value;
                        }
                    }
                    i += 4;
                }
                for i in i..rows.end {
                    let row = i * n + first;
                    for r in 0..m {
                        let scale = a[r * k + i];
                        let s = _mm512_set1_ps(scale);
                        let out_row = r * count;
                        let mut c = 0;
                        while c < whole {
                            unsafe {
                                let w = $load(weight, row + c);
                                let at = out.as_mut_ptr().add(out_row + c);
                                _mm512_storeu_ps(at, _mm512_fmadd_ps(s, w, _mm512_loadu_ps(at)));
                            }
                            c += 16;
                        }
                        for c in whole..count {
                            out[out_row + c] =
                                scale.mul_add(get(weight, row + c), out[out_row + c]);
                        }
                    }
                }
            }
        };
    }

    kernels!(
        output_major_f32,
        input_major_f32,
        f32,
        load_f32,
        |values: &[f32], index: usize| values[index]
    );
    kernels!(
        output_major_bf16,
        input_major_bf16,
        u16,
        load_bf16,
        |bits: &[u16], index: usize| f32::from_bits(u32::from(bits[index]) << 16)
    );

    #[target_feature(enable = "avx512f")]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::native) unsafe fn output_major(
        a: &[f32],
        m: usize,
        k: usize,
        weight: Values<'_>,
        first: usize,
        count: usize,
        out: &mut [f32],
    ) {
        unsafe {
            match weight {
                Values::F32(values) => output_major_f32(a, m, k, values, first, count, out),
                Values::Bf16(bits) => output_major_bf16(a, m, k, bits, first, count, out),
            }
        }
    }

    #[target_feature(enable = "avx512f")]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::native) unsafe fn input_major(
        a: &[f32],
        m: usize,
        k: usize,
        rows: std::ops::Range<usize>,
        weight: Values<'_>,
        n: usize,
        first: usize,
        count: usize,
        out: &mut [f32],
    ) {
        unsafe {
            match weight {
                Values::F32(values) => input_major_f32(a, m, k, rows, values, n, first, count, out),
                Values::Bf16(bits) => input_major_bf16(a, m, k, rows, bits, n, first, count, out),
            }
        }
    }
}
