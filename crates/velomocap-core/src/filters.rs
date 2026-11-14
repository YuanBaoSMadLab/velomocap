//! 1-D signal helpers with exact scipy/numpy semantics.
//!
//! The contact planner and the correction report depend on matching
//! `scipy.ndimage` and `numpy` bit-for-bit, so every routine here documents
//! the reference call it replaces and keeps the same boundary handling.

use crate::cpp;

/// `scipy.ndimage.gaussian_filter1d(input, sigma, mode='nearest')`
/// (default `truncate=4.0`).
pub fn gaussian_filter1d(input: &[f64], sigma: f64) -> Vec<f64> {
    let mut out = vec![0.0; input.len()];
    cpp::gaussian_filter1d(input, sigma, 4.0, &mut out);
    out
}

/// `scipy.ndimage.minimum_filter1d(input, size, mode='nearest')`
pub fn minimum_filter1d(input: &[f64], size: usize) -> Vec<f64> {
    let mut out = vec![0.0; input.len()];
    cpp::minmax_filter1d(input, size.max(1), false, &mut out);
    out
}

/// `scipy.ndimage.maximum_filter1d(input, size, mode='nearest')`
pub fn maximum_filter1d(input: &[f64], size: usize) -> Vec<f64> {
    let mut out = vec![0.0; input.len()];
    cpp::minmax_filter1d(input, size.max(1), true, &mut out);
    out
}

/// `numpy.gradient(input)` with the default `edge_order=1`.
pub fn gradient1d(input: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; input.len()];
    cpp::gradient1d(input, &mut out);
    out
}

/// `numpy.interp(x, xp, fp)` for monotonically increasing `xp`.
pub fn interp(xp: &[f64], fp: &[f64], x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    cpp::interp(xp, fp, x, &mut out);
    out
}

/// `numpy.interp(arange(n), indices, values)` on integer sample indices.
pub fn interp_on_samples(indices: &[usize], values: &[f64], n: usize) -> Vec<f64> {
    let xp: Vec<f64> = indices.iter().map(|&i| i as f64).collect();
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    interp(&xp, values, &x)
}

/// `numpy.quantile(values, q)` with the default linear interpolation method.
pub fn quantile_linear(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let h = (n as f64 - 1.0) * q.clamp(0.0, 1.0);
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let t = h - lo as f64;
        sorted[lo] + t * (sorted[hi] - sorted[lo])
    }
}

/// `numpy.minimum.accumulate(values)`
pub fn minimum_accumulate(values: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len());
    let mut best = f64::INFINITY;
    for &v in values {
        if v < best {
            best = v;
        }
        out.push(best);
    }
    out
}

/// Contiguous `True` runs of a mask as `(start, end)` half-open intervals.
pub fn intervals(mask: &[bool]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    for (i, &m) in mask.iter().enumerate() {
        match (m, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                out.push((s, i));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push((s, mask.len()));
    }
    out
}

/// Solve a symmetric banded system with `p` super/sub diagonals.
///
/// `diag` holds the main diagonal, `super_diags[k]` the (k+1)-th super diagonal
/// (length `n - k - 1`), and each column of `rhs` is one right-hand side.
/// Elimination without pivoting is exact enough here because every matrix the
/// pipeline builds is strictly diagonally dominant (see `contact::grounding_shift`).
pub fn solve_symmetric_banded(p: usize, diag: &[f64], super_diags: &[Vec<f64>], rhs: &mut [Vec<f64>]) {
    let n = diag.len();
    if n == 0 {
        return;
    }
    assert!(super_diags.len() == p, "expected {p} super diagonals");
    for (k, band) in super_diags.iter().enumerate() {
        assert_eq!(band.len(), n - k - 1, "super diagonal {k} length");
    }
    // Band storage: a[i][p + off] = A[i][i + off] for off in -p..=p.
    let width = 2 * p + 1;
    let mut a = vec![0.0; n * width];
    for i in 0..n {
        a[i * width + p] = diag[i];
        for k in 0..p {
            let off = k + 1;
            if i + off < n {
                let v = super_diags[k][i];
                a[i * width + p + off] = v;
                a[(i + off) * width + p - off] = v;
            }
        }
    }

    for k in 0..n {
        let pivot = a[k * width + p];
        if pivot == 0.0 {
            continue;
        }
        let last_row = (k + p + 1).min(n);
        for i in (k + 1)..last_row {
            // a[i][k] lives at offset k - i, which is structurally zero when
            // the row is more than p bands away.
            let factor = a[i * width + p - (i - k)] / pivot;
            if factor == 0.0 {
                continue;
            }
            let last_col = (k + p + 1).min(n);
            for j in k..last_col {
                // Reordered so the usize arithmetic never goes negative.
                let target = i * width + p + j - i;
                let source = k * width + p + j - k;
                a[target] -= factor * a[source];
            }
            for column in rhs.iter_mut() {
                let value = column[k];
                column[i] -= factor * value;
            }
        }
    }

    for column in rhs.iter_mut() {
        for k in (0..n).rev() {
            let mut sum = column[k];
            let last_col = (k + p + 1).min(n);
            for j in (k + 1)..last_col {
                sum -= a[k * width + p + j - k] * column[j];
            }
            let pivot = a[k * width + p];
            column[k] = if pivot == 0.0 { 0.0 } else { sum / pivot };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_matches_scipy() {
        // scipy.ndimage.gaussian_filter1d([0,1,0,0,0], 1.0, mode='nearest')
        let out = gaussian_filter1d(&[0.0, 1.0, 0.0, 0.0, 0.0], 1.0);
        let expected = [0.241_971_45, 0.398_943_47, 0.241_971_45, 0.053_991_13, 0.004_431_86];
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-7, "{out:?} vs {expected:?}");
        }
        // scipy.ndimage.gaussian_filter1d([0,1,2,3,4], 1.5, mode='nearest')
        let out = gaussian_filter1d(&[0.0, 1.0, 2.0, 3.0, 4.0], 1.5);
        let expected = [0.574_492_2, 1.198_759_51, 2.0, 2.801_240_49, 3.425_507_8];
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-7, "{out:?} vs {expected:?}");
        }
    }

    #[test]
    fn rank_filters_match_scipy() {
        let input = [5.0, 3.0, 9.0, 1.0, 7.0];
        assert_eq!(minimum_filter1d(&input, 3), vec![3.0, 3.0, 1.0, 1.0, 1.0]);
        assert_eq!(maximum_filter1d(&input, 3), vec![5.0, 9.0, 9.0, 9.0, 7.0]);
        assert_eq!(gradient1d(&[0.0, 1.0, 2.0, 3.0, 4.0]), vec![1.0; 5]);
    }

    #[test]
    fn banded_solve_matches_dense() {
        // Tridiagonal 4x4 with diagonal 4, off -1.
        let diag = vec![4.0; 4];
        let off = vec![-1.0; 3];
        let mut rhs = vec![vec![1.0, 2.0, 3.0, 4.0]];
        solve_symmetric_banded(1, &diag, &[off], &mut rhs);
        let x = &rhs[0];
        // Residual check against the dense product.
        for i in 0..4 {
            let mut acc = diag[i] * x[i];
            if i > 0 {
                acc -= x[i - 1];
            }
            if i + 1 < 4 {
                acc -= x[i + 1];
            }
            let want = (i + 1) as f64;
            assert!((acc - want).abs() < 1e-12, "row {i}: {acc} vs {want}");
        }
    }

    #[test]
    fn quantile_matches_numpy() {
        let values = [0.0, 10.0, 20.0, 30.0, 40.0];
        assert!((quantile_linear(&values, 0.05) - 2.0).abs() < 1e-12);
        assert!((quantile_linear(&values, 0.5) - 20.0).abs() < 1e-12);
    }
}
