//! Reader for NumPy `.npz` archives and `.npy` arrays, in pure Rust.
//!
//! SMPL-X ships as `.npz`: a zip archive whose members are `.npy` files. This
//! module parses the small subset of the format those files actually use —
//! C-order or Fortran-order numeric arrays of a handful of dtypes — so the
//! body model can be loaded without NumPy, Python or any native dependency.

use std::io::Read;

use anyhow::{Context, Result};

/// A decoded array, flattened to row-major order regardless of on-disk layout.
#[derive(Clone, Debug)]
pub enum Array {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U32(Vec<u32>),
}

impl Array {
    pub fn len(&self) -> usize {
        match self {
            Array::F32(v) => v.len(),
            Array::F64(v) => v.len(),
            Array::I32(v) => v.len(),
            Array::I64(v) => v.len(),
            Array::U32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copy to `f32`, which is what the model maths uses.
    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            Array::F32(v) => v.clone(),
            Array::F64(v) => v.iter().map(|x| *x as f32).collect(),
            Array::I32(v) => v.iter().map(|x| *x as f32).collect(),
            Array::I64(v) => v.iter().map(|x| *x as f32).collect(),
            Array::U32(v) => v.iter().map(|x| *x as f32).collect(),
        }
    }

    /// Copy to `i64` (used for the kinematic tree).
    pub fn to_i64(&self) -> Vec<i64> {
        match self {
            Array::F32(v) => v.iter().map(|x| *x as i64).collect(),
            Array::F64(v) => v.iter().map(|x| *x as i64).collect(),
            Array::I32(v) => v.iter().map(|x| *x as i64).collect(),
            Array::I64(v) => v.clone(),
            Array::U32(v) => v.iter().map(|x| *x as i64).collect(),
        }
    }

    pub fn to_u32(&self) -> Vec<u32> {
        match self {
            Array::F32(v) => v.iter().map(|x| *x as u32).collect(),
            Array::F64(v) => v.iter().map(|x| *x as u32).collect(),
            Array::I32(v) => v.iter().map(|x| *x as u32).collect(),
            Array::I64(v) => v.iter().map(|x| *x as u32).collect(),
            Array::U32(v) => v.clone(),
        }
    }
}

/// One member of an `.npz` archive.
#[derive(Clone, Debug)]
pub struct NpzArray {
    pub shape: Vec<usize>,
    pub fortran_order: bool,
    pub array: Array,
}

impl NpzArray {
    /// Total element count.
    pub fn len(&self) -> usize {
        self.array.len()
    }

    pub fn is_empty(&self) -> bool {
        self.array.is_empty()
    }

    /// Element `(i, j)` of a 2-D array, honouring the on-disk order.
    pub fn get2(&self, i: usize, j: usize) -> f32 {
        let rows = self.shape.first().copied().unwrap_or(0);
        let cols = self.shape.get(1).copied().unwrap_or(0);
        let index = if self.fortran_order { j * rows + i } else { i * cols + j };
        match &self.array {
            Array::F32(v) => v[index],
            Array::F64(v) => v[index] as f32,
            Array::I32(v) => v[index] as f32,
            Array::I64(v) => v[index] as f32,
            Array::U32(v) => v[index] as f32,
        }
    }

    /// Row-major copy of a 2-D array (transposing Fortran storage).
    pub fn to_row_major(&self) -> Vec<f32> {
        if !self.fortran_order {
            return self.array.to_f32();
        }
        let rows = self.shape.first().copied().unwrap_or(0);
        let cols = self.shape.get(1).copied().unwrap_or(0);
        let source = self.array.to_f32();
        let mut out = vec![0.0f32; source.len()];
        for i in 0..rows {
            for j in 0..cols {
                out[i * cols + j] = source[j * rows + i];
            }
        }
        out
    }
}

/// Parse a `.npy` byte buffer.
pub fn parse_npy(bytes: &[u8]) -> Result<NpzArray> {
    const MAGIC: &[u8; 6] = b"\x93NUMPY";
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        anyhow::bail!("not a .npy buffer");
    }
    let major = bytes[6];
    let (header_len, header_start) = match major {
        1 => (
            u16::from_le_bytes([bytes[8], bytes[9]]) as usize,
            10usize,
        ),
        2 | 3 => {
            if bytes.len() < 12 {
                anyhow::bail!("truncated .npy header");
            }
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12usize,
            )
        }
        other => anyhow::bail!("unsupported .npy version {other}"),
    };
    let header_end = header_start + header_len;
    if bytes.len() < header_end {
        anyhow::bail!("truncated .npy header payload");
    }
    let header = std::str::from_utf8(&bytes[header_start..header_end])
        .context(".npy header is not UTF-8")?;

    let descr = header_value(header, "descr").ok_or_else(|| anyhow::anyhow!("missing descr"))?;
    let fortran_order = header
        .split("'fortran_order':")
        .nth(1)
        .map(|rest| rest.trim_start().starts_with("True"))
        .unwrap_or(false);
    let shape_text =
        header_value(header, "shape").ok_or_else(|| anyhow::anyhow!("missing shape"))?;
    let shape = parse_shape(&shape_text)?;

    let payload = &bytes[header_end..];
    let count: usize = shape.iter().product::<usize>().max(1);

    // descr looks like "<f8", "<i4", "|u1", "<U7".
    let (kind, itemsize) = descr_kind(&descr)?;
    let needed = count * itemsize;
    if payload.len() < needed {
        anyhow::bail!(
            ".npy payload is {} bytes, expected {needed} for shape {shape:?}",
            payload.len()
        );
    }

    let array = match kind {
        Kind::F32 => Array::F32(read_f32(payload, count)),
        Kind::F64 => Array::F64(read_f64(payload, count)),
        Kind::I32 => Array::I32(read_i32(payload, count)),
        Kind::I64 => Array::I64(read_i64(payload, count)),
        Kind::U32 => Array::U32(read_u32(payload, count)),
        Kind::Text => {
            // Strings (e.g. bs_type='lrotmin') are not needed numerically.
            Array::U32(vec![0; count])
        }
    };

    Ok(NpzArray {
        shape,
        fortran_order,
        array,
    })
}

/// Read one member of an `.npz` archive by name (without the `.npy` suffix).
pub fn read_member(path: &std::path::Path, name: &str) -> Result<NpzArray> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("reading {} as a zip archive", path.display()))?;
    let key = format!("{name}.npy");
    let mut entry = archive
        .by_name(&key)
        .with_context(|| format!("{name} not found in {}", path.display()))?;
    let mut buffer = Vec::with_capacity(entry.size() as usize);
    entry
        .read_to_end(&mut buffer)
        .with_context(|| format!("decompressing {key}"))?;
    parse_npy(&buffer)
}

/// Names of every member in an `.npz` archive.
pub fn members(path: &std::path::Path) -> Result<Vec<String>> {
    let file = std::fs::File::open(path)?;
    let archive = zip::ZipArchive::new(file)?;
    Ok(archive.file_names().map(|n| n.to_string()).collect())
}

enum Kind {
    F32,
    F64,
    I32,
    I64,
    U32,
    Text,
}

fn descr_kind(descr: &str) -> Result<(Kind, usize)> {
    let cleaned = descr.trim().trim_matches('\'').trim_matches('"');
    // Strip byte-order marker.
    let body = cleaned
        .strip_prefix('<')
        .or_else(|| cleaned.strip_prefix('>'))
        .or_else(|| cleaned.strip_prefix('='))
        .or_else(|| cleaned.strip_prefix('|'))
        .unwrap_or(cleaned);
    let bytes = body.as_bytes();
    if bytes.is_empty() {
        anyhow::bail!("empty dtype in {descr:?}");
    }
    let kind = match bytes[0] as char {
        'f' => match bytes.get(1) {
            Some(b'4') => Kind::F32,
            Some(b'8') => Kind::F64,
            _ => anyhow::bail!("unsupported float dtype {descr:?}"),
        },
        'i' => match bytes.get(1) {
            Some(b'4') => Kind::I32,
            Some(b'8') => Kind::I64,
            _ => anyhow::bail!("unsupported int dtype {descr:?}"),
        },
        'u' => match bytes.get(1) {
            Some(b'4') => Kind::U32,
            Some(b'1') => Kind::U32, // widened below
            _ => anyhow::bail!("unsupported uint dtype {descr:?}"),
        },
        'U' => Kind::Text,
        other => anyhow::bail!("unsupported dtype kind {other:?} in {descr:?}"),
    };
    let itemsize = body[1..].parse::<usize>().unwrap_or(8);
    Ok((kind, itemsize))
}

fn header_value(header: &str, key: &str) -> Option<String> {
    let needle = format!("'{key}':");
    let rest = header.split(&needle).nth(1)?;
    let rest = rest.trim_start();
    if rest.starts_with('(') {
        let end = rest.find(')')?;
        Some(rest[1..end].to_string())
    } else {
        let end = rest.find(',').unwrap_or(rest.len());
        Some(rest[..end].trim().trim_matches('\'').to_string())
    }
}

fn parse_shape(text: &str) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for part in text.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        out.push(part.parse::<usize>()?);
    }
    Ok(out)
}

fn read_f32(bytes: &[u8], count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| f32::from_le_bytes([bytes[i * 4], bytes[i * 4 + 1], bytes[i * 4 + 2], bytes[i * 4 + 3]]))
        .collect()
}

fn read_f64(bytes: &[u8], count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let mut b = [0u8; 8];
            b.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
            f64::from_le_bytes(b)
        })
        .collect()
}

fn read_i32(bytes: &[u8], count: usize) -> Vec<i32> {
    (0..count)
        .map(|i| i32::from_le_bytes([bytes[i * 4], bytes[i * 4 + 1], bytes[i * 4 + 2], bytes[i * 4 + 3]]))
        .collect()
}

fn read_i64(bytes: &[u8], count: usize) -> Vec<i64> {
    (0..count)
        .map(|i| {
            let mut b = [0u8; 8];
            b.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
            i64::from_le_bytes(b)
        })
        .collect()
}

fn read_u32(bytes: &[u8], count: usize) -> Vec<u32> {
    (0..count)
        .map(|i| u32::from_le_bytes([bytes[i * 4], bytes[i * 4 + 1], bytes[i * 4 + 2], bytes[i * 4 + 3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy_v1(descr: &str, shape: &str, fortran: bool, payload: &[u8]) -> Vec<u8> {
        // NumPy writes Python literals: `True` / `False`, not Rust's lowercase.
        let flag = if fortran { "True" } else { "False" };
        let header = format!(
            "{{'descr': '{descr}', 'fortran_order': {flag}, 'shape': ({shape}), }}"
        );
        let mut padded = header;
        // NumPy pads the header to a 64-byte boundary including the magic.
        while (10 + padded.len()) % 64 != 63 {
            padded.push(' ');
        }
        padded.push('\n');
        let mut out = Vec::new();
        out.extend_from_slice(b"\x93NUMPY");
        out.push(1);
        out.push(0);
        out.extend_from_slice(&(padded.len() as u16).to_le_bytes());
        out.extend_from_slice(padded.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn reads_c_order_float64() {
        let payload: Vec<u8> = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bytes = npy_v1("<f8", "2, 3", false, &payload);
        let array = parse_npy(&bytes).unwrap();
        assert_eq!(array.shape, vec![2, 3]);
        assert!(!array.fortran_order);
        assert_eq!(array.get2(1, 2), 6.0);
        assert_eq!(array.get2(0, 1), 2.0);
    }

    #[test]
    fn reads_fortran_order_and_transposes() {
        // Same 2x3 matrix stored column-major: columns (1,4) (2,5) (3,6).
        let payload: Vec<u8> = [1.0f64, 4.0, 2.0, 5.0, 3.0, 6.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bytes = npy_v1("<f8", "2, 3", true, &payload);
        let array = parse_npy(&bytes).unwrap();
        assert!(array.fortran_order);
        assert_eq!(array.get2(0, 0), 1.0);
        assert_eq!(array.get2(1, 2), 6.0);
        assert_eq!(array.to_row_major(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn reads_int64() {
        let payload: Vec<u8> = [0i64, 1, 2]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bytes = npy_v1("<i8", "3,", false, &payload);
        let array = parse_npy(&bytes).unwrap();
        assert_eq!(array.array.to_i64(), vec![0, 1, 2]);
    }
}
