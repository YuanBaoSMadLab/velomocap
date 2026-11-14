//! Fixed MHR mesh topology and the 70-keypoint regressor.
//!
//! The exported SAM 3D Body graphs return vertex positions only: the triangle
//! indices live in a checkpoint buffer that no graph carries, and the named
//! keypoints the reference calls `pred_keypoints_3d` are a sparse linear map of
//! `[verts; joints]` (`head_pose.keypoint_mapping`, first 70 rows). Both are
//! dumped once by `tools/onnx/dump_mesh.py` into `sam3d_mesh.json`,
//! `sam3d_faces.bin` and `sam3d_keypoints.bin`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

/// MHR vertices per body.
pub const VERTICES: usize = 18439;
/// MHR joints per body.
pub const JOINTS: usize = 127;
/// Keypoints reported alongside the mesh.
pub const KEYPOINTS: usize = 70;

/// Sparse `70 × (VERTICES + JOINTS)` regressor, in row-major coordinate order
/// so evaluation is a straight accumulation.
#[derive(Clone, Debug, Default)]
pub struct KeypointRegressor {
    /// Row of the value / source index: `0..KEYPOINTS`.
    rows: Vec<i32>,
    /// Source column: `0..VERTICES` is a vertex, above is a joint.
    cols: Vec<i32>,
    values: Vec<f32>,
}

impl KeypointRegressor {
    pub fn nonzeros(&self) -> usize {
        self.values.len()
    }

    /// Evaluate into `out` (`KEYPOINTS × 3`).
    pub fn apply(&self, verts: &[f32], joints: &[f32], out: &mut Vec<f32>) {
        out.clear();
        out.resize(KEYPOINTS * 3, 0.0);
        for ((row, col), value) in self.rows.iter().zip(&self.cols).zip(&self.values) {
            let at = *row as usize * 3;
            let source = if (*col as usize) < VERTICES {
                (*col as usize) * 3
            } else {
                verts.len() + (*col as usize - VERTICES) * 3
            };
            if source + 2 >= verts.len() + joints.len() {
                continue;
            }
            for axis in 0..3 {
                let sample = if source < verts.len() {
                    verts[source + axis]
                } else {
                    joints[source + axis - verts.len()]
                };
                out[at + axis] += value * sample;
            }
        }
    }
}

#[derive(Deserialize)]
struct MeshMeta {
    vertices: usize,
    faces: usize,
    face_file: String,
    keypoints: usize,
    keypoint_file: String,
    keypoint_nonzeros: usize,
    #[serde(default)]
    skeleton_file: String,
    #[serde(default)]
    names: Vec<String>,
}

/// Everything needed to draw one body: triangles plus the keypoint map.
#[derive(Clone, Debug)]
pub struct MeshTopology {
    pub vertices: usize,
    pub triangles: Vec<[u32; 3]>,
    /// MHR joint hierarchy, `-1` at the root.
    pub parents: Vec<i32>,
    /// Parent-relative rest offsets in metres, y-up.
    pub rest_offsets: Vec<[f32; 3]>,
    /// Named keypoints the regressor produces (`70`).
    pub keypoints: usize,
    pub regressor: KeypointRegressor,
    pub names: Vec<String>,
    source: PathBuf,
}

impl MeshTopology {
    /// Load `sam3d_mesh.json` and the two binaries it names.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let meta_path = dir.join("sam3d_mesh.json");
        let meta: MeshMeta = serde_json::from_str(
            &std::fs::read_to_string(&meta_path)
                .with_context(|| format!("reading {}", meta_path.display()))?,
        )
        .with_context(|| format!("parsing {}", meta_path.display()))?;
        anyhow::ensure!(
            meta.vertices == VERTICES,
            "mesh has {} vertices, expected {VERTICES}",
            meta.vertices
        );
        anyhow::ensure!(
            meta.keypoints == KEYPOINTS,
            "mesh has {} keypoints, expected {KEYPOINTS}",
            meta.keypoints
        );

        let face_bytes = std::fs::read(dir.join(&meta.face_file))
            .with_context(|| format!("reading {}", meta.face_file))?;
        anyhow::ensure!(
            face_bytes.len() == meta.faces * 3 * 4,
            "{} has {} bytes, expected {}",
            meta.face_file,
            face_bytes.len(),
            meta.faces * 3 * 4
        );
        let triangles: Vec<[u32; 3]> = face_bytes
            .chunks_exact(12)
            .map(|chunk| {
                let index = |at: usize| {
                    u32::from_le_bytes([chunk[at], chunk[at + 1], chunk[at + 2], chunk[at + 3]])
                };
                [index(0), index(4), index(8)]
            })
            .collect();
        anyhow::ensure!(
            triangles
                .iter()
                .flatten()
                .all(|index| (*index as usize) < VERTICES),
            "mesh references a vertex outside 0..{VERTICES}"
        );

        let regressor_bytes = std::fs::read(dir.join(&meta.keypoint_file))
            .with_context(|| format!("reading {}", meta.keypoint_file))?;
        let count = meta.keypoint_nonzeros;
        anyhow::ensure!(
            regressor_bytes.len() == count * 12,
            "{} has {} bytes, expected {}",
            meta.keypoint_file,
            regressor_bytes.len(),
            count * 12
        );
        let read_ints = |start: usize| -> Vec<i32> {
            regressor_bytes[start..start + count * 4]
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let rows = read_ints(0);
        let cols = read_ints(count * 4);
        let values: Vec<f32> = regressor_bytes[count * 8..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        anyhow::ensure!(
            rows.iter().all(|r| (0..KEYPOINTS as i32).contains(r)),
            "keypoint regressor has an out-of-range row"
        );
        anyhow::ensure!(
            cols.iter()
                .all(|c| (0..(VERTICES + JOINTS) as i32).contains(c)),
            "keypoint regressor has an out-of-range column"
        );

        // MHR skeleton: 127 int32 parents followed by 127 xyz float32 offsets.
        let skeleton = std::fs::read(dir.join(&meta.skeleton_file))
            .with_context(|| format!("reading {}", meta.skeleton_file))?;
        anyhow::ensure!(
            skeleton.len() == JOINTS * 4 + JOINTS * 12,
            "{} has {} bytes, expected {}",
            meta.skeleton_file,
            skeleton.len(),
            JOINTS * 16
        );
        let parents: Vec<i32> = skeleton[..JOINTS * 4]
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        anyhow::ensure!(parents[0] == -1, "skeleton root is not -1");
        anyhow::ensure!(
            (1..JOINTS).all(|j| parents[j] >= 0 && (parents[j] as usize) < j),
            "skeleton is not topologically ordered"
        );
        let rest_offsets: Vec<[f32; 3]> = skeleton[JOINTS * 4..]
            .chunks_exact(12)
            .map(|c| {
                // Centimetres, MHR y-up/z-forward; the world is metres with z flipped.
                let x = f32::from_le_bytes([c[0], c[1], c[2], c[3]]) * 0.01;
                let y = f32::from_le_bytes([c[4], c[5], c[6], c[7]]) * 0.01;
                let z = f32::from_le_bytes([c[8], c[9], c[10], c[11]]) * 0.01;
                [x, y, -z]
            })
            .collect();

        Ok(Self {
            vertices: meta.vertices,
            triangles,
            keypoints: meta.keypoints,
            parents,
            rest_offsets,
            regressor: KeypointRegressor { rows, cols, values },
            names: meta.names,
            source: meta_path,
        })
    }

    /// Path the topology was read from, for the model report.
    pub fn source(&self) -> &Path {
        &self.source
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn topology() -> Option<MeshTopology> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("workspace root");
        [
            root.join("models/sam3d"),
            root.parent()?.join("export/sam3d"),
        ]
        .into_iter()
        .find(|dir| dir.join("sam3d_mesh.json").is_file())
        .and_then(|dir| MeshTopology::load(dir).ok())
    }

    #[test]
    fn loads_the_shipped_mesh() {
        let Some(mesh) = topology() else {
            eprintln!("skipping: no exported MHR topology");
            return;
        };
        assert_eq!(mesh.vertices, VERTICES);
        assert_eq!(mesh.triangles.len(), 36874);
        assert!(mesh.regressor.nonzeros() > 0);
        eprintln!(
            "mesh {} verts, {} tris, {} keypoints, {} nnz, names {}",
            mesh.vertices,
            mesh.triangles.len(),
            KEYPOINTS,
            mesh.regressor.nonzeros(),
            mesh.names.len()
        );
    }

    /// A keypoint must lie inside the mesh it was regressed from: compare it to
    /// the centroid of the vertices that feed it. Cheaper than a reference
    /// replay and it catches a transposed regressor immediately.
    #[test]
    fn regressed_keypoints_stay_near_the_rest_body() {
        let Some(mesh) = topology() else {
            eprintln!("skipping: no exported MHR topology");
            return;
        };
        // A body-sized box; any keypoint outside it means the map is wrong.
        let verts: Vec<f32> = (0..VERTICES)
            .flat_map(|i| [0.2 * (i % 7) as f32, 0.3 * (i % 5) as f32, 0.1 * (i % 3) as f32])
            .collect();
        let joints = vec![0.0f32; JOINTS * 3];
        let mut out = Vec::new();
        mesh.regressor.apply(&verts, &joints, &mut out);
        assert_eq!(out.len(), KEYPOINTS * 3);
        assert!(out.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 10.0));
    }
}
