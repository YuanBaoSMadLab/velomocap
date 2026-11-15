//! Import a humanoid character (VRM / glTF / GLB) and drive it with the clip.
//!
//! The reference's "Import Character" takes a Mixamo FBX, a VRM or a PMX and
//! retargets the generated motion onto it. This is the native reader for the
//! glTF-container formats, which is where VRM avatars and most modern "virtual
//! live" files live: one binary chunk holding the node tree, one skin, and the
//! skinned mesh.
//!
//! The retarget is orientation-only, which is what a T-pose bind needs: the
//! character keeps its own skeleton (its limb lengths are its own), and each bone
//! is rotated by the clip's world-space rotation relative to the clip's rest
//! pose. Bone names are matched with [`velomocap_motion::retarget::canonicalize`],
//! which already knows Mixamo, VRM `J_Bip_*`, MMD and the other vendor spellings.

use std::path::Path;

use anyhow::{Context, Result};

/// One bone of the imported skeleton.
#[derive(Clone, Debug)]
pub struct Bone {
    /// Name as it appears in the file, for the report and for unmatched bones.
    pub name: String,
    /// Canonical bone index (0..52), when the name resolves.
    pub canonical: Option<usize>,
    /// Bind-pose world position.
    pub bind: [f32; 3],
    /// Bind-pose world rotation, row-major 3×3.
    pub bind_rot: [f32; 9],
    /// `inverse(bind world matrix)`, the skinning factor.
    pub inverse_bind: [f32; 16],
}

/// A skinned mesh ready to be drawn.
#[derive(Clone)]
pub struct Character {
    pub name: String,
    pub bones: Vec<Bone>,
    /// Bind-pose vertices, xyz.
    pub rest: Vec<f32>,
    /// Four joint indices and weights per vertex.
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
    pub faces: Vec<[u32; 3]>,
}

impl Character {
    pub fn vertex_count(&self) -> usize {
        self.rest.len() / 3
    }

    /// How many bones resolved to a canonical name, out of the 52 the clip has.
    pub fn matched(&self) -> usize {
        self.bones.iter().filter(|bone| bone.canonical.is_some()).count()
    }

    /// The bones a clip cannot drive, as names, for the message.
    pub fn unmatched(&self) -> Vec<&str> {
        self.bones
            .iter()
            .filter(|bone| bone.canonical.is_none())
            .map(|bone| bone.name.as_str())
            .collect()
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let gltf = Gltf::parse(&bytes)
            .with_context(|| format!("{} is not a readable glTF/GLB file", path.display()))?;
        let name = path
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_else(|| "character".to_string());
        gltf.build(name)
    }

    /// Pose the mesh with the clip's joint rotations.
    ///
    /// `world` is the clip's per-bone world rotation (52 × 3×3), `rest` the
    /// clip's rest rotation, both in the same order as [`Bone::canonical`].
    /// `translation` moves the character with the root.
    pub fn pose(&self, world: &[[f32; 9]], rest: &[[f32; 9]], translation: [f32; 3]) -> Vec<f32> {
        // Skin matrix per bone: rotate its bind orientation by the clip's own
        // world-space change, then undo the bind pose.
        let mut skin = vec![[0.0f32; 16]; self.bones.len()];
        for (index, bone) in self.bones.iter().enumerate() {
            let rotation = match bone.canonical {
                Some(canonical) => {
                    let delta = multiply(&world[canonical], &transpose(&rest[canonical]));
                    multiply(&delta, &bone.bind_rot)
                }
                None => bone.bind_rot,
            };
            // M = [R | translation + bind] * inverse_bind
            let at = bone.bind;
            let mut world_matrix = [
                rotation[0], rotation[1], rotation[2], at[0] + translation[0],
                rotation[3], rotation[4], rotation[5], at[1] + translation[1],
                rotation[6], rotation[7], rotation[8], at[2] + translation[2],
                0.0, 0.0, 0.0, 1.0,
            ];
            world_matrix = multiply4(&world_matrix, &bone.inverse_bind);
            skin[index] = world_matrix;
        }
        let mut out = vec![0.0f32; self.rest.len()];
        for vertex in 0..self.vertex_count() {
            let at = [self.rest[vertex * 3], self.rest[vertex * 3 + 1], self.rest[vertex * 3 + 2]];
            let mut sum = [0.0f32; 3];
            let mut total = 0.0f32;
            for slot in 0..4 {
                let weight = self.weights[vertex][slot];
                if weight <= 0.0 {
                    continue;
                }
                let joint = self.joints[vertex][slot] as usize;
                let Some(matrix) = skin.get(joint) else {
                    continue;
                };
                for axis in 0..3 {
                    sum[axis] += weight
                        * (matrix[axis * 4] * at[0]
                            + matrix[axis * 4 + 1] * at[1]
                            + matrix[axis * 4 + 2] * at[2]
                            + matrix[axis * 4 + 3]);
                }
                total += weight;
            }
            if total <= 0.0 {
                out[vertex * 3..vertex * 3 + 3].copy_from_slice(&at);
            } else {
                for axis in 0..3 {
                    out[vertex * 3 + axis] = sum[axis] / total;
                }
            }
        }
        out
    }
}

fn transpose(m: &[f32; 9]) -> [f32; 9] {
    [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]
}

fn multiply(a: &[f32; 9], b: &[f32; 9]) -> [f32; 9] {
    let mut out = [0.0f32; 9];
    for row in 0..3 {
        for column in 0..3 {
            out[row * 3 + column] = (0..3).map(|k| a[row * 3 + k] * b[k * 3 + column]).sum();
        }
    }
    out
}

fn multiply4(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for row in 0..4 {
        for column in 0..4 {
            out[row * 4 + column] = (0..4).map(|k| a[row * 4 + k] * b[k * 4 + column]).sum();
        }
    }
    out
}

fn rotation_of(m: &[f32; 16]) -> [f32; 9] {
    [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]]
}

fn translation_of(m: &[f32; 16]) -> [f32; 3] {
    [m[3], m[7], m[11]]
}

/// glTF matrices are column-major; everything here works row-major.
fn transpose4(m: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for row in 0..4 {
        for column in 0..4 {
            out[row * 4 + column] = m[column * 4 + row];
        }
    }
    out
}

/// The parts of a glTF document this reader needs.
struct Gltf {
    document: serde_json::Value,
    bin: Vec<u8>,
}

impl Gltf {
    fn parse(bytes: &[u8]) -> Result<Self> {
        // GLB: magic, version, length, then chunks (JSON, BIN).
        if bytes.len() >= 12 && &bytes[..4] == b"glTF" {
            let mut bin = Vec::new();
            let mut json = None;
            let mut offset = 12;
            while offset + 8 <= bytes.len() {
                let length = u32::from_le_bytes([
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                ]) as usize;
                let kind = &bytes[offset + 4..offset + 8];
                let start = offset + 8;
                let end = (start + length).min(bytes.len());
                if kind == b"JSON" {
                    json = Some(&bytes[start..end]);
                } else if kind == b"BIN\0" {
                    bin = bytes[start..end].to_vec();
                }
                offset = start + length;
                // Every chunk starts on a four-byte boundary.
                offset = (offset + 3) & !3;
            }
            let json = json.context("the GLB has no JSON chunk")?;
            let document = serde_json::from_slice(json).context("the GLB JSON is invalid")?;
            return Ok(Self { document, bin });
        }
        // A plain .gltf has its geometry in an external buffer, which this
        // reader does not open: say so instead of reporting a broken skin.
        anyhow::bail!("a plain .gltf keeps its buffer in a separate file; use .glb or .vrm");
    }

    fn build(&self, name: String) -> Result<Character> {
        let skins = self
            .document
            .get("skins")
            .and_then(|value| value.as_array())
            .context("no skin: the file is not a skinned character")?;
        let skin = skins.first().context("no skin")?;
        let nodes = self
            .document
            .get("nodes")
            .and_then(|value| value.as_array())
            .context("no nodes")?;
        let joint_nodes: Vec<usize> = skin
            .get("joints")
            .and_then(|value| value.as_array())
            .context("the skin has no joints")?
            .iter()
            .filter_map(|value| value.as_u64().map(|value| value as usize))
            .collect();
        anyhow::ensure!(!joint_nodes.is_empty(), "the skin has no joints");

        // Bind world matrices come straight from the inverse bind accessor.
        let inverse = skin
            .get("inverseBindMatrices")
            .and_then(|value| value.as_u64())
            .context("the skin has no inverse bind matrices")? as usize;
        let inverse = self.read_mat4(inverse)?;
        anyhow::ensure!(
            inverse.len() >= joint_nodes.len(),
            "the inverse bind matrices do not cover the joints"
        );

        let mut bones = Vec::with_capacity(joint_nodes.len());
        for (slot, node) in joint_nodes.iter().enumerate() {
            let inverse_bind = transpose4(&inverse[slot]);
            let bind_matrix = invert(&inverse_bind);
            bones.push(Bone {
                name: nodes
                    .get(*node)
                    .and_then(|node| node.get("name"))
                    .and_then(|name| name.as_str())
                    .unwrap_or("bone")
                    .to_string(),
                canonical: nodes
                    .get(*node)
                    .and_then(|node| node.get("name"))
                    .and_then(|name| name.as_str())
                    .and_then(velomocap_motion::retarget::canonicalize),
                bind: translation_of(&bind_matrix),
                bind_rot: rotation_of(&bind_matrix),
                inverse_bind,
            });
        }

        // Every skinned primitive becomes triangles in one buffer.
        let mut rest: Vec<f32> = Vec::new();
        let mut joints: Vec<[u16; 4]> = Vec::new();
        let mut weights: Vec<[f32; 4]> = Vec::new();
        let mut faces: Vec<[u32; 3]> = Vec::new();
        let meshes = self
            .document
            .get("meshes")
            .and_then(|value| value.as_array())
            .cloned()
            .unwrap_or_default();
        for mesh in &meshes {
            let primitives = mesh
                .get("primitives")
                .and_then(|value| value.as_array())
                .cloned()
                .unwrap_or_default();
            for primitive in &primitives {
                let attributes = primitive.get("attributes").cloned().unwrap_or_default();
                let Some(position) = attributes.get("POSITION").and_then(|v| v.as_u64()) else {
                    continue;
                };
                let Some(joint_accessor) = attributes.get("JOINTS_0").and_then(|v| v.as_u64()) else {
                    continue;
                };
                let Some(weight_accessor) = attributes.get("WEIGHTS_0").and_then(|v| v.as_u64()) else {
                    continue;
                };
                let positions = self.read_vec3(position as usize)?;
                let base = (rest.len() / 3) as u32;
                for corner in positions.chunks_exact(3) {
                    rest.extend_from_slice(corner);
                }
                for joint in self.read_joints(joint_accessor as usize)? {
                    joints.push(joint);
                }
                for weight in self.read_weights(weight_accessor as usize)? {
                    weights.push(weight);
                }
                let indices = match primitive.get("indices").and_then(|v| v.as_u64()) {
                    Some(accessor) => self.read_indices(accessor as usize)?,
                    None => (0..positions.len() as u32 / 3).collect(),
                };
                for triangle in indices.chunks_exact(3) {
                    faces.push([
                        base + triangle[0],
                        base + triangle[1],
                        base + triangle[2],
                    ]);
                }
            }
        }
        anyhow::ensure!(!faces.is_empty(), "the file has no skinned mesh");
        anyhow::ensure!(
            joints.len() == rest.len() / 3 && weights.len() == rest.len() / 3,
            "the mesh attributes disagree on the vertex count"
        );
        Ok(Character {
            name,
            bones,
            rest,
            joints,
            weights,
            faces,
        })
    }

    fn view(&self, accessor: usize) -> Result<(&serde_json::Value, &[u8], usize, usize, usize)> {
        let accessors = self
            .document
            .get("accessors")
            .and_then(|value| value.as_array())
            .context("no accessors")?;
        let accessor = accessors.get(accessor).context("accessor out of range")?;
        let view_index = accessor.get("bufferView").and_then(|v| v.as_u64());
        let count = accessor.get("count").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let offset = accessor.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let Some(view_index) = view_index else {
            anyhow::bail!("accessor has no buffer view");
        };
        let views = self
            .document
            .get("bufferViews")
            .and_then(|value| value.as_array())
            .context("no buffer views")?;
        let view = views
            .get(view_index as usize)
            .context("buffer view out of range")?;
        let start = view.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let length = view.get("byteLength").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let stride = view.get("byteStride").and_then(|v| v.as_u64()).map(|v| v as usize);
        let end = (start + length).min(self.bin.len());
        Ok((accessor, &self.bin[start.min(end)..end], offset, count, stride.unwrap_or(0)))
    }

    fn read_vec3(&self, accessor: usize) -> Result<Vec<f32>> {
        let (_, data, offset, count, stride) = self.view(accessor)?;
        let stride = if stride == 0 { 12 } else { stride };
        let mut out = Vec::with_capacity(count * 3);
        for index in 0..count {
            let at = offset + index * stride;
            if at + 12 > data.len() {
                break;
            }
            for axis in 0..3 {
                out.push(f32::from_le_bytes([
                    data[at + axis * 4],
                    data[at + axis * 4 + 1],
                    data[at + axis * 4 + 2],
                    data[at + axis * 4 + 3],
                ]));
            }
        }
        anyhow::ensure!(!out.is_empty(), "the position accessor is empty");
        Ok(out)
    }

    fn read_mat4(&self, accessor: usize) -> Result<Vec<[f32; 16]>> {
        let (_, data, offset, count, stride) = self.view(accessor)?;
        let stride = if stride == 0 { 64 } else { stride };
        let mut out = Vec::with_capacity(count);
        for index in 0..count {
            let at = offset + index * stride;
            if at + 64 > data.len() {
                break;
            }
            let mut matrix = [0.0f32; 16];
            for value in 0..16 {
                matrix[value] = f32::from_le_bytes([
                    data[at + value * 4],
                    data[at + value * 4 + 1],
                    data[at + value * 4 + 2],
                    data[at + value * 4 + 3],
                ]);
            }
            out.push(matrix);
        }
        Ok(out)
    }

    fn read_joints(&self, accessor: usize) -> Result<Vec<[u16; 4]>> {
        let (accessor_value, data, offset, count, stride) = self.view(accessor)?;
        let kind = accessor_value
            .get("componentType")
            .and_then(|v| v.as_u64())
            .unwrap_or(5123);
        let width = match kind {
            5121 => 1, // unsigned byte
            5123 => 2, // unsigned short
            _ => 4,    // unsigned int
        };
        let stride = if stride == 0 { width * 4 } else { stride };
        let mut out = Vec::with_capacity(count);
        for index in 0..count {
            let at = offset + index * stride;
            let mut joints = [0u16; 4];
            for slot in 0..4 {
                let start = at + slot * width;
                if start + width > data.len() {
                    break;
                }
                joints[slot] = match width {
                    1 => data[start] as u16,
                    2 => u16::from_le_bytes([data[start], data[start + 1]]),
                    _ => u32::from_le_bytes([
                        data[start],
                        data[start + 1],
                        data[start + 2],
                        data[start + 3],
                    ]) as u16,
                };
            }
            out.push(joints);
        }
        Ok(out)
    }

    fn read_weights(&self, accessor: usize) -> Result<Vec<[f32; 4]>> {
        let (_, data, offset, count, stride) = self.view(accessor)?;
        let stride = if stride == 0 { 16 } else { stride };
        let mut out = Vec::with_capacity(count);
        for index in 0..count {
            let at = offset + index * stride;
            if at + 16 > data.len() {
                break;
            }
            let mut weights = [0.0f32; 4];
            for slot in 0..4 {
                weights[slot] = f32::from_le_bytes([
                    data[at + slot * 4],
                    data[at + slot * 4 + 1],
                    data[at + slot * 4 + 2],
                    data[at + slot * 4 + 3],
                ]);
            }
            out.push(weights);
        }
        Ok(out)
    }

    fn read_indices(&self, accessor: usize) -> Result<Vec<u32>> {
        let (accessor_value, data, offset, count, stride) = self.view(accessor)?;
        let kind = accessor_value
            .get("componentType")
            .and_then(|v| v.as_u64())
            .unwrap_or(5123);
        let width = match kind {
            5121 => 1,
            5123 => 2,
            _ => 4,
        };
        let stride = if stride == 0 { width } else { stride };
        let mut out = Vec::with_capacity(count);
        for index in 0..count {
            let at = offset + index * stride;
            if at + width > data.len() {
                break;
            }
            out.push(match width {
                1 => data[at] as u32,
                2 => u16::from_le_bytes([data[at], data[at + 1]]) as u32,
                _ => u32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]]),
            });
        }
        Ok(out)
    }
}

/// 4×4 inverse. A singular bind matrix is left as the identity, which keeps a
/// broken bone from taking the whole character with it.
fn invert(m: &[f32; 16]) -> [f32; 16] {
    let mut inv = [0.0f32; 16];
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15]
        + m[9] * m[7] * m[14]
        + m[13] * m[6] * m[11]
        - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15]
        - m[8] * m[7] * m[14]
        - m[12] * m[6] * m[11]
        + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15]
        + m[8] * m[7] * m[13]
        + m[12] * m[5] * m[11]
        - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14]
        - m[8] * m[6] * m[13]
        - m[12] * m[5] * m[10]
        + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15]
        - m[9] * m[3] * m[14]
        - m[13] * m[2] * m[11]
        + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15]
        + m[8] * m[3] * m[14]
        + m[12] * m[2] * m[11]
        - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15]
        - m[8] * m[3] * m[13]
        - m[12] * m[1] * m[11]
        + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14]
        + m[8] * m[2] * m[13]
        + m[12] * m[1] * m[10]
        - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15]
        + m[5] * m[3] * m[14]
        + m[13] * m[2] * m[7]
        - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15]
        - m[4] * m[3] * m[14]
        - m[12] * m[2] * m[7]
        + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15]
        + m[4] * m[3] * m[13]
        + m[12] * m[1] * m[7]
        - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14]
        - m[4] * m[2] * m[13]
        - m[12] * m[1] * m[6]
        + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11]
        - m[5] * m[3] * m[10]
        - m[9] * m[2] * m[7]
        + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11]
        + m[4] * m[3] * m[10]
        + m[8] * m[2] * m[7]
        - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11]
        - m[4] * m[3] * m[9]
        - m[8] * m[1] * m[7]
        + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10]
        + m[4] * m[2] * m[9]
        + m[8] * m[1] * m[6]
        - m[8] * m[2] * m[5];
    let mut determinant = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if determinant.abs() < 1e-8 {
        let mut identity = [0.0f32; 16];
        identity[0] = 1.0;
        identity[5] = 1.0;
        identity[10] = 1.0;
        identity[15] = 1.0;
        return identity;
    }
    determinant = 1.0 / determinant;
    for value in &mut inv {
        *value *= determinant;
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A GLB with a two-bone skin and a one-triangle mesh: the smallest file
    /// that exercises the container, the accessors and the skinning.
    fn sample() -> Vec<u8> {
        let mut bin: Vec<u8> = Vec::new();
        for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        for _ in 0..3 {
            for _ in 0..3 {
                bin.extend_from_slice(&0.0f32.to_le_bytes());
            }
        }
        let joints_at = bin.len();
        for joint in [0u16, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0] {
            bin.extend_from_slice(&joint.to_le_bytes());
        }
        let weights_at = bin.len();
        for weight in [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] {
            bin.extend_from_slice(&weight.to_le_bytes());
        }
        let indices_at = bin.len();
        for index in [0u16, 1, 2] {
            bin.extend_from_slice(&index.to_le_bytes());
        }
        let inverse_at = bin.len();
        for _ in 0..2 {
            let mut identity = [0.0f32; 16];
            identity[0] = 1.0;
            identity[5] = 1.0;
            identity[10] = 1.0;
            identity[15] = 1.0;
            for value in identity {
                bin.extend_from_slice(&value.to_le_bytes());
            }
        }
        let document = serde_json::json!({
            "asset": {"version": "2.0"},
            "nodes": [
                {"name": "Hips", "children": [1]},
                {"name": "LeftUpLeg"}
            ],
            "skins": [{"joints": [0, 1], "inverseBindMatrices": 5}],
            "meshes": [{"primitives": [{
                "attributes": {"POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3},
                "indices": 4
            }]}],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
                {"bufferView": 2, "componentType": 5123, "count": 3, "type": "VEC4"},
                {"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC4"},
                {"bufferView": 4, "componentType": 5123, "count": 3, "type": "SCALAR"},
                {"bufferView": 5, "componentType": 5126, "count": 2, "type": "MAT4"}
            ],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": 36},
                {"buffer": 0, "byteOffset": 36, "byteLength": 36},
                {"buffer": 0, "byteOffset": 72, "byteLength": 24},
                {"buffer": 0, "byteOffset": 96, "byteLength": 48},
                {"buffer": 0, "byteOffset": 144, "byteLength": 6},
                {"buffer": 0, "byteOffset": 150, "byteLength": 128}
            ],
            "buffers": [{"byteLength": 0}]
        });
        let _ = (joints_at, weights_at, indices_at, inverse_at);
        let json = serde_json::to_vec(&document).unwrap();
        // glTF pads every chunk to four bytes, and the chunk length counts the
        // padding: a length that does not would put the next chunk header off by
        // one to three bytes.
        let mut json = json;
        while json.len() % 4 != 0 {
            json.push(b' ');
        }
        let mut out = Vec::new();
        out.extend_from_slice(b"glTF");
        out.extend_from_slice(&2u32.to_le_bytes());
        let total = 12 + 8 + json.len() + 8 + bin.len();
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&(json.len() as u32).to_le_bytes());
        out.extend_from_slice(b"JSON");
        out.extend_from_slice(&json);
        out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        out.extend_from_slice(b"BIN\0");
        out.extend_from_slice(&bin);
        out
    }

    #[test]
    fn a_glb_skin_imports_with_its_bones_and_mesh() {
        let bytes = sample();
        let gltf = Gltf::parse(&bytes).expect("parse");
        let character = gltf.build("sample".into()).expect("build");
        assert_eq!(character.bones.len(), 2);
        assert_eq!(character.bones[0].name, "Hips");
        assert_eq!(character.bones[0].canonical, Some(0));
        assert_eq!(character.vertex_count(), 3);
        assert_eq!(character.faces.len(), 1);
        assert_eq!(character.matched(), 2);
    }

    /// A rigid pose change moves every weighted vertex with its bone.
    fn neutral() -> [f32; 9] {
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    }

    #[test]
    fn a_pinned_vertex_follows_a_rotated_bone() {
        let bytes = sample();
        let character = Gltf::parse(&bytes).unwrap().build("sample".into()).unwrap();
        let world = vec![neutral(); 52];
        let rest = vec![neutral(); 52];
        let posed = character.pose(&world, &rest, [0.0, 1.0, 0.0]);
        // Vertex 0 is fully weighted to Hips, whose bind position is the origin.
        assert!((posed[1] - 1.0).abs() < 1e-5, "{posed:?}");
        // The translation is global: a vertex weighted to the other bone moves
        // with the character too.
        assert!((posed[4] - 1.0).abs() < 1e-5, "{posed:?}");
    }
}
