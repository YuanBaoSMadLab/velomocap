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
    /// Nearest ancestor that is one of this character's bones. The hierarchy is
    /// what lets a bone's joint carry its children along.
    pub parent: Option<usize>,
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
    /// Bone indices, parents before children.
    pub order: Vec<usize>,
    /// Bind-pose vertices, xyz.
    pub rest: Vec<f32>,
    /// Four joint indices and weights per vertex.
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
    pub faces: Vec<[u32; 3]>,
    /// How much the bind space was scaled to bring the character to human size.
    pub scale: f32,
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
        let name = path
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_else(|| "character".to_string());
        let extension = path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        // A binary FBX is a container of its own; the glTF-family files (VRM is
        // GLB) share the reader below.
        if extension == "fbx" {
            let document = velomocap_core::fbx_read::parse(&bytes)
                .map_err(|error| anyhow::anyhow!(error))
                .with_context(|| format!("{} is not a readable binary FBX", path.display()))?;
            return Self::from_fbx(&document, name);
        }
        if extension == "pmx" {
            return Self::from_pmx(&bytes, name);
        }
        let gltf = Gltf::parse(&bytes)
            .with_context(|| format!("{} is not a readable glTF/GLB file", path.display()))?;
        gltf.build(name)
    }

    /// Build a character from a PMX (MMD) model.
    ///
    /// PMX has no bind matrices: the rest pose is the bone positions with an
    /// identity orientation, so each bone's bind is a translation and the skin
    /// factor is its inverse. That is exactly what MMD's own skinning does.
    fn from_pmx(bytes: &[u8], name: String) -> Result<Self> {
        let mut cursor = Cursor { data: bytes, offset: 0 };
        anyhow::ensure!(bytes.starts_with(b"PMX "), "not a PMX file");
        cursor.offset = 4;
        let _version = cursor.f32();
        let settings = cursor.u8() as usize;
        let settings_start = cursor.offset;
        let encoding = cursor.u8();
        let extra_uvs = cursor.u8() as usize;
        let vertex_index_size = cursor.u8() as usize;
        let _texture_index_size = cursor.u8() as usize;
        let _material_index_size = cursor.u8() as usize;
        let bone_index_size = cursor.u8() as usize;
        let _morph_index_size = cursor.u8() as usize;
        let _rigidbody_index_size = cursor.u8() as usize;
        cursor.offset = settings_start + settings;
        for _ in 0..4 {
            let _ = cursor.text(encoding);
        }

        let vertices = cursor.i32();
        anyhow::ensure!(vertices > 0, "the PMX has no vertices");
        let mut rest = Vec::with_capacity(vertices as usize * 3);
        let mut joints: Vec<[u16; 4]> = Vec::with_capacity(vertices as usize);
        let mut weights: Vec<[f32; 4]> = Vec::with_capacity(vertices as usize);
        for _ in 0..vertices {
            rest.push(cursor.f32());
            rest.push(cursor.f32());
            rest.push(cursor.f32());
            for _ in 0..3 {
                cursor.f32(); // normal
            }
            for _ in 0..2 {
                cursor.f32(); // uv
            }
            for _ in 0..extra_uvs * 4 {
                cursor.f32();
            }
            let weight_kind = cursor.u8();
            let mut vertex_joints = [0u16; 4];
            let mut vertex_weights = [0.0f32; 4];
            match weight_kind {
                0 => {
                    vertex_joints[0] = cursor.bone_index(bone_index_size) as u16;
                    vertex_weights[0] = 1.0;
                }
                1 => {
                    vertex_joints[0] = cursor.bone_index(bone_index_size) as u16;
                    vertex_joints[1] = cursor.bone_index(bone_index_size) as u16;
                    let first = cursor.f32();
                    vertex_weights[0] = first;
                    vertex_weights[1] = 1.0 - first;
                }
                2 | 4 => {
                    for slot in 0..4 {
                        vertex_joints[slot] = cursor.bone_index(bone_index_size) as u16;
                    }
                    for slot in 0..4 {
                        vertex_weights[slot] = cursor.f32();
                    }
                }
                _ => {
                    // SDEF: two bones, one weight, then C/R0/R1 spheres.
                    vertex_joints[0] = cursor.bone_index(bone_index_size) as u16;
                    vertex_joints[1] = cursor.bone_index(bone_index_size) as u16;
                    let first = cursor.f32();
                    vertex_weights[0] = first;
                    vertex_weights[1] = 1.0 - first;
                    for _ in 0..9 {
                        cursor.f32();
                    }
                }
            }
            cursor.f32(); // edge scale
            if !vertex_weights.iter().any(|weight| *weight > 0.0) {
                vertex_weights[0] = 1.0;
            }
            joints.push(vertex_joints);
            weights.push(vertex_weights);
        }

        let indices = cursor.i32();
        anyhow::ensure!(indices > 0, "the PMX has no faces");
        let mut faces = Vec::with_capacity(indices as usize / 3);
        for _ in 0..indices {
            let index = match vertex_index_size {
                1 => cursor.u8() as u32,
                2 => cursor.u16() as u32,
                _ => cursor.i32() as u32,
            };
            faces.push(index);
        }

        let textures = cursor.i32();
        for _ in 0..textures.max(0) {
            let _ = cursor.text(encoding);
        }
        let materials = cursor.i32();
        for _ in 0..materials.max(0) {
            for _ in 0..4 {
                let _ = cursor.text(encoding);
            }
            // Diffuse 4, specular 3, specular strength 1, ambient 3.
            for _ in 0..11 {
                cursor.f32();
            }
            cursor.u8(); // flags
            for _ in 0..5 {
                cursor.f32(); // edge colour and size
            }
            let _texture = cursor.index(_texture_index_size);
            let _sphere = cursor.index(_texture_index_size);
            cursor.u8(); // sphere mode
            if cursor.u8() == 0 {
                let _ = cursor.index(_texture_index_size);
            } else {
                cursor.u8();
            }
            let _ = cursor.text(encoding);
        }

        let bone_count = cursor.i32();
        anyhow::ensure!(bone_count > 0, "the PMX has no bones");
        let mut bones = Vec::with_capacity(bone_count as usize);
        for _ in 0..bone_count {
            let bone_name = cursor.text(encoding);
            let _english = cursor.text(encoding);
            let position = [cursor.f32(), cursor.f32(), cursor.f32()];
            let parent = cursor.index(bone_index_size);
            let _layer = cursor.i32();
            let flags = cursor.u16();
            if flags & 0x0001 != 0 {
                for _ in 0..3 {
                    cursor.f32();
                }
            }
            if flags & 0x0002 != 0 {
                let _ = cursor.index(bone_index_size);
            }
            if flags & 0x0100 != 0 {
                let _ = cursor.index(bone_index_size);
                cursor.f32();
            }
            if flags & 0x0400 != 0 {
                for _ in 0..3 {
                    cursor.f32();
                }
            }
            if flags & 0x0800 != 0 {
                for _ in 0..6 {
                    cursor.f32();
                }
            }
            if flags & 0x2000 != 0 {
                cursor.i32();
            }
            if flags & 0x0020 != 0 {
                let _ = cursor.index(bone_index_size);
                let loops = cursor.i32();
                cursor.f32();
                for _ in 0..loops.max(0) {
                    let _ = cursor.index(bone_index_size);
                    if cursor.u8() != 0 {
                        for _ in 0..3 {
                            cursor.f32();
                        }
                    }
                }
            }
            let mut bind_matrix = [0.0f32; 16];
            bind_matrix[0] = 1.0;
            bind_matrix[5] = 1.0;
            bind_matrix[10] = 1.0;
            bind_matrix[15] = 1.0;
            bind_matrix[3] = position[0];
            bind_matrix[7] = position[1];
            bind_matrix[11] = position[2];
            bones.push(Bone {
                canonical: velomocap_motion::retarget::canonicalize(&bone_name),
                name: bone_name,
                parent: if parent >= 0 && parent < bone_count {
                    Some(parent as usize)
                } else {
                    None
                },
                bind: position,
                bind_rot: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                inverse_bind: invert(&bind_matrix),
            });
        }
        anyhow::ensure!(
            faces.len() % 3 == 0,
            "the PMX face list is not triangles"
        );
        let faces: Vec<[u32; 3]> = faces
            .chunks_exact(3)
            .map(|face| [face[0], face[1], face[2]])
            .collect();
        Ok(finalize(name, bones, rest, joints, weights, faces))
    }

    /// Build a character from a parsed FBX: the skin deformer's clusters give the
    /// bones and their bind matrices, the geometry gives the mesh.
    fn from_fbx(
        document: &velomocap_core::fbx_read::Document,
        name: String,
    ) -> Result<Self> {
        use std::collections::HashMap;
        let objects = document.child("Objects").context("the FBX has no Objects node")?;
        // A model's name is `name\0\x01Class`.
        let mut model_names: HashMap<i64, String> = HashMap::new();
        for node in objects.children_named("Model") {
            let Some(id) = node.i64_at(0) else {
                continue;
            };
            let raw = node.str_at(1).unwrap_or("bone");
            model_names.insert(id, raw.split('\0').next().unwrap_or(raw).to_string());
        }

        let mut deformer_by_id: HashMap<i64, &velomocap_core::fbx_read::Node> = HashMap::new();
        let mut skin_id = None;
        for node in objects.children_named("Deformer") {
            let Some(id) = node.i64_at(0) else {
                continue;
            };
            deformer_by_id.insert(id, node);
            if node.str_at(2) == Some("Skin") {
                skin_id = Some(id);
            }
        }
        skin_id.context("the FBX has no skin deformer")?;

        let mut connections: Vec<(String, i64, i64)> = Vec::new();
        if let Some(node) = document.child("Connections") {
            for link in node.children_named("C") {
                let (Some(child), Some(parent)) = (link.i64_at(1), link.i64_at(2)) else {
                    continue;
                };
                connections.push((link.str_at(0).unwrap_or("").to_string(), child, parent));
            }
        }

        let mut geometry = None;
        for node in objects.children_named("Geometry") {
            let (Some(vertices), Some(polygons)) =
                (node.child("Vertices"), node.child("PolygonVertexIndex"))
            else {
                continue;
            };
            let positions = vertices
                .props
                .first()
                .map(|prop| prop.as_f64s().to_vec())
                .unwrap_or_default();
            let indices = polygons
                .props
                .first()
                .map(|prop| prop.as_i32s().to_vec())
                .unwrap_or_default();
            if positions.len() >= 3 && !indices.is_empty() {
                geometry = Some((positions, indices));
                break;
            }
        }
        let (positions, polygons) =
            geometry.context("the FBX has no mesh geometry (Vertices / PolygonVertexIndex)")?;

        // Each cluster is a bone: its linked Model names it, `TransformLink` is
        // the bind world matrix, and Indexes/Weights say which vertices it moves.
        let mut vertex_count = positions.len() / 3;
        let mut per_vertex: Vec<Vec<(usize, f32)>> = vec![Vec::new(); vertex_count];
        let mut bones: Vec<Bone> = Vec::new();
        let mut bone_models: Vec<i64> = Vec::new();
        for node in objects.children_named("Deformer") {
            if node.str_at(2) != Some("Cluster") {
                continue;
            }
            let Some(id) = node.i64_at(0) else {
                continue;
            };
            let Some(bone_id) = connections
                .iter()
                .find(|(kind, child, parent)| kind == "OO" && *child == id && model_names.contains_key(parent))
                .map(|(_, _, parent)| *parent)
            else {
                continue;
            };
            let link = node
                .child("TransformLink")
                .and_then(|link| link.props.first())
                .map(|prop| prop.as_f64s().to_vec())
                .unwrap_or_default();
            anyhow::ensure!(
                link.len() == 16,
                "cluster {} has no TransformLink matrix",
                model_names.get(&bone_id).cloned().unwrap_or_default()
            );
            let mut matrix = [0.0f32; 16];
            for (slot, value) in link.iter().enumerate() {
                matrix[slot] = *value as f32;
            }
            // FBX matrices are column-major.
            let bind_matrix = transpose4(&matrix);
            let bone_name = model_names.get(&bone_id).cloned().unwrap_or_default();
            let slot = bones.len();
            bone_models.push(bone_id);
            bones.push(Bone {
                canonical: velomocap_motion::retarget::canonicalize(&bone_name),
                name: bone_name,
                parent: None,
                bind: translation_of(&bind_matrix),
                bind_rot: rotation_of(&bind_matrix),
                inverse_bind: invert(&bind_matrix),
            });
            let indexes = node
                .child("Indexes")
                .and_then(|indexes| indexes.props.first())
                .map(|prop| prop.as_i32s().to_vec())
                .unwrap_or_default();
            let weights = node
                .child("Weights")
                .and_then(|weights| weights.props.first())
                .map(|prop| prop.as_f64s().to_vec())
                .unwrap_or_default();
            for (index, weight) in indexes.iter().zip(weights.iter()) {
                if let Some(list) = per_vertex.get_mut(*index as usize) {
                    list.push((slot, *weight as f32));
                }
            }
        }
        anyhow::ensure!(!bones.is_empty(), "the skin has no clusters");

        // The bone hierarchy is in the `OO` connections: each Model names the
        // Model it hangs from. Walk up until the parent is a bone of this skin.
        let mut slot_of: HashMap<i64, usize> = HashMap::new();
        for (slot, id) in bone_models.iter().enumerate() {
            slot_of.entry(*id).or_insert(slot);
        }
        for slot in 0..bones.len() {
            let mut child = bone_models[slot];
            for _ in 0..model_names.len().max(1) {
                let Some((_, _, next)) = connections
                    .iter()
                    .find(|(kind, id, _)| kind == "OO" && *id == child)
                else {
                    break;
                };
                if let Some(parent) = slot_of.get(next) {
                    if *parent != slot {
                        bones[slot].parent = Some(*parent);
                    }
                    break;
                }
                if !model_names.contains_key(next) {
                    break;
                }
                child = *next;
            }
        }

        let mut rest: Vec<f32> = Vec::with_capacity(vertex_count * 3);
        for value in &positions {
            rest.push(*value as f32);
        }
        // Top four influences per vertex, normalised like a glTF skin.
        let mut joints: Vec<[u16; 4]> = vec![[0; 4]; vertex_count];
        let mut weights: Vec<[f32; 4]> = vec![[0.0; 4]; vertex_count];
        for (vertex, list) in per_vertex.iter_mut().enumerate() {
            list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            list.truncate(4);
            let total: f32 = list.iter().map(|(_, weight)| *weight).sum();
            if total <= 0.0 {
                continue;
            }
            for (slot, (joint, weight)) in list.iter().enumerate() {
                joints[vertex][slot] = *joint as u16;
                weights[vertex][slot] = weight / total;
            }
        }
        // Polygons: the last index of each is bitwise-not'ed; fan-triangulate.
        let mut faces: Vec<[u32; 3]> = Vec::new();
        let mut corners: Vec<u32> = Vec::new();
        for index in &polygons {
            if *index < 0 {
                corners.push((!*index) as u32);
                for corner in 1..corners.len().saturating_sub(1) {
                    faces.push([corners[0], corners[corner], corners[corner + 1]]);
                }
                corners.clear();
            } else {
                corners.push(*index as u32);
            }
        }
        anyhow::ensure!(!faces.is_empty(), "the FBX geometry has no polygons");
        vertex_count = rest.len() / 3;
        joints.truncate(vertex_count);
        weights.truncate(vertex_count);
        Ok(finalize(name, bones, rest, joints, weights, faces))
    }

    /// Pose the mesh with the clip's joint rotations.
    ///
    /// `world` and `rest` are the clip's per-joint world rotations in **SMPL-H
    /// joint order**, which is the order the clip stores them in; `names` is the
    /// same order's bone names (`PreparedBvh::names`), which is what maps a
    /// character bone's canonical index onto the joint that drives it.
    /// `translation` moves the character with the root.
    pub fn pose(
        &self,
        world: &[[f32; 9]],
        rest: &[[f32; 9]],
        rest_positions: &[[f32; 3]],
        names: &[String],
        translation: [f32; 3],
    ) -> Vec<f32> {
        // A character's bones carry canonical indices (Hips, LeftUpLeg …), the
        // clip's rotations are in SMPL-H joint order. Resolve once per pose.
        let mut joint_of = [usize::MAX; 52];
        for (joint, name) in names.iter().enumerate() {
            if joint >= world.len() {
                break;
            }
            if let Some(canonical) = velomocap_motion::retarget::canonicalize(name) {
                if canonical < joint_of.len() && joint_of[canonical] == usize::MAX {
                    joint_of[canonical] = joint;
                }
            }
        }
        // How far the clip has turned each bone from its own rest pose.
        let mut delta = [identity(); 52];
        let mut mapped = [false; 52];
        for canonical in 0..52 {
            let joint = joint_of[canonical];
            if joint == usize::MAX || joint >= world.len() || joint >= rest.len() {
                continue;
            }
            delta[canonical] = multiply(&world[joint], &transpose(&rest[joint]));
            mapped[canonical] = true;
        }
        let swing = self.bind_swing(&joint_of, rest_positions);
        let ground = self.ground_offset(&joint_of, rest_positions);
        // Walk parents first: each bone's world rotation is the clip's own delta
        // for that joint, and its pivot follows its parent's motion. Without the
        // second half a raised arm would rotate about the bind elbow.
        let mut effective = vec![identity(); self.bones.len()];
        let mut position = vec![[0.0f32; 3]; self.bones.len()];
        let mut skin = vec![[0.0f32; 16]; self.bones.len()];
        for &index in &self.order {
            let bone = &self.bones[index];
            let carried = bone
                .parent
                .map(|parent| effective[parent])
                .unwrap_or_else(identity);
            let rotation = match bone.canonical {
                Some(canonical) if mapped[canonical] => {
                    multiply(&delta[canonical], &swing[index])
                }
                // A bone the clip cannot drive rides with its parent.
                _ => carried,
            };
            let at = match bone.parent {
                Some(parent) => {
                    let from = self.bones[parent].bind;
                    let mut at = position[parent];
                    for axis in 0..3 {
                        for axis_in in 0..3 {
                            at[axis] += carried[axis * 3 + axis_in]
                                * (bone.bind[axis_in] - from[axis_in]);
                        }
                    }
                    at
                }
                None => [
                    bone.bind[0] + translation[0],
                    bone.bind[1] + translation[1] + ground,
                    bone.bind[2] + translation[2],
                ],
            };
            effective[index] = rotation;
            position[index] = at;
            let rotation = multiply(&rotation, &bone.bind_rot);
            // M = [R | pivot] * inverse_bind
            let world_matrix = multiply4(
                &[
                    rotation[0], rotation[1], rotation[2], at[0],
                    rotation[3], rotation[4], rotation[5], at[1],
                    rotation[6], rotation[7], rotation[8], at[2],
                    0.0, 0.0, 0.0, 1.0,
                ],
                &bone.inverse_bind,
            );
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

    /// How far the root has to rise or fall for this character's feet to stand
    /// where the clip's feet stand.
    ///
    /// The root follows the clip's pelvis, so an avatar whose legs are three
    /// centimetres longer floats three centimetres in the air for the whole take.
    /// Both sides measure the same thing — hips down to the lower ankle — so this
    /// is a constant per character, not a per-frame correction that could jitter.
    fn ground_offset(&self, joint_of: &[usize; 52], rest_positions: &[[f32; 3]]) -> f32 {
        let clip_height = |canonical: usize| -> Option<f32> {
            let joint = *joint_of.get(canonical)?;
            if joint == usize::MAX {
                return None;
            }
            rest_positions.get(joint).map(|at| at[1])
        };
        let bind_height = |canonical: usize| -> Option<f32> {
            self.bones
                .iter()
                .find(|bone| bone.canonical == Some(canonical))
                .map(|bone| bone.bind[1])
        };
        let lowest = |heights: [Option<f32>; 2]| -> Option<f32> {
            let mut value = f32::INFINITY;
            for height in heights.into_iter().flatten() {
                value = value.min(height);
            }
            value.is_finite().then_some(value)
        };
        let (Some(clip_hips), Some(bind_hips)) = (clip_height(0), bind_height(0)) else {
            return 0.0;
        };
        let (Some(clip_ankle), Some(bind_ankle)) = (
            lowest([clip_height(46), clip_height(50)]),
            lowest([bind_height(46), bind_height(50)]),
        ) else {
            return 0.0;
        };
        // Longer legs mean the character stands too high on the clip's pelvis.
        (bind_hips - bind_ankle - (clip_hips - clip_ankle)).clamp(-0.15, 0.15)
    }

    /// The rotation that takes each bone from its bind direction to the clip's
    /// rest direction, so a character bound in an A-pose plays the clip's angles
    /// instead of keeping its own.
    ///
    /// A bone's direction is the segment to its first child, which is exactly
    /// what that joint's rotation swings; the clip's rest comes from the body
    /// model's neutral pose. Bones whose bind already matches, or whose angle is
    /// too large to be a bind-pose difference, are left alone.
    fn bind_swing(&self, joint_of: &[usize; 52], rest_positions: &[[f32; 3]]) -> Vec<[f32; 9]> {
        use velomocap_motion::retarget::PARENTS;
        let mut swing = vec![identity(); self.bones.len()];
        let mut primary = [usize::MAX; 52];
        for (child, parent) in PARENTS.iter().enumerate() {
            let Some(parent) = parent else {
                continue;
            };
            let parent = *parent as usize;
            if parent < primary.len() && primary[parent] == usize::MAX {
                primary[parent] = child;
            }
        }
        let mut slot_of = [usize::MAX; 52];
        for (slot, bone) in self.bones.iter().enumerate() {
            if let Some(canonical) = bone.canonical {
                if slot_of[canonical] == usize::MAX {
                    slot_of[canonical] = slot;
                }
            }
        }
        for (slot, bone) in self.bones.iter().enumerate() {
            let Some(canonical) = bone.canonical else {
                continue;
            };
            let Some(child) = primary.get(canonical).copied() else {
                continue;
            };
            if child == usize::MAX {
                continue;
            }
            let child_slot = slot_of.get(child).copied().unwrap_or(usize::MAX);
            if child_slot == usize::MAX {
                continue;
            }
            let (joint, child_joint) = (joint_of[canonical], joint_of[child]);
            if joint == usize::MAX || child_joint == usize::MAX {
                continue;
            }
            if joint >= rest_positions.len() || child_joint >= rest_positions.len() {
                continue;
            }
            let Some(bind_dir) = unit([
                self.bones[child_slot].bind[0] - bone.bind[0],
                self.bones[child_slot].bind[1] - bone.bind[1],
                self.bones[child_slot].bind[2] - bone.bind[2],
            ]) else {
                continue;
            };
            let Some(rest_dir) = unit([
                rest_positions[child_joint][0] - rest_positions[joint][0],
                rest_positions[child_joint][1] - rest_positions[joint][1],
                rest_positions[child_joint][2] - rest_positions[joint][2],
            ]) else {
                continue;
            };
            let cosine = bind_dir[0] * rest_dir[0]
                + bind_dir[1] * rest_dir[1]
                + bind_dir[2] * rest_dir[2];
            // Already the same direction: nothing to swing.
            if cosine > 0.9995 {
                continue;
            }
            // Past 110° this is a different rig convention, not a bind pose.
            if cosine < -0.342 {
                continue;
            }
            if let Some(rotation) = rotation_between(bind_dir, rest_dir) {
                swing[slot] = rotation;
            }
        }
        swing
    }
}

/// A little-endian cursor over a PMX byte stream. PMX has no alignment, so every
/// field is read in order; a truncated file yields zeros rather than panicking,
/// and the caller's `anyhow::ensure!` checks catch what that produces.
struct Cursor<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, count: usize) -> &'a [u8] {
        let end = (self.offset + count).min(self.data.len());
        let slice = &self.data[self.offset.min(end)..end];
        self.offset = end;
        slice
    }

    fn u8(&mut self) -> u8 {
        self.take(1).first().copied().unwrap_or(0)
    }

    fn u16(&mut self) -> u16 {
        let bytes = self.take(2);
        if bytes.len() < 2 {
            return 0;
        }
        u16::from_le_bytes([bytes[0], bytes[1]])
    }

    fn i32(&mut self) -> i32 {
        let bytes = self.take(4);
        if bytes.len() < 4 {
            return 0;
        }
        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    fn f32(&mut self) -> f32 {
        f32::from_bits(self.i32() as u32)
    }

    /// A PMX index of the given width. Width 1 is unsigned, 2 and 4 are signed;
    /// -1 means "none".
    fn index(&mut self, width: usize) -> i32 {
        match width {
            1 => self.u8() as i32,
            2 => {
                let bytes = self.take(2);
                if bytes.len() < 2 {
                    return -1;
                }
                i16::from_le_bytes([bytes[0], bytes[1]]) as i32
            }
            _ => self.i32(),
        }
    }

    /// A bone index, where width 1 uses 0xFF for "none".
    fn bone_index(&mut self, width: usize) -> i32 {
        if width == 1 {
            let value = self.u8();
            if value == 0xFF {
                return -1;
            }
            value as i32
        } else {
            self.index(width)
        }
    }

    /// A length-prefixed string, UTF-16LE (encoding 0) or UTF-8 (1).
    fn text(&mut self, encoding: u8) -> String {
        let length = self.i32();
        if length <= 0 {
            return String::new();
        }
        let bytes = self.take(length as usize);
        if encoding == 0 {
            let units: Vec<u16> = bytes
                .chunks_exact(2)
                .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
                .collect();
            String::from_utf16_lossy(&units)
        } else {
            String::from_utf8_lossy(bytes).to_string()
        }
    }
}

/// Bring an imported character to human size and wrap it up.
///
/// Characters arrive at very different sizes — Mixamo's GLB is 2.2× life size
/// (3.78 m tall), an Unreal mannequin is exactly 1.8 m, a Mixamo FBX is in
/// centimetres — so the bind space is scaled, not the posed output: the root
/// translation the clip supplies has to stay in metres either way.
fn finalize(
    name: String,
    mut bones: Vec<Bone>,
    mut rest: Vec<f32>,
    joints: Vec<[u16; 4]>,
    weights: Vec<[f32; 4]>,
    faces: Vec<[u32; 3]>,
) -> Character {
    let mut height_min = f32::INFINITY;
    let mut height_max = f32::NEG_INFINITY;
    for vertex in rest.chunks_exact(3) {
        height_min = height_min.min(vertex[1]);
        height_max = height_max.max(vertex[1]);
    }
    let height = height_max - height_min;
    let mut scale = if height > 0.02 { 1.75 / height } else { 1.0 };
    if (scale - 1.0).abs() <= 0.15 {
        scale = 1.0;
    }
    if scale != 1.0 {
        for value in &mut rest {
            *value *= scale;
        }
        for bone in &mut bones {
            bone.bind = [
                bone.bind[0] * scale,
                bone.bind[1] * scale,
                bone.bind[2] * scale,
            ];
            let m = &bone.bind_rot;
            let bind_matrix = [
                m[0], m[1], m[2], bone.bind[0],
                m[3], m[4], m[5], bone.bind[1],
                m[6], m[7], m[8], bone.bind[2],
                0.0, 0.0, 0.0, 1.0,
            ];
            bone.inverse_bind = invert(&bind_matrix);
        }
    }
    let order = hierarchy_order(&bones);
    Character {
        name,
        bones,
        order,
        rest,
        joints,
        weights,
        faces,
        scale,
    }
}

/// Parents before children, so one pass can carry each joint's motion down the
/// chain. A malformed file with a cycle keeps the remaining bones in file order.
fn hierarchy_order(bones: &[Bone]) -> Vec<usize> {
    let mut order = Vec::with_capacity(bones.len());
    let mut placed = vec![false; bones.len()];
    while order.len() < bones.len() {
        let before = order.len();
        for (index, bone) in bones.iter().enumerate() {
            if placed[index] {
                continue;
            }
            let ready = match bone.parent {
                Some(parent) => parent < bones.len() && placed[parent],
                None => true,
            };
            if ready {
                placed[index] = true;
                order.push(index);
            }
        }
        if order.len() == before {
            for (index, done) in placed.iter_mut().enumerate() {
                if !*done {
                    *done = true;
                    order.push(index);
                }
            }
        }
    }
    order
}

fn transpose(m: &[f32; 9]) -> [f32; 9] {    [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]
}

fn identity() -> [f32; 9] {
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
}

fn unit(v: [f32; 3]) -> Option<[f32; 3]> {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if length < 1e-6 {
        return None;
    }
    Some([v[0] / length, v[1] / length, v[2] / length])
}

/// The smallest rotation that takes `from` onto `to`; both must be unit vectors.
fn rotation_between(from: [f32; 3], to: [f32; 3]) -> Option<[f32; 9]> {
    let cross = [
        from[1] * to[2] - from[2] * to[1],
        from[2] * to[0] - from[0] * to[2],
        from[0] * to[1] - from[1] * to[0],
    ];
    let sine = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    let cosine = from[0] * to[0] + from[1] * to[1] + from[2] * to[2];
    if sine < 1e-6 {
        // Parallel, or exactly opposite with no axis to pick.
        return if cosine > 0.0 { Some(identity()) } else { None };
    }
    let angle = sine.atan2(cosine);
    let axis = [cross[0] / sine, cross[1] / sine, cross[2] / sine];
    let (sine, cosine) = angle.sin_cos();
    let t = 1.0 - cosine;
    let [x, y, z] = axis;
    Some([
        t * x * x + cosine, t * x * y - sine * z, t * x * z + sine * y,
        t * x * y + sine * z, t * y * y + cosine, t * y * z - sine * x,
        t * x * z - sine * y, t * y * z + sine * x, t * z * z + cosine,
    ])
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
                parent: None,
                bind: translation_of(&bind_matrix),
                bind_rot: rotation_of(&bind_matrix),
                inverse_bind,
            });
        }

        // The joint hierarchy is the node tree: a joint's parent is its nearest
        // ancestor that is also a joint of this skin.
        let mut node_parent: Vec<Option<usize>> = vec![None; nodes.len()];
        for (index, node) in nodes.iter().enumerate() {
            if let Some(children) = node.get("children").and_then(|value| value.as_array()) {
                for child in children {
                    if let Some(child) = child.as_u64() {
                        if let Some(slot) = node_parent.get_mut(child as usize) {
                            *slot = Some(index);
                        }
                    }
                }
            }
        }
        let mut slot_of_node: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for (slot, node) in joint_nodes.iter().enumerate() {
            slot_of_node.entry(*node).or_insert(slot);
        }
        for (slot, node) in joint_nodes.iter().enumerate() {
            let mut current = node_parent.get(*node).copied().flatten();
            for _ in 0..nodes.len().max(1) {
                let Some(parent) = current else {
                    break;
                };
                if let Some(found) = slot_of_node.get(&parent) {
                    if *found != slot {
                        bones[slot].parent = Some(*found);
                    }
                    break;
                }
                current = node_parent.get(parent).copied().flatten();
            }
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
        Ok(finalize(name, bones, rest, joints, weights, faces))
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
        let (accessor_value, data, offset, count, stride) = self.view(accessor)?;
        // Real files store the weights as normalised unsigned shorts or bytes to
        // save space (Mixamo and UE5 exports both do), so the component type
        // decides the width and the scale. Assuming floats read half the
        // vertices and made the counts disagree with POSITION.
        let kind = accessor_value
            .get("componentType")
            .and_then(|v| v.as_u64())
            .unwrap_or(5126);
        let (width, scale) = match kind {
            5121 => (1usize, 1.0 / 255.0),
            5123 => (2, 1.0 / 65535.0),
            _ => (4, 1.0),
        };
        let stride = if stride == 0 { width * 4 } else { stride };
        let mut out = Vec::with_capacity(count);
        for index in 0..count {
            let at = offset + index * stride;
            if at + width * 4 > data.len() {
                break;
            }
            let mut weights = [0.0f32; 4];
            for slot in 0..4 {
                let start = at + slot * width;
                weights[slot] = match width {
                    1 => data[start] as f32 * scale,
                    2 => u16::from_le_bytes([data[start], data[start + 1]]) as f32 * scale,
                    _ => f32::from_le_bytes([
                        data[start],
                        data[start + 1],
                        data[start + 2],
                        data[start + 3],
                    ]),
                };
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

    /// Loads a real character when `VELOMOCAP_CHARACTER` points at one, so the
    /// reader can be checked against the avatars people actually have.
    #[test]
    fn a_real_character_file_loads() {
        let Some(path) = std::env::var_os("VELOMOCAP_CHARACTER") else {
            return;
        };
        let character = Character::load(std::path::Path::new(&path)).expect("load");
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for vertex in character.rest.chunks_exact(3) {
            lo = lo.min(vertex[1]);
            hi = hi.max(vertex[1]);
        }
        eprintln!(
            "{}: {} bones ({} matched), {} vertices, {} faces, height {:.3} m, feet at {:.3} m, unmatched {:?}",
            character.name,
            character.bones.len(),
            character.matched(),
            character.vertex_count(),
            character.faces.len(),
            hi - lo,
            lo,
            character.unmatched()
        );
        // What the pose fix depends on: a hierarchy to carry motion down, and
        // the limb angles it has to swing away. A T-pose arm is level, an A-pose
        // arm sits tens of degrees below it, and that gap is what used to be
        // added on top of the clip's own rotation.
        let rooted = character.bones.iter().filter(|bone| bone.parent.is_some()).count();
        eprintln!(
            "{}: {rooted}/{} bones hang from a parent",
            character.name,
            character.bones.len()
        );
        let mut slot_of = [usize::MAX; 52];
        for (slot, bone) in character.bones.iter().enumerate() {
            if let Some(canonical) = bone.canonical {
                if slot_of[canonical] == usize::MAX {
                    slot_of[canonical] = slot;
                }
            }
        }
        for (label, parent, child) in [
            ("LeftArm", 7, 8),
            ("RightArm", 26, 27),
            ("LeftUpLeg", 44, 45),
            ("RightUpLeg", 48, 49),
        ] {
            let (parent, child) = (slot_of[parent], slot_of[child]);
            if parent == usize::MAX || child == usize::MAX {
                continue;
            }
            let from = character.bones[parent].bind;
            let to = character.bones[child].bind;
            let offset = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
            let length = (offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2]).sqrt();
            if length < 1e-6 {
                continue;
            }
            let below = (offset[1] / length).asin().to_degrees();
            eprintln!(
                "  {label}: {:.0}° below level, {:.3} m, from {:?} ({}) to {:?} ({})",
                below.abs(),
                length,
                from,
                character.bones[parent].name,
                to,
                character.bones[child].name,
            );
        }
        assert!(character.vertex_count() > 0);
    }

    /// How far the model's legs sit from the clip's own leg length, which is what
    /// a character's feet would float or sink by if the root only followed the
    /// clip's pelvis.
    #[test]
    fn a_real_character_measures_its_legs() {
        let (Some(path), Some(model)) = (
            std::env::var_os("VELOMOCAP_CHARACTER"),
            std::env::var_os("VELOMOCAP_SMPLH_MODEL"),
        ) else {
            return;
        };
        let character = Character::load(std::path::Path::new(&path)).expect("load");
        let body = velomocap_body::BodyModel::load(std::path::Path::new(&model), 16).expect("body");
        let skeleton = body.forward_joints(&velomocap_body::Pose::neutral(body.num_joints, 16));
        let at = |name: &str| {
            velomocap_body::JOINT_NAMES
                .iter()
                .position(|joint| *joint == name)
                .map(|joint| skeleton.positions[joint * 3 + 1])
        };
        let (Some(hips), Some(left), Some(right)) =
            (at("pelvis"), at("left_ankle"), at("right_ankle"))
        else {
            return;
        };
        let clip_legs = hips - left.min(right);
        let bind = |canonical: usize| {
            character
                .bones
                .iter()
                .find(|bone| bone.canonical == Some(canonical))
                .map(|bone| bone.bind[1])
        };
        let (Some(char_hips), Some(char_left), Some(char_right)) =
            (bind(0), bind(46), bind(50))
        else {
            return;
        };
        let character_legs = char_hips - char_left.min(char_right);
        let mut lo = f32::INFINITY;
        for vertex in character.rest.chunks_exact(3) {
            lo = lo.min(vertex[1]);
        }
        // What the pose would shift the root by, with the clip's own ankles.
        let mut joint_of = [usize::MAX; 52];
        for (canonical, name) in [(0, "pelvis"), (46, "left_ankle"), (50, "right_ankle")] {
            joint_of[canonical] = velomocap_body::JOINT_NAMES
                .iter()
                .position(|joint| *joint == name)
                .unwrap_or(usize::MAX);
        }
        let mut positions = vec![[0.0f32; 3]; skeleton.positions.len() / 3];
        for (joint, at) in positions.iter_mut().enumerate() {
            *at = [
                skeleton.positions[joint * 3],
                skeleton.positions[joint * 3 + 1],
                skeleton.positions[joint * 3 + 2],
            ];
        }
        eprintln!(
            "{}: hips {:.3} m, ankle {:.3}/{:.3} m, legs {:.3} m vs clip {:.3} m ({}), mesh floor {:.3} m, root shift {:+.3} m",
            character.name,
            char_hips,
            char_left,
            char_right,
            character_legs,
            clip_legs,
            if character_legs < clip_legs { "shorter" } else { "longer" },
            lo,
            character.ground_offset(&joint_of, &positions),
        );
    }

    /// The same real file, posed with a clip that has not moved: every joint's
    /// delta is the identity, so the skinning has to put every vertex back
    /// exactly where the bind pose had it. Anything the hierarchy, the pivots or
    /// the swing gets wrong shows up as the mesh sliding.
    #[test]
    fn a_real_character_poses_without_drifting() {
        let Some(path) = std::env::var_os("VELOMOCAP_CHARACTER") else {
            return;
        };
        let character = Character::load(std::path::Path::new(&path)).expect("load");
        let mut names = vec![String::new(); 52];
        for (bone, name) in names.iter_mut().enumerate() {
            *name = velomocap_motion::retarget::BONES.get(bone).unwrap_or(&"").to_string();
        }
        let world = vec![neutral(); 52];
        let rest = vec![neutral(); 52];
        let posed = character.pose(&world, &rest, &vec![[0.0f32; 3]; 52], &names, [0.0; 3]);
        let mut worst = 0.0f32;
        for (rest, posed) in character.rest.chunks_exact(3).zip(posed.chunks_exact(3)) {
            for axis in 0..3 {
                worst = worst.max((rest[axis] - posed[axis]).abs());
            }
        }
        eprintln!("{}: bind pose drifts {:.6} m", character.name, worst);
        assert!(worst < 0.005, "{worst} m");
    }

    /// The FBX path: a skin with two clusters, a quad, and the connections that
    /// tie a cluster to its bone. Everything below is what a Mixamo export has,
    /// minus the hundreds of nodes around it.
    fn fbx_sample() -> velomocap_core::fbx_read::Document {
        use velomocap_core::fbx_read::{Document, Node, Prop};
        let identity = Prop::F64s(vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        let cluster = |id: i64, indexes: Vec<i32>, weights: Vec<f64>| Node {
            name: "Deformer".into(),
            props: vec![
                Prop::I64(id),
                Prop::Str("Cluster".into()),
                Prop::Str("Cluster".into()),
            ],
            children: vec![
                Node {
                    name: "Indexes".into(),
                    props: vec![Prop::I32s(indexes)],
                    children: vec![],
                },
                Node {
                    name: "Weights".into(),
                    props: vec![Prop::F64s(weights)],
                    children: vec![],
                },
                Node {
                    name: "TransformLink".into(),
                    props: vec![identity.clone()],
                    children: vec![],
                },
            ],
        };
        let objects = Node {
            name: "Objects".into(),
            props: vec![],
            children: vec![
                Node {
                    name: "Model".into(),
                    props: vec![
                        Prop::I64(1),
                        Prop::Str("Hips\0\u{1}Model".into()),
                        Prop::Str("Null".into()),
                    ],
                    children: vec![],
                },
                Node {
                    name: "Model".into(),
                    props: vec![
                        Prop::I64(2),
                        Prop::Str("LeftUpLeg\0\u{1}Model".into()),
                        Prop::Str("Null".into()),
                    ],
                    children: vec![],
                },
                Node {
                    name: "Geometry".into(),
                    props: vec![
                        Prop::I64(10),
                        Prop::Str("Body\0\u{1}Geometry".into()),
                        Prop::Str("Mesh".into()),
                    ],
                    children: vec![
                        Node {
                            name: "Vertices".into(),
                            props: vec![Prop::F64s(vec![
                                0.0, 0.0, 0.0, 0.0, 1.75, 0.0, 0.5, 0.0, 0.0, 0.5, 1.75, 0.0,
                            ])],
                            children: vec![],
                        },
                        Node {
                            name: "PolygonVertexIndex".into(),
                            props: vec![Prop::I32s(vec![0, 1, 3, -3])],
                            children: vec![],
                        },
                    ],
                },
                Node {
                    name: "Deformer".into(),
                    props: vec![
                        Prop::I64(20),
                        Prop::Str("Skin".into()),
                        Prop::Str("Skin".into()),
                    ],
                    children: vec![],
                },
                cluster(21, vec![0, 1], vec![1.0, 1.0]),
                cluster(22, vec![0, 2, 3], vec![1.0, 1.0, 1.0]),
            ],
        };
        let link = |child: i64, parent: i64| Node {
            name: "C".into(),
            props: vec![
                Prop::Str("OO".into()),
                Prop::I64(child),
                Prop::I64(parent),
            ],
            children: vec![],
        };
        let connections = Node {
            name: "Connections".into(),
            props: vec![],
            children: vec![
                link(21, 20),
                link(21, 1),
                link(22, 20),
                link(22, 2),
            ],
        };
        Document {
            version: 7400,
            nodes: vec![objects, connections],
        }
    }

    #[test]
    fn an_fbx_skin_imports_with_its_clusters() {
        let character =
            Character::from_fbx(&fbx_sample(), "fbx".into()).expect("from_fbx");
        assert_eq!(character.bones.len(), 2);
        assert_eq!(character.bones[0].name, "Hips");
        assert_eq!(character.bones[0].canonical, Some(0));
        assert_eq!(character.bones[1].name, "LeftUpLeg");
        assert_eq!(
            character.bones[1].canonical,
            velomocap_motion::retarget::canonicalize("LeftUpLeg")
        );
        assert_eq!(character.vertex_count(), 4);
        assert_eq!(character.faces.len(), 2, "the quad fans into two triangles");
        // Vertex 0 is driven by both clusters, so its weights are normalised.
        let weights = character.weights[0];
        assert!((weights[0] - 0.5).abs() < 1e-5, "{weights:?}");
        assert!((weights[1] - 0.5).abs() < 1e-5, "{weights:?}");
        // Already 1.75 m tall, so nothing is scaled.
        assert!((character.scale - 1.0).abs() < 1e-6);
    }

    /// A minimal PMX: two bones, four vertices (BDEF1), two triangles.
    fn pmx_sample() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"PMX ");
        out.extend_from_slice(&2.0f32.to_le_bytes());
        out.push(8u8);
        for setting in [1u8, 0, 4, 4, 4, 4, 4, 4] {
            out.push(setting);
        }
        for text in ["model", "model_en", "comment", "comment_en"] {
            out.extend_from_slice(&(text.len() as i32).to_le_bytes());
            out.extend_from_slice(text.as_bytes());
        }
        let positions = [
            [0.0f32, 0.0, 0.0],
            [0.0, 1.75, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 1.75, 0.0],
        ];
        out.extend_from_slice(&(positions.len() as i32).to_le_bytes());
        for position in positions {
            for value in position {
                out.extend_from_slice(&value.to_le_bytes());
            }
            for _ in 0..3 {
                out.extend_from_slice(&0.0f32.to_le_bytes()); // normal
            }
            for _ in 0..2 {
                out.extend_from_slice(&0.0f32.to_le_bytes()); // uv
            }
            out.push(0); // BDEF1
            out.extend_from_slice(&0i32.to_le_bytes()); // bone 0
            out.extend_from_slice(&1.0f32.to_le_bytes()); // edge scale
        }
        let indices = [0i32, 1, 3, 0, 3, 2];
        out.extend_from_slice(&(indices.len() as i32).to_le_bytes());
        for index in indices {
            out.extend_from_slice(&index.to_le_bytes());
        }
        out.extend_from_slice(&0i32.to_le_bytes()); // textures
        out.extend_from_slice(&0i32.to_le_bytes()); // materials
        out.extend_from_slice(&2i32.to_le_bytes()); // bones
        let mut bone = |out: &mut Vec<u8>, name: &str, position: [f32; 3], parent: i32| {
            for text in [name, ""] {
                out.extend_from_slice(&(text.len() as i32).to_le_bytes());
                out.extend_from_slice(text.as_bytes());
            }
            for value in position {
                out.extend_from_slice(&value.to_le_bytes());
            }
            out.extend_from_slice(&parent.to_le_bytes());
            out.extend_from_slice(&0i32.to_le_bytes()); // layer
            out.extend_from_slice(&0u16.to_le_bytes()); // flags
        };
        bone(&mut out, "Hips", [0.0, 1.0, 0.0], -1);
        bone(&mut out, "LeftUpLeg", [0.1, 1.0, 0.0], 0);
        out
    }

    #[test]
    fn a_pmx_model_imports_with_its_bones_and_faces() {
        let character =
            Character::from_pmx(&pmx_sample(), "pmx".into()).expect("from_pmx");
        assert_eq!(character.bones.len(), 2);
        assert_eq!(character.bones[0].name, "Hips");
        assert_eq!(character.bones[0].canonical, Some(0));
        assert_eq!(
            character.bones[1].canonical,
            velomocap_motion::retarget::canonicalize("LeftUpLeg")
        );
        assert_eq!(character.vertex_count(), 4);
        assert_eq!(character.faces.len(), 2);
        assert!((character.scale - 1.0).abs() < 1e-6);
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
        let rest_positions = vec![[0.0f32; 3]; 52];
        // Only Hips (canonical 0) is named, and it is SMPL-H joint 0.
        let mut names = vec![String::new(); 52];
        names[0] = "Hips".to_string();
        let posed = character.pose(&world, &rest, &rest_positions, &names, [0.0, 1.0, 0.0]);
        // Vertex 0 is fully weighted to Hips, whose bind position is the origin.
        assert!((posed[1] - 1.0).abs() < 1e-5, "{posed:?}");
        // The translation is global: a vertex weighted to the other bone moves
        // with the character too.
        assert!((posed[4] - 1.0).abs() < 1e-5, "{posed:?}");
    }

    /// A hand-built character: `bind` sits where the bone is in the file, the
    /// inverse bind undoes it.
    fn bound(name: &str, canonical: usize, parent: Option<usize>, bind: [f32; 3]) -> Bone {
        let matrix = [
            1.0, 0.0, 0.0, bind[0],
            0.0, 1.0, 0.0, bind[1],
            0.0, 0.0, 1.0, bind[2],
            0.0, 0.0, 0.0, 1.0,
        ];
        Bone {
            name: name.to_string(),
            canonical: Some(canonical),
            parent,
            bind,
            bind_rot: identity(),
            inverse_bind: invert(&matrix),
        }
    }

    fn handmade(bones: Vec<Bone>, rest: Vec<f32>) -> Character {
        let order = hierarchy_order(&bones);
        let count = rest.len() / 3;
        Character {
            name: "handmade".to_string(),
            bones,
            order,
            rest,
            joints: vec![[0, 0, 0, 0]; count],
            weights: vec![[1.0, 0.0, 0.0, 0.0]; count],
            faces: Vec::new(),
            scale: 1.0,
        }
    }

    /// A joint's pivot must travel with its parent: when the hips turn, the leg
    /// swings around the hips, not around the place the leg was bound at.
    #[test]
    fn a_child_joint_swings_around_its_parent() {
        let bones = vec![
            bound("Hips", 0, None, [0.0, 0.0, 0.0]),
            bound("LeftUpLeg", 44, Some(0), [0.0, -1.0, 0.0]),
        ];
        // One vertex a metre below the leg joint, fully weighted to the leg.
        let mut character = handmade(bones, vec![0.0, -2.0, 0.0]);
        character.joints[0][0] = 1;
        // The hips and the leg both turn a quarter turn about Z.
        let quarter = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let mut world = vec![neutral(); 52];
        let mut rest = vec![neutral(); 52];
        world[0] = quarter;
        world[44] = quarter;
        let mut names = vec![String::new(); 52];
        names[0] = "Hips".to_string();
        names[44] = "LeftUpLeg".to_string();
        let posed = character.pose(
            &world,
            &rest,
            &vec![[0.0f32; 3]; 52],
            &names,
            [0.0, 0.0, 0.0],
        );
        // Hips pivot (0,0,0), leg pivot rotated to (1,0,0), the vertex a metre
        // further along that rotated leg.
        assert!((posed[0] - 2.0).abs() < 1e-5, "{posed:?}");
        assert!(posed[1].abs() < 1e-5, "{posed:?}");
    }

    /// A character bound in an A-pose must play the clip's own rest angles: the
    /// bind arm's 45 degrees are swung away before the clip is applied.
    #[test]
    fn an_a_pose_arm_is_swung_onto_the_clip_rest() {
        let down = std::f32::consts::FRAC_1_SQRT_2;
        let bones = vec![
            bound("LeftArm", 7, None, [0.0, 0.0, 0.0]),
            bound("LeftForeArm", 8, Some(0), [down, -down, 0.0]),
        ];
        let mut character = handmade(bones, vec![down, -down, 0.0]);
        character.joints[0][0] = 1;
        let world = vec![neutral(); 52];
        let rest = vec![neutral(); 52];
        // The clip's rest holds the arm out horizontally.
        let mut rest_positions = vec![[0.0f32; 3]; 52];
        rest_positions[8] = [1.0, 0.0, 0.0];
        let mut names = vec![String::new(); 52];
        names[7] = "LeftArm".to_string();
        names[8] = "LeftForeArm".to_string();
        let posed = character.pose(&world, &rest, &rest_positions, &names, [0.0; 3]);
        // A quarter turn out of the bind, so the forearm joint ends up level.
        assert!((posed[0] - 1.0).abs() < 1e-5, "{posed:?}");
        assert!(posed[1].abs() < 1e-5, "{posed:?}");
    }

    /// Legs longer than the clip's must not lift the character off the floor.
    #[test]
    fn longer_legs_stand_on_the_clip_floor() {
        let bones = vec![
            bound("Hips", 0, None, [0.0, 0.95, 0.0]),
            bound("LeftFoot", 46, Some(0), [0.0, 0.05, 0.0]),
        ];
        let mut character = handmade(bones, vec![0.0, 0.05, 0.0]);
        character.joints[0][0] = 1;
        let world = vec![neutral(); 52];
        let rest = vec![neutral(); 52];
        // The clip's rest leaves its ankle 0.078 m above the floor.
        let mut rest_positions = vec![[0.0f32; 3]; 52];
        rest_positions[0] = [0.0, 0.95, 0.0];
        rest_positions[46] = [0.0, 0.078, 0.0];
        let mut names = vec![String::new(); 52];
        names[0] = "Hips".to_string();
        names[46] = "LeftFoot".to_string();
        let posed = character.pose(&world, &rest, &rest_positions, &names, [0.0; 3]);
        assert!((posed[1] - 0.078).abs() < 1e-5, "{posed:?}");
    }
}
