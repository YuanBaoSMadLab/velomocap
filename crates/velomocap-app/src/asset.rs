//! Write a character back out as a glTF binary (.glb): mesh, skin, materials and
//! the animation the viewport is playing.
//!
//! This is the export path the reference does not have: it writes BVH and FBX
//! *motion* only, so the model a user picked has to be re-skinned by hand in
//! Blender before it can go into a game. Here the model that is on screen is the
//! model that lands in the file — positions, normals, texture coordinates, skin
//! weights, the skeleton it hangs from and one animation track per bone.
//!
//! glTF was chosen first because a single `.glb` carries its textures inside it:
//! no sidecar folder to lose, and Unity, Unreal, Blender and Godot all read it.

use anyhow::{Context, Result};
use std::path::Path;

use crate::character::Character;

/// One frame of the clip, already posed onto this character.
pub struct Frame {
    /// Bone world rotations, row-major 3×3, one per bone.
    pub rotations: Vec<[f32; 9]>,
    /// Bone world positions, one per bone.
    pub positions: Vec<[f32; 3]>,
    /// Root translation for the frame.
    pub root: [f32; 3],
}

/// What a caller can leave out of the file.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub skin: bool,
    pub materials: bool,
    pub animation: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            skin: true,
            materials: true,
            animation: true,
        }
    }
}

fn align4(len: usize) -> usize {
    (4 - len % 4) % 4
}

/// A growable binary buffer that remembers where each block starts.
#[derive(Default)]
struct Bin {
    bytes: Vec<u8>,
    views: Vec<serde_json::Value>,
}

impl Bin {
    fn push(&mut self, data: &[u8], target: Option<&str>) -> usize {
        let padding = align4(self.bytes.len());
        self.bytes.extend(std::iter::repeat_n(0u8, padding));
        let offset = self.bytes.len();
        self.bytes.extend_from_slice(data);
        let mut view = serde_json::json!({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": data.len(),
        });
        if let Some(target) = target {
            view["target"] = serde_json::json!(target);
        }
        self.views.push(view);
        self.views.len() - 1
    }

    fn floats(&mut self, values: &[f32], target: Option<&str>) -> usize {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        self.push(&bytes, target)
    }
}

/// Write `character` as a `.glb`, animated by `frames`.
pub fn write_glb(
    character: &Character,
    frames: &[Frame],
    options: Options,
    out: &Path,
) -> Result<()> {
    let bones = &character.bones;
    anyhow::ensure!(!character.faces.is_empty(), "the model has no faces");

    let mut bin = Bin::default();
    let mut accessors: Vec<serde_json::Value> = Vec::new();

    // --- geometry ---------------------------------------------------------
    // Normals are not stored by the readers, and a lit game engine needs them.
    let count = character.vertex_count();
    let mut normals = vec![0.0f32; count * 3];
    for face in &character.faces {
        let a = vertex_of(&character.rest, face[0]);
        let b = vertex_of(&character.rest, face[1]);
        let c = vertex_of(&character.rest, face[2]);
        let normal = cross(sub(b, a), sub(c, a));
        for index in face {
            let at = *index as usize * 3;
            for axis in 0..3 {
                normals[at + axis] += normal[axis];
            }
        }
    }
    for value in normals.chunks_exact_mut(3) {
        let length = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
        if length > 1e-9 {
            for axis in 0..3 {
                value[axis] /= length;
            }
        }
    }

    // --- materials and textures ------------------------------------------
    let mut materials: Vec<serde_json::Value> = Vec::new();
    let mut gltf_textures: Vec<serde_json::Value> = Vec::new();
    let mut images: Vec<serde_json::Value> = Vec::new();
    if options.materials {
        for (slot, texture) in character.textures.iter().enumerate() {
            let mut png = Vec::new();
            {
                let encoder = image::codecs::png::PngEncoder::new(&mut png);
                image::ImageEncoder::write_image(
                    encoder,
                    &texture.rgba,
                    texture.width as u32,
                    texture.height as u32,
                    image::ExtendedColorType::Rgba8,
                )
                .context("encoding a texture")?;
            }
            let view = bin.push(&png, None);
            images.push(serde_json::json!({
                "bufferView": view,
                "mimeType": "image/png",
                "name": format!("texture{slot}"),
            }));
            gltf_textures.push(serde_json::json!({ "sampler": 0, "source": images.len() - 1 }));
            materials.push(serde_json::json!({
                "name": format!("material{slot}"),
                "pbrMetallicRoughness": {
                    "baseColorTexture": { "index": gltf_textures.len() - 1 },
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.9,
                },
                "doubleSided": true,
            }));
        }
    }

    // One primitive per material, each carrying only the vertices it uses: a
    // single shared vertex array would be written once per primitive, and a
    // six-material avatar came out six times the size.
    let base_normals = normals;
    let mut primitives: Vec<serde_json::Value> = Vec::new();
    let mut slots: Vec<u16> = character
        .face_texture
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    if slots.is_empty() {
        slots.push(u16::MAX);
    }
    for slot in slots {
        let faces: Vec<[u32; 3]> = character
            .faces
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                character
                    .face_texture
                    .get(*index)
                    .copied()
                    .unwrap_or(u16::MAX)
                    == slot
            })
            .map(|(_, face)| *face)
            .collect();
        if faces.is_empty() {
            continue;
        }
        // Compact this primitive's vertices and remember where each one went.
        let mut remap = vec![u32::MAX; count];
        let mut used: Vec<u32> = Vec::new();
        for face in &faces {
            for index in face {
                let at = *index as usize;
                if remap[at] == u32::MAX {
                    remap[at] = used.len() as u32;
                    used.push(*index);
                }
            }
        }
        let mut positions = Vec::with_capacity(used.len() * 3);
        let mut vertex_normals = Vec::with_capacity(used.len() * 3);
        let mut uvs = Vec::with_capacity(used.len() * 2);
        let mut joints: Vec<f32> = Vec::with_capacity(used.len() * 4);
        let mut weights: Vec<f32> = Vec::with_capacity(used.len() * 4);
        for index in &used {
            let at = *index as usize;
            for axis in 0..3 {
                positions.push(character.rest[at * 3 + axis]);
                vertex_normals.push(base_normals[at * 3 + axis]);
            }
            if character.uv.len() == count {
                uvs.push(character.uv[at][0]);
                uvs.push(character.uv[at][1]);
            }
            if options.skin {
                for slot in 0..4 {
                    joints.push(character.joints[at][slot] as f32);
                    weights.push(character.weights[at][slot]);
                }
            }
        }
        let mut indices: Vec<u32> = Vec::with_capacity(faces.len() * 3);
        for face in &faces {
            for index in face {
                indices.push(remap[*index as usize]);
            }
        }
        let position_accessor = push_accessor(&mut bin, &mut accessors, positions, "VEC3", 5126, Some("ARRAY_BUFFER"));
        let normal_accessor = push_accessor(&mut bin, &mut accessors, vertex_normals, "VEC3", 5126, Some("ARRAY_BUFFER"));
        let mut attributes = serde_json::json!({
            "POSITION": position_accessor,
            "NORMAL": normal_accessor,
        });
        if character.uv.len() == count {
            let uv_accessor = push_accessor(&mut bin, &mut accessors, uvs, "VEC2", 5126, Some("ARRAY_BUFFER"));
            attributes["TEXCOORD_0"] = serde_json::json!(uv_accessor);
        }
        if options.skin {
            let joints_accessor = push_accessor(&mut bin, &mut accessors, joints, "VEC4", 5126, Some("ARRAY_BUFFER"));
            let weights_accessor = push_accessor(&mut bin, &mut accessors, weights, "VEC4", 5126, Some("ARRAY_BUFFER"));
            attributes["JOINTS_0"] = serde_json::json!(joints_accessor);
            attributes["WEIGHTS_0"] = serde_json::json!(weights_accessor);
        }
        // Indices are 32-bit: write the bytes directly rather than through the
        // float path.
        let mut index_bytes = Vec::with_capacity(indices.len() * 4);
        for index in &indices {
            index_bytes.extend_from_slice(&index.to_le_bytes());
        }
        let index_view = bin.push(&index_bytes, Some("ELEMENT_ARRAY_BUFFER"));
        accessors.push(serde_json::json!({
            "bufferView": index_view,
            "componentType": 5125,
            "count": indices.len(),
            "type": "SCALAR",
        }));
        let index_accessor = accessors.len() - 1;
        let mut primitive = serde_json::json!({
            "attributes": attributes,
            "indices": index_accessor,
            "mode": 4,
        });
        if options.materials && (slot as usize) < materials.len() {
            primitive["material"] = serde_json::json!(slot);
        }
        primitives.push(primitive);
    }

    // --- skeleton ---------------------------------------------------------
    // One node per bone, parent before child so the array index is the parent's.
    let mut nodes: Vec<serde_json::Value> = Vec::new();
    let mut node_of = vec![usize::MAX; bones.len()];
    for &index in &character.order {
        node_of[index] = nodes.len();
        nodes.push(serde_json::json!({ "name": bones[index].name }));
    }
    for &index in &character.order {
        let node = node_of[index];
        let local = match bones[index].parent {
            Some(parent) => multiply_local(parent, index, character),
            None => bind_matrix(index, character),
        };
        nodes[node]["matrix"] = serde_json::json!(column_major(&local));
        if let Some(parent) = bones[index].parent {
            let parent_node = node_of[parent];
            match nodes[parent_node].get_mut("children") {
                Some(children) => children.as_array_mut().unwrap().push(serde_json::json!(node)),
                None => nodes[parent_node]["children"] = serde_json::json!([node]),
            }
        }
    }

    let mut roots: Vec<serde_json::Value> = Vec::new();
    for &index in &character.order {
        if bones[index].parent.is_none() {
            roots.push(serde_json::json!(node_of[index]));
        }
    }

    // The mesh node sits under no bone: the skin does the work.
    let mesh_node = nodes.len();
    // A skinned mesh takes its transform from the skin, but its node still has to
    // be listed in the scene or an importer sees a scene with no geometry.
    let mut mesh_entry = serde_json::json!({
        "name": format!("{}_mesh", character.name),
        "mesh": 0,
    });
    if options.skin {
        mesh_entry["skin"] = serde_json::json!(0);
    }
    nodes.push(mesh_entry);
    let mut scene_nodes: Vec<serde_json::Value> = vec![serde_json::json!(mesh_node)];
    scene_nodes.extend(roots.iter().cloned());

    let mut skin_entry = serde_json::Value::Null;
    if options.skin {
        let mut inverse = Vec::with_capacity(bones.len() * 16);
        for bone in bones {
            // glTF stores column-major matrices; `inverse_bind` is row-major.
            for column in 0..4 {
                for row in 0..4 {
                    inverse.push(bone.inverse_bind[row * 4 + column]);
                }
            }
        }
        let accessor = push_accessor(
            &mut bin,
            &mut accessors,
            inverse,
            "MAT4",
            5126,
            None,
        );
        let joints: Vec<serde_json::Value> = node_of.iter().map(|node| serde_json::json!(node)).collect();
        skin_entry = serde_json::json!({
            "joints": joints,
            "inverseBindMatrices": accessor,
            "skeleton": roots.first().cloned().unwrap_or(serde_json::json!(0)),
        });
    }

    // --- animation --------------------------------------------------------
    let mut animations: Vec<serde_json::Value> = Vec::new();
    if options.animation && frames.len() > 1 {
        let mut samplers: Vec<serde_json::Value> = Vec::new();
        let mut channels: Vec<serde_json::Value> = Vec::new();
        let times: Vec<f32> = (0..frames.len()).map(|frame| frame as f32 / 30.0).collect();
        let time_accessor = push_accessor(&mut bin, &mut accessors, times, "SCALAR", 5126, None);
        for (index, bone) in bones.iter().enumerate() {
            let node = node_of[index];
            // Translation: the bone's world position relative to its parent's
            // world position, in the parent's rest frame.
            let mut moves = Vec::with_capacity(frames.len() * 3);
            let mut turns = Vec::with_capacity(frames.len() * 4);
            for frame in frames {
                let at = frame.positions.get(index).copied().unwrap_or(bone.bind);
                let local = match bone.parent {
                    Some(parent) => {
                        let from = frame.positions.get(parent).copied().unwrap_or([0.0; 3]);
                        let rotation = frame.rotations.get(parent).copied().unwrap_or([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
                        let offset = [at[0] - from[0], at[1] - from[1], at[2] - from[2]];
                        turn(&transpose(&rotation), offset)
                    }
                    None => at,
                };
                moves.extend_from_slice(&local);
                let rotation = frame.rotations.get(index).copied().unwrap_or([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
                turns.extend_from_slice(&quaternion(&rotation));
            }
            let translation = push_accessor(&mut bin, &mut accessors, moves, "VEC3", 5126, None);
            let rotation = push_accessor(&mut bin, &mut accessors, turns, "VEC4", 5126, None);
            samplers.push(serde_json::json!({ "input": time_accessor, "output": translation, "interpolation": "LINEAR" }));
            channels.push(serde_json::json!({ "sampler": samplers.len() - 1, "target": { "node": node, "path": "translation" } }));
            samplers.push(serde_json::json!({ "input": time_accessor, "output": rotation, "interpolation": "LINEAR" }));
            channels.push(serde_json::json!({ "sampler": samplers.len() - 1, "target": { "node": node, "path": "rotation" } }));
        }
        animations.push(serde_json::json!({
            "name": "clip",
            "samplers": samplers,
            "channels": channels,
        }));
    }

    // --- document ---------------------------------------------------------
    let mut document = serde_json::json!({
        "asset": { "version": "2.0", "generator": "VeloMocap" },
        "scene": 0,
        "scenes": [ { "name": character.name, "nodes": scene_nodes } ],
        "nodes": nodes,
        "meshes": [ { "name": character.name, "primitives": primitives } ],
        "accessors": accessors,
        "bufferViews": bin.views,
        "buffers": [ { "byteLength": bin.bytes.len() } ],
    });
    if options.materials && !materials.is_empty() {
        document["materials"] = serde_json::json!(materials);
        document["textures"] = serde_json::json!(gltf_textures);
        document["images"] = serde_json::json!(images);
        document["samplers"] = serde_json::json!([{ "magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497 }]);
    }
    if !skin_entry.is_null() {
        document["skins"] = serde_json::json!([skin_entry]);
    }
    if !animations.is_empty() {
        document["animations"] = serde_json::json!(animations);
    }
    write_container(document, &bin.bytes, out)
}

fn write_container(document: serde_json::Value, bin: &[u8], out: &Path) -> Result<()> {
    let mut json = serde_json::to_vec(&document).context("serialising the glTF document")?;
    let json_padding = align4(json.len());
    json.extend(std::iter::repeat_n(b' ', json_padding));
    let mut binary = bin.to_vec();
    binary.extend(std::iter::repeat_n(0u8, align4(binary.len())));

    let mut file = Vec::with_capacity(12 + 8 + json.len() + 8 + binary.len());
    file.extend_from_slice(b"glTF");
    file.extend_from_slice(&2u32.to_le_bytes());
    let total = 12 + 8 + json.len() + 8 + binary.len();
    file.extend_from_slice(&(total as u32).to_le_bytes());
    file.extend_from_slice(&(json.len() as u32).to_le_bytes());
    file.extend_from_slice(b"JSON");
    file.extend_from_slice(&json);
    file.extend_from_slice(&(binary.len() as u32).to_le_bytes());
    file.extend_from_slice(b"BIN\0");
    file.extend_from_slice(&binary);
    std::fs::write(out, file).with_context(|| format!("writing {}", out.display()))?;
    Ok(())
}

fn push_accessor(
    bin: &mut Bin,
    accessors: &mut Vec<serde_json::Value>,
    values: Vec<f32>,
    kind: &str,
    component: u32,
    target: Option<&str>,
) -> usize {
    let view = bin.floats(&values, target);
    let per_item = match kind {
        "SCALAR" => 1,
        "VEC2" => 2,
        "VEC3" => 3,
        "VEC4" => 4,
        "MAT4" => 16,
        _ => 1,
    };
    let count = values.len() / per_item;
    accessors.push(serde_json::json!({
        "bufferView": view,
        "componentType": component,
        "count": count,
        "type": kind,
    }));
    accessors.len() - 1
}

fn vertex_of(rest: &[f32], index: u32) -> [f32; 3] {
    let at = index as usize * 3;
    [rest[at], rest[at + 1], rest[at + 2]]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn transpose(m: &[f32; 9]) -> [f32; 9] {
    [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]
}

fn turn(m: &[f32; 9], v: [f32; 3]) -> [f32; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

/// Row-major `bind_rot | bind` as a 4×4.
fn bind_matrix(index: usize, character: &Character) -> [f32; 16] {
    let bone = &character.bones[index];
    let m = bone.bind_rot;
    [
        m[0], m[1], m[2], bone.bind[0],
        m[3], m[4], m[5], bone.bind[1],
        m[6], m[7], m[8], bone.bind[2],
        0.0, 0.0, 0.0, 1.0,
    ]
}

/// The child's bind matrix expressed in its parent's bind frame.
fn multiply_local(parent: usize, child: usize, character: &Character) -> [f32; 16] {
    let inverse = character.bones[parent].inverse_bind;
    multiply4(&inverse, &bind_matrix(child, character))
}

fn multiply4(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for row in 0..4 {
        for column in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += a[row * 4 + k] * b[k * 4 + column];
            }
            out[row * 4 + column] = sum;
        }
    }
    out
}

fn column_major(m: &[f32; 16]) -> Vec<f32> {
    let mut out = Vec::with_capacity(16);
    for column in 0..4 {
        for row in 0..4 {
            out.push(m[row * 4 + column]);
        }
    }
    out
}

/// A rotation matrix as a glTF quaternion (`x, y, z, w`).
fn quaternion(m: &[f32; 9]) -> [f32; 4] {
    let trace = m[0] + m[4] + m[8];
    let (x, y, z, w) = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        ((m[7] - m[5]) / s, (m[2] - m[6]) / s, (m[3] - m[1]) / s, 0.25 * s)
    } else if m[0] > m[4] && m[0] > m[8] {
        let s = (1.0 + m[0] - m[4] - m[8]).sqrt() * 2.0;
        (0.25 * s, (m[1] + m[3]) / s, (m[2] + m[6]) / s, (m[7] - m[5]) / s)
    } else if m[4] > m[8] {
        let s = (1.0 + m[4] - m[0] - m[8]).sqrt() * 2.0;
        ((m[1] + m[3]) / s, 0.25 * s, (m[5] + m[7]) / s, (m[2] - m[6]) / s)
    } else {
        let s = (1.0 + m[8] - m[0] - m[4]).sqrt() * 2.0;
        ((m[2] + m[6]) / s, (m[5] + m[7]) / s, 0.25 * s, (m[3] - m[1]) / s)
    };
    [x, y, z, w]
}
