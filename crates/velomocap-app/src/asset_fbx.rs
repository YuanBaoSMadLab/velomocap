//! Write a character as a binary FBX 7.4: mesh, skin, materials and the take.
//!
//! [`crate::asset`] writes a single-file `.glb`; this is the container the DCCs
//! and engines ask for by name. Both use the character that is on screen, so
//! the file is the model the viewport shows rather than a skeleton to re-skin by
//! hand.
//!
//! The element encoding is [`velomocap_core::fbx`]'s: the 13-byte records, the
//! zlib-compressed arrays and the footer are the part that is easy to get subtly
//! wrong, so there is one copy of them.
//!
//! The layout follows what Maya and Blender write, because that is what the
//! importers were built against:
//!
//! * one `Geometry` in bind pose with per-corner normals and UVs and one
//!   `LayerElementMaterial` per polygon,
//! * one `Model` per bone (`LimbNode`) plus the mesh `Model`,
//! * a `Skin` deformer holding one `Cluster` per bone: the vertex indices it
//!   moves, their weights, and the bone's bind matrix as `TransformLink` (the
//!   mesh sits at the origin, so the cluster's own transform is identity),
//! * `Material` + `Texture` objects; a texture is a PNG written next to the FBX,
//!   which is the only way an FBX carries one that everything reads,
//! * one `AnimationStack` with a rotation curve per bone and a translation curve
//!   on the root, keyed in FBX ticks.
//!
//! Two conventions are worth knowing, because both importers disagree about
//! them and both were measured rather than assumed:
//!
//! * a cluster's `Transform` is stored in *bone* space (the mesh's bind
//!   transform seen from the bone), not in global space — writing identity there
//!   makes every importer offset the whole mesh,
//! * the rotation curves carry the bone's *absolute* local rotation, not the
//!   swing away from a `PreRotation`: Blender's animation path applies the
//!   static node rotation instead of `PreRotation`, so a delta-based file plays
//!   back nearly in its bind pose. Absolute values are also what the `.glb`
//!   holds, which is why the two exports pose identically.
//!
//! An FBX "XYZ" rotation is `Rz · Ry · Rx`, not the SciPy reading
//! [`velomocap_core::math::mat_to_euler`] implements, so the angle extraction
//! lives here and is checked against a matrix Blender printed.

use anyhow::{Context, Result};
use std::path::Path;

use velomocap_core::fbx::{self, Node, Prop};

use crate::asset::{
    bind_matrix, column_length, multiply, multiply_local, png_bytes, transpose, vertex_normals,
    Frame, Options,
};
use crate::character::Character;

const IDENTITY9: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

const IDENTITY4: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
];

/// FBX stores matrices column-major; ours are row-major.
fn matrix16(m: &[f32; 16]) -> Vec<f64> {
    let mut out = Vec::with_capacity(16);
    for column in 0..4 {
        for row in 0..4 {
            out.push(m[row * 4 + column] as f64);
        }
    }
    out
}

/// `Name\0\1Class`, the form every FBX object name takes.
fn class_name(name: &str, class: &str) -> String {
    format!("{name}\u{0}\u{1}{class}")
}

fn class_prop(name: &str, class: &str) -> Prop {
    Prop::Str(class_name(name, class))
}

/// One `C` record: `OO` is object to object, `OP` object to property.
fn connect(kind: &str, child: i64, parent: i64, property: Option<String>) -> Node {
    let mut props = vec![
        Prop::Str(kind.to_string()),
        Prop::I64(child),
        Prop::I64(parent),
    ];
    if let Some(property) = property {
        props.push(Prop::Str(property));
    }
    Node::with("C", props)
}

fn array(name: &str, prop: Prop) -> Node {
    Node::with(name, vec![prop])
}

fn string(name: &str, value: &str) -> Node {
    Node::with(name, vec![Prop::Str(value.to_string())])
}

fn int(name: &str, value: i64) -> Node {
    Node::with(name, vec![Prop::I64(value)])
}

fn object_type(kind: &str, count: i64) -> Node {
    Node::with("ObjectType", vec![Prop::Str(kind.to_string())]).child(int("Count", count))
}

/// Euler angles in degrees for a row-major 3×3 rotation, in the order the
/// readers rebuild it.
///
/// FBX's "XYZ" (and Blender's `Euler(..., 'XYZ').to_matrix()`) means the matrix
/// product `Rz · Ry · Rx`: X turns first. That is *not* what
/// [`velomocap_core::math::mat_to_euler`] returns — it follows SciPy's
/// lowercase, intrinsic reading — so the extraction lives here, stated in the
/// convention the file is read back with, and is checked against a matrix
/// Blender printed rather than against its own inverse.
fn euler_degrees(m: &[f32; 9]) -> [f64; 3] {
    let r00 = m[0] as f64;
    let r10 = m[3] as f64;
    let r20 = m[6] as f64;
    let r21 = m[7] as f64;
    let r22 = m[8] as f64;
    let y = (-r20).clamp(-1.0, 1.0).asin();
    let (z, x) = if y.cos().abs() > 1e-6 {
        (r10.atan2(r00), r21.atan2(r22))
    } else {
        // Gimbal lock: only the sum or difference of the outer angles is
        // defined, so one of them is pinned to zero.
        let up = m[6] < 0.0;
        let sign = if up { 1.0 } else { -1.0 };
        (0.0, (sign * m[1] as f64).atan2(sign * m[2] as f64))
    };
    [x.to_degrees(), y.to_degrees(), z.to_degrees()]
}

/// The inverse of [`euler_degrees`], for the tests and for nothing else: the
/// writer only ever converts a rotation into angles.
#[cfg(test)]
fn euler_matrix(angles: [f64; 3]) -> [f64; 9] {
    let (x, y, z) = (
        angles[0].to_radians(),
        angles[1].to_radians(),
        angles[2].to_radians(),
    );
    let (sx, cx) = x.sin_cos();
    let (sy, cy) = y.sin_cos();
    let (sz, cz) = z.sin_cos();
    // Rz · Ry · Rx
    [
        cz * cy,
        cz * sy * sx - sz * cx,
        cz * sy * cx + sz * sx,
        sz * cy,
        sz * sy * sx + cz * cx,
        sz * sy * cx - cz * sx,
        -sy,
        cy * sx,
        cy * cx,
    ]
}

/// Remove the ±360° jumps an Euler track picks up when an angle crosses the
/// wrapping point, so a curve interpolates the short way round.
///
/// This is [`velomocap_core::math::continuous_angles`]'s job, but that one works
/// in radians and these angles are in the degrees an FBX stores. The two
/// equivalent Euler triples `(x, y, z)` and `(x+180, 180−y, z+180)` both get
/// measured, so a rotation that passes through the gimbal-lock band still
/// unfolds into the nearest track instead of spinning.
fn unwrap_degrees(angles: &mut [f64]) {
    let mut previous = [0.0f64; 3];
    for frame in angles.chunks_exact_mut(3) {
        let direct = [frame[0], frame[1], frame[2]];
        let flipped = [frame[0] + 180.0, 180.0 - frame[1], frame[2] + 180.0];
        let mut best = [0.0f64; 3];
        let mut best_distance = f64::INFINITY;
        for candidate in [direct, flipped] {
            let mut shifted = candidate;
            for axis in 0..3 {
                shifted[axis] += 360.0 * ((previous[axis] - shifted[axis]) / 360.0).round();
            }
            let distance = (0..3)
                .map(|axis| (shifted[axis] - previous[axis]).powi(2))
                .sum::<f64>();
            if distance < best_distance {
                best_distance = distance;
                best = shifted;
            }
        }
        frame.copy_from_slice(&best);
        previous = best;
    }
}

/// Translation, unit rotation and uniform scale of a local matrix.
fn decompose(m: &[f32; 16]) -> ([f64; 3], [f32; 9], f64) {
    let rotation = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
    let scale = column_length(&rotation).max(1e-6);
    let unit = rotation.map(|value| value / scale);
    (
        [m[3] as f64, m[7] as f64, m[11] as f64],
        unit,
        scale as f64,
    )
}

/// The bind-space transform of one bone: its parent's inverse bind times its
/// own, which is what a node's local TRS has to reproduce.
fn local_bind(index: usize, character: &Character) -> [f32; 16] {
    match character.bones[index].parent {
        Some(parent) => multiply_local(parent, index, character),
        None => bind_matrix(index, character),
    }
}

/// Per-type object counts, so `Definitions` states them without a second pass.
#[derive(Default)]
struct Counts {
    model: i64,
    geometry: i64,
    material: i64,
    texture: i64,
    deformer: i64,
    pose: i64,
    stack: i64,
    layer: i64,
    curve_node: i64,
    curve: i64,
}

impl Counts {
    fn total(&self) -> i64 {
        1 + self.model
            + self.geometry
            + self.material
            + self.texture
            + self.deformer
            + self.pose
            + self.stack
            + self.layer
            + self.curve_node
            + self.curve
    }

    fn object_types(&self) -> Vec<(&'static str, i64)> {
        let mut types: Vec<(&'static str, i64)> = vec![
            ("GlobalSettings", 1),
            ("Model", self.model),
            ("Geometry", self.geometry),
            ("Material", self.material),
            ("Texture", self.texture),
            ("Deformer", self.deformer),
            ("Pose", self.pose),
            ("AnimationStack", self.stack),
            ("AnimationLayer", self.layer),
            ("AnimationCurveNode", self.curve_node),
            ("AnimationCurve", self.curve),
        ];
        types.retain(|(_, count)| *count > 0);
        types
    }
}

struct Ids(i64);

impl Ids {
    fn take(&mut self) -> i64 {
        let id = self.0;
        self.0 += 1;
        id
    }
}

/// One `AnimationCurveNode` and its three axis curves, targeting `target`.
#[allow(clippy::too_many_arguments)]
fn write_curve_node(
    ids: &mut Ids,
    counts: &mut Counts,
    objects: &mut Vec<Node>,
    connections: &mut Vec<Node>,
    layer: i64,
    target: i64,
    property: &str,
    kind: &str,
    times: &[i64],
    channels: &[Vec<f32>; 3],
) {
    let curve_node = ids.take();
    counts.curve_node += 1;
    let mut defaults = Vec::with_capacity(3);
    for (axis, channel) in ["X", "Y", "Z"].iter().zip(channels) {
        defaults.push(fbx::property(
            &format!("d|{axis}"),
            "Number",
            "",
            vec![Prop::F64(channel.first().copied().unwrap_or(0.0) as f64)],
        ));
    }
    objects.push(
        Node::with(
            "AnimationCurveNode",
            vec![
                Prop::I64(curve_node),
                class_prop(kind, "AnimCurveNode"),
                Prop::Str(String::new()),
            ],
        )
        .child(fbx::properties(defaults)),
    );
    connections.push(connect("OO", curve_node, layer, None));
    connections.push(connect(
        "OP",
        curve_node,
        target,
        Some(property.to_string()),
    ));
    for (axis, channel) in ["X", "Y", "Z"].iter().zip(channels) {
        let curve = ids.take();
        counts.curve += 1;
        objects.push(
            Node::with(
                "AnimationCurve",
                vec![
                    Prop::I64(curve),
                    class_prop(&format!("{kind}|{axis}"), "AnimCurve"),
                    Prop::Str(String::new()),
                ],
            )
            .child(Node::with(
                "Default",
                vec![Prop::F64(channel.first().copied().unwrap_or(0.0) as f64)],
            ))
            .child(int("KeyVer", 4009))
            .child(array("KeyTime", Prop::I64Array(times.to_vec())))
            .child(array("KeyValueFloat", Prop::F32Array(channel.clone())))
            .child(array("KeyAttrFlags", Prop::I32Array(vec![4])))
            .child(array(
                "KeyAttrDataFloat",
                Prop::F32Array(vec![0.0, 0.0, 0.0, 0.0]),
            ))
            .child(array(
                "KeyAttrRefCount",
                Prop::I32Array(vec![times.len() as i32]),
            )),
        );
        connections.push(connect("OP", curve, curve_node, Some(format!("d|{axis}"))));
    }
}

/// Write `character` as an FBX, animated by `frames`.
pub fn write_fbx(
    character: &Character,
    frames: &[Frame],
    options: Options,
    fps: f32,
    out: &Path,
) -> Result<()> {
    anyhow::ensure!(!character.faces.is_empty(), "the model has no faces");
    let folder = out.parent().unwrap_or(Path::new("."));
    let stem = out
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| "character".to_string());
    let label = if character.name.trim().is_empty() {
        stem.clone()
    } else {
        character.name.trim().to_string()
    };
    let bones = &character.bones;
    let count = character.vertex_count();

    let mut ids = Ids(100_000);
    let mut counts = Counts::default();
    let mut objects: Vec<Node> = Vec::new();
    let mut connections: Vec<Node> = Vec::new();

    // --- textures on disk --------------------------------------------------
    // An FBX has no embedded images that every importer reads, so the PNGs go
    // beside the file and both the absolute and the relative path are written.
    let mut textures: Vec<(String, String)> = Vec::new();
    if options.materials && !character.textures.is_empty() {
        let folder_name = format!("{stem}_textures");
        let directory = folder.join(&folder_name);
        std::fs::create_dir_all(&directory)
            .with_context(|| format!("creating {}", directory.display()))?;
        for (slot, texture) in character.textures.iter().enumerate() {
            let file = format!("tex{slot}.png");
            let path = directory.join(&file);
            std::fs::write(&path, png_bytes(texture)?)
                .with_context(|| format!("writing {}", path.display()))?;
            let absolute = path
                .canonicalize()
                .unwrap_or_else(|_| path.clone())
                .to_string_lossy()
                .replace('\\', "/");
            textures.push((absolute, file));
        }
    }
    let texture_folder = format!("{stem}_textures");

    // --- geometry ----------------------------------------------------------
    // Faces stay triangles: FBX polygons are arbitrary, so the simplest thing
    // that round-trips is the triangle list the renderer already draws.
    let geometry_id = ids.take();
    counts.geometry += 1;
    let mut vertices: Vec<f64> = Vec::with_capacity(count * 3);
    for value in &character.rest {
        vertices.push(*value as f64);
    }
    let mut polygon_vertex: Vec<i32> = Vec::with_capacity(character.faces.len() * 3);
    let mut corner: Vec<i32> = Vec::with_capacity(character.faces.len() * 3);
    let mut polygons: Vec<i32> = Vec::with_capacity(character.faces.len());
    for (index, face) in character.faces.iter().enumerate() {
        polygon_vertex.push(face[0] as i32);
        polygon_vertex.push(face[1] as i32);
        // The last corner of a polygon is stored negated (XOR -1): that is how
        // the format marks where one polygon ends and the next begins.
        polygon_vertex.push(!(face[2] as i32));
        corner.extend_from_slice(&[face[0] as i32, face[1] as i32, face[2] as i32]);
        let slot = character.face_texture.get(index).copied().unwrap_or(0);
        polygons.push(slot as i32);
    }
    let mut geometry = Node::with(
        "Geometry",
        vec![
            Prop::I64(geometry_id),
            class_prop(&format!("{label}_mesh"), "Geometry"),
            Prop::Str("Mesh".to_string()),
        ],
    )
    .child(int("GeometryVersion", 124))
    .child(array("Vertices", Prop::F64Array(vertices)))
    .child(array("PolygonVertexIndex", Prop::I32Array(polygon_vertex)));

    // Normals and UVs are indexed per polygon corner into the per-vertex
    // arrays: one corner index serves both layers.
    let mut layers: Vec<(&str, i32)> = Vec::new();
    let normals = vertex_normals(character);
    let normal_values: Vec<f64> = normals.iter().map(|value| *value as f64).collect();
    geometry = geometry.child(
        Node::with("LayerElementNormal", vec![Prop::I64(0)])
            .child(int("Version", 101))
            .child(string("Name", ""))
            .child(string("MappingInformationType", "ByPolygonVertex"))
            .child(string("ReferenceInformationType", "IndexToDirect"))
            .child(array("Normals", Prop::F64Array(normal_values)))
            .child(array("NormalsIndex", Prop::I32Array(corner.clone()))),
    );
    layers.push(("LayerElementNormal", 0));
    if character.uv.len() == count {
        let mut uvs: Vec<f64> = Vec::with_capacity(count * 2);
        for uv in &character.uv {
            uvs.push(uv[0] as f64);
            uvs.push(uv[1] as f64);
        }
        geometry = geometry.child(
            Node::with("LayerElementUV", vec![Prop::I64(0)])
                .child(int("Version", 101))
                .child(string("Name", "UVMap"))
                .child(string("MappingInformationType", "ByPolygonVertex"))
                .child(string("ReferenceInformationType", "IndexToDirect"))
                .child(array("UV", Prop::F64Array(uvs)))
                .child(array("UVIndex", Prop::I32Array(corner.clone()))),
        );
        layers.push(("LayerElementUV", 0));
    }
    if !textures.is_empty() {
        geometry = geometry.child(
            Node::with("LayerElementMaterial", vec![Prop::I64(0)])
                .child(int("Version", 101))
                .child(string("Name", ""))
                .child(string("MappingInformationType", "ByPolygon"))
                .child(string("ReferenceInformationType", "IndexToDirect"))
                .child(array("Materials", Prop::I32Array(polygons))),
        );
        layers.push(("LayerElementMaterial", 0));
    }
    let mut layer = Node::with("Layer", vec![Prop::I32(0)]).child(int("Version", 100));
    for (kind, index) in &layers {
        layer = layer.child(Node::with(
            "LayerElement",
            vec![Prop::Str((*kind).to_string()), Prop::I32(*index)],
        ));
    }
    geometry = geometry.child(layer);
    objects.push(geometry);

    // --- the mesh object ---------------------------------------------------
    let mesh_model = ids.take();
    counts.model += 1;
    objects.push(
        Node::with(
            "Model",
            vec![
                Prop::I64(mesh_model),
                class_prop(&label, "Model"),
                Prop::Str("Mesh".to_string()),
            ],
        )
        .child(int("Version", 232))
        .child(fbx::properties(vec![
            fbx::property_vec("Lcl Translation", "Lcl Translation", "A", [0.0, 0.0, 0.0]),
            fbx::property_vec("Lcl Rotation", "Lcl Rotation", "A", [0.0, 0.0, 0.0]),
            fbx::property_vec("Lcl Scaling", "Lcl Scaling", "A", [1.0, 1.0, 1.0]),
        ])),
    );
    connections.push(connect("OO", geometry_id, mesh_model, None));
    connections.push(connect("OO", mesh_model, 0, None));

    // --- skeleton ----------------------------------------------------------
    // `order` is parent before child, which is the order the objects and the
    // connections are written in.
    let mut node_of: Vec<i64> = vec![0; bones.len()];
    for &index in &character.order {
        let id = ids.take();
        node_of[index] = id;
        counts.model += 1;
        let (translation, rotation, scale) = decompose(&local_bind(index, character));
        // The node's own rotation, in the degrees FBX stores. The animation
        // curves then carry the absolute local rotation for each frame, exactly
        // as the `.glb` writes it: an importer that honours `PreRotation` and
        // one that quietly drops it would otherwise disagree, and Blender is the
        // second kind on its animation path.
        let rest_angles = euler_degrees(&rotation);
        objects.push(
            Node::with(
                "Model",
                vec![
                    Prop::I64(id),
                    class_prop(&bones[index].name, "Model"),
                    Prop::Str("LimbNode".to_string()),
                ],
            )
            .child(int("Version", 232))
            .child(fbx::properties(vec![
                fbx::property_vec("Lcl Translation", "Lcl Translation", "A", translation),
                fbx::property_vec("Lcl Rotation", "Lcl Rotation", "A", rest_angles),
                fbx::property_vec("Lcl Scaling", "Lcl Scaling", "A", [scale, scale, scale]),
                fbx::property_int("RotationActive", "bool", 1),
                fbx::property_int("RotationOrder", "enum", 0),
                // Rotate-then-scale inheritance, the default the DCCs write.
                fbx::property_int("InheritType", "enum", 1),
            ])),
        );
    }
    for &index in &character.order {
        let parent = match bones[index].parent {
            Some(parent) => node_of[parent],
            None => 0,
        };
        connections.push(connect("OO", node_of[index], parent, None));
    }

    // --- skin --------------------------------------------------------------
    if options.skin {
        let skin = ids.take();
        counts.deformer += 1;
        objects.push(
            Node::with(
                "Deformer",
                vec![
                    Prop::I64(skin),
                    class_prop(&format!("{label}_skin"), "Deformer"),
                    Prop::Str("Skin".to_string()),
                ],
            )
            .child(int("Version", 101))
            .child(array("Link_DeformAcuracy", Prop::F64(50.0)))
            .child(string("SkinningType", "Linear")),
        );
        // The skin hangs off the *geometry*, not the object: that is the
        // connection the FBX SDK attaches deformers through, and the one every
        // importer follows to find the mesh a set of clusters drives.
        connections.push(connect("OO", skin, geometry_id, None));
        // One cluster per bone: which vertices it moves, by how much, and the
        // bone's bind matrix. The mesh is in bind space at the origin, so the
        // cluster's own transform is identity.
        let mut indices: Vec<Vec<i32>> = vec![Vec::new(); bones.len()];
        let mut weights: Vec<Vec<f64>> = vec![Vec::new(); bones.len()];
        for vertex in 0..count {
            for slot in 0..4 {
                let weight = character.weights[vertex][slot];
                let bone = character.joints[vertex][slot] as usize;
                if weight <= 1e-6 || bone >= bones.len() {
                    continue;
                }
                indices[bone].push(vertex as i32);
                weights[bone].push(weight as f64);
            }
        }
        for index in 0..bones.len() {
            if indices[index].is_empty() {
                continue;
            }
            let cluster = ids.take();
            counts.deformer += 1;
            objects.push(
                Node::with(
                    "Deformer",
                    vec![
                        Prop::I64(cluster),
                        class_prop(&format!("{label}_skin_{}", bones[index].name), "Deformer"),
                        Prop::Str("Cluster".to_string()),
                    ],
                )
                .child(int("Version", 100))
                .child(array("Indexes", Prop::I32Array(indices[index].clone())))
                .child(array("Weights", Prop::F64Array(weights[index].clone())))
                // `Transform` is stored in *bone* space, not global space (the
                // SDK's own documentation is wrong about this): it is the
                // mesh's bind transform expressed in the bone's frame, which
                // for a mesh at the origin is the inverse bind matrix. Writing
                // identity here makes every importer offset the whole mesh.
                .child(array(
                    "Transform",
                    Prop::F64Array(matrix16(&bones[index].inverse_bind)),
                ))
                .child(array(
                    "TransformLink",
                    Prop::F64Array(matrix16(&bind_matrix(index, character))),
                ))
                .child(array(
                    "TransformAssociateModel",
                    Prop::F64Array(matrix16(&IDENTITY4)),
                )),
            );
            connections.push(connect("OO", cluster, skin, None));
            connections.push(connect("OO", node_of[index], cluster, None));
        }
    }

    // --- materials and textures -------------------------------------------
    for (slot, (absolute, file)) in textures.iter().enumerate() {
        let name = format!("{label}_material{slot}");
        let material = ids.take();
        counts.material += 1;
        objects.push(
            Node::with(
                "Material",
                vec![
                    Prop::I64(material),
                    class_prop(&name, "Material"),
                    Prop::Str(String::new()),
                ],
            )
            .child(int("Version", 102))
            .child(string("ShadingModel", "Phong"))
            .child(int("MultiLayer", 0))
            .child(fbx::properties(vec![
                fbx::property_vec("DiffuseColor", "Color", "A", [0.8, 0.8, 0.8]),
                fbx::property_double("DiffuseFactor", 1.0),
                fbx::property_vec("AmbientColor", "Color", "A", [0.0, 0.0, 0.0]),
                fbx::property_vec("SpecularColor", "Color", "A", [0.0, 0.0, 0.0]),
                fbx::property_vec("EmissiveColor", "Color", "A", [0.0, 0.0, 0.0]),
                fbx::property_double("ShininessExponent", 20.0),
                fbx::property_double("TransparencyFactor", 0.0),
                fbx::property_double("Opacity", 1.0),
            ])),
        );
        connections.push(connect("OO", material, mesh_model, None));

        let texture = ids.take();
        counts.texture += 1;
        let texture_name = format!("{label}_texture{slot}");
        objects.push(
            Node::with(
                "Texture",
                vec![
                    Prop::I64(texture),
                    class_prop(&texture_name, "Texture"),
                    Prop::Str(String::new()),
                ],
            )
            .child(string("Type", "TextureVideoClip"))
            .child(int("Version", 202))
            .child(Node::with(
                "TextureName",
                vec![class_prop(&texture_name, "Texture")],
            ))
            .child(string("FileName", absolute))
            .child(string(
                "RelativeFilename",
                &format!("{texture_folder}/{file}"),
            ))
            .child(Node::with(
                "ModelUVTranslation",
                vec![Prop::F64(0.0), Prop::F64(0.0)],
            ))
            .child(Node::with(
                "ModelUVScaling",
                vec![Prop::F64(1.0), Prop::F64(1.0)],
            ))
            .child(string("Texture_Alpha_Source", "None"))
            .child(Node::with(
                "Cropping",
                vec![
                    Prop::F64(0.0),
                    Prop::F64(0.0),
                    Prop::F64(0.0),
                    Prop::F64(0.0),
                ],
            )),
        );
        connections.push(connect(
            "OP",
            texture,
            material,
            Some("DiffuseColor".to_string()),
        ));
    }

    // --- animation ---------------------------------------------------------
    let fps = if fps.is_finite() && fps > 0.0 { fps } else { 30.0 };
    if options.animation && frames.len() > 1 {
        let stack = ids.take();
        counts.stack += 1;
        let times: Vec<i64> = (0..frames.len())
            .map(|frame| (frame as f64 * fbx::TICKS / fps as f64).round() as i64)
            .collect();
        let stop = times.last().copied().unwrap_or(0);
        objects.push(
            Node::with(
                "AnimationStack",
                vec![
                    Prop::I64(stack),
                    class_prop(&format!("{label}_take"), "AnimStack"),
                    Prop::Str(String::new()),
                ],
            )
            .child(fbx::properties(vec![
                fbx::property_kint("LocalStart", 0),
                fbx::property("LocalStop", "KTime", "", vec![Prop::I64(stop)]),
                fbx::property_kint("ReferenceStart", 0),
                fbx::property("ReferenceStop", "KTime", "", vec![Prop::I64(stop)]),
            ])),
        );
        let layer = ids.take();
        counts.layer += 1;
        objects.push(
            Node::with(
                "AnimationLayer",
                vec![
                    Prop::I64(layer),
                    class_prop("BaseLayer", "AnimLayer"),
                    Prop::Str(String::new()),
                ],
            )
            .child(fbx::properties(Vec::new())),
        );
        connections.push(connect("OO", layer, stack, None));

        for &index in &character.order {
            let bone = &bones[index];
            // Absolute local rotation per frame, the same number the `.glb`
            // writes for that node: no rest subtraction, so a reader that
            // ignores `PreRotation` still lands on the pose the viewport shows.
            let mut angles: Vec<f64> = Vec::with_capacity(frames.len() * 3);
            for frame in frames {
                let world = frame.rotations.get(index).copied().unwrap_or(IDENTITY9);
                let local = match bone.parent {
                    Some(parent) => {
                        let parent_world = frame
                            .rotations
                            .get(parent)
                            .copied()
                            .unwrap_or(IDENTITY9);
                        multiply(&transpose(&parent_world), &world)
                    }
                    None => world,
                };
                angles.extend_from_slice(&euler_degrees(&local));
            }
            unwrap_degrees(&mut angles);
            if std::env::var_os("VELOMOCAP_DEBUG_FBX").is_some() {
                let worst = angles.iter().fold(0.0f64, |worst, value| worst.max(value.abs()));
                eprintln!("fbx-anim: {:<28} max {worst:7.1}°", bone.name);
            }
            let mut channels: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
            for (axis, channel) in channels.iter_mut().enumerate() {
                *channel = angles
                    .chunks_exact(3)
                    .map(|angles| angles[axis] as f32)
                    .collect();
            }
            write_curve_node(
                &mut ids,
                &mut counts,
                &mut objects,
                &mut connections,
                layer,
                node_of[index],
                "Lcl Rotation",
                "R",
                &times,
                &channels,
            );
            if bone.parent.is_none() {
                // The root is the one bone whose offset changes; every other
                // bone keeps its bind offset and only rotates.
                let mut channels: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
                for frame in frames {
                    let at = frame.positions.get(index).copied().unwrap_or([0.0; 3]);
                    for (axis, channel) in channels.iter_mut().enumerate() {
                        channel.push(at[axis]);
                    }
                }
                write_curve_node(
                    &mut ids,
                    &mut counts,
                    &mut objects,
                    &mut connections,
                    layer,
                    node_of[index],
                    "Lcl Translation",
                    "T",
                    &times,
                    &channels,
                );
            }
        }
    }

    // --- bind pose ---------------------------------------------------------
    let pose = ids.take();
    counts.pose += 1;
    let mut pose_nodes: Vec<Node> = Vec::with_capacity(bones.len() + 1);
    pose_nodes.push(
        Node::with("PoseNode", Vec::new())
            .child(int("Node", mesh_model))
            .child(array("Matrix", Prop::F64Array(matrix16(&IDENTITY4)))),
    );
    for &index in &character.order {
        pose_nodes.push(
            Node::with("PoseNode", Vec::new())
                .child(int("Node", node_of[index]))
                .child(array(
                    "Matrix",
                    Prop::F64Array(matrix16(&bind_matrix(index, character))),
                )),
        );
    }
    objects.push(
        Node::with(
            "Pose",
            vec![
                Prop::I64(pose),
                class_prop("BindPose", "Pose"),
                Prop::Str("BindPose".to_string()),
            ],
        )
        .child(string("Type", "BindPose"))
        .child(int("Version", 100))
        .child(int("NbPoseNodes", pose_nodes.len() as i64))
        .children(pose_nodes),
    );

    // --- document ----------------------------------------------------------
    let mut bytes: Vec<u8> = Vec::with_capacity(1 << 22);
    fbx::begin(&mut bytes);
    fbx::write_element(
        &mut bytes,
        &Node::new("FBXHeaderExtension")
            .child(int("FBXHeaderVersion", 1003))
            .child(int("FBXVersion", 7400))
            .child(string("Creator", "VeloMocap")),
    )?;
    let settings = vec![
        fbx::property_int("UpAxis", "int", 1),
        fbx::property_int("UpAxisSign", "int", 1),
        fbx::property_int("FrontAxis", "int", 2),
        fbx::property_int("FrontAxisSign", "int", 1),
        fbx::property_int("CoordAxis", "int", 0),
        fbx::property_int("CoordAxisSign", "int", 1),
        fbx::property_int("OriginalUpAxis", "int", 1),
        fbx::property_int("OriginalUpAxisSign", "int", 1),
        // 100 centimetres per unit: the mesh is in metres, like the `.glb` and
        // like the rest of the pipeline.
        fbx::property_double("UnitScaleFactor", 100.0),
        fbx::property_double("OriginalUnitScaleFactor", 100.0),
        // 13 is "custom", and `CustomFrameRate` then carries the clip's rate.
        fbx::property_int("TimeMode", "enum", 13),
        fbx::property_double("CustomFrameRate", fps as f64),
    ];
    fbx::write_element(
        &mut bytes,
        &Node::new("GlobalSettings")
            .child(int("Version", 1000))
            .child(fbx::properties(settings)),
    )?;
    fbx::write_element(
        &mut bytes,
        &Node::new("Documents")
            .child(int("Count", 1))
            .child(
                Node::with(
                    "Document",
                    vec![
                        Prop::I64(900),
                        Prop::Str("Scene".to_string()),
                        Prop::Str("Scene".to_string()),
                    ],
                )
                .child(fbx::properties(Vec::new()))
                .child(int("RootNode", 0)),
            ),
    )?;
    fbx::write_element(&mut bytes, &Node::new("References"))?;
    let mut definitions = Node::new("Definitions")
        .child(int("Version", 100))
        .child(int("Count", counts.total()));
    for (kind, count) in counts.object_types() {
        definitions = definitions.child(object_type(kind, count));
    }
    fbx::write_element(&mut bytes, &definitions)?;
    fbx::write_element(&mut bytes, &Node::new("Objects").children(objects))?;
    fbx::write_element(&mut bytes, &Node::new("Connections").children(connections))?;
    fbx::write_element(
        &mut bytes,
        &Node::new("Takes").child(string("Current", "Take 001")),
    )?;
    fbx::finish(&mut bytes);
    std::fs::write(out, &bytes).with_context(|| format!("writing {}", out.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The local matrix of a bone decomposes into TRS the way a node reads it
    /// back: scale, unit rotation and the offset in the parent's frame.
    #[test]
    fn a_local_matrix_decomposes_into_the_node_trs() {
        let angle = 0.6f32;
        let scale = 0.01f32;
        // Parent frame: rotate about Y, shift, and shrink.
        let parent = [
            angle.cos() * scale, 0.0, angle.sin() * scale, 1.5,
            0.0, scale, 0.0, 2.0,
            -angle.sin() * scale, 0.0, angle.cos() * scale, 3.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let (translation, rotation, factor) = decompose(&parent);
        assert!((factor as f32 - scale as f32).abs() < 1e-6);
        assert!((translation[0] - 1.5).abs() < 1e-6);
        assert!((translation[1] - 2.0).abs() < 1e-6);
        assert!((translation[2] - 3.0).abs() < 1e-6);
        // The unit rotation turns +X into (cos, 0, -sin).
        let x = [rotation[0], rotation[3], rotation[6]];
        assert!((x[0] - angle.cos()).abs() < 1e-5, "{x:?}");
        assert!((x[2] + angle.sin()).abs() < 1e-5, "{x:?}");
        assert!((rotation[1]).abs() < 1e-5 && (rotation[4] - 1.0).abs() < 1e-5);
    }

    /// Euler angles and the matrix they came from agree, so a node's stored
    /// rest and the curve written for the same pose are the same rotation.
    #[test]
    fn euler_angles_round_trip_through_the_writers_own_convention() {
        let angle = 0.85f32;
        let m = [
            angle.cos(), 0.0, angle.sin(),
            0.0, 1.0, 0.0,
            -angle.sin(), 0.0, angle.cos(),
        ];
        let angles = euler_degrees(&m);
        let back = euler_matrix(angles);
        for (a, b) in m.iter().zip(&back) {
            assert!((a - *b as f32).abs() < 1e-5, "{m:?} vs {back:?}");
        }
    }

    /// The convention has to be Blender's, because that is the matrix a reader
    /// rebuilds: `Euler((30, 50, 70)°, "XYZ").to_matrix()` is Rz·Ry·Rx, and this
    /// is the matrix it printed.
    #[test]
    fn euler_xyz_matches_the_convention_readers_rebuild() {
        let matrix = [
            0.2198, -0.6828, 0.6967,
            0.6040, 0.6561, 0.4524,
            -0.7660, 0.3214, 0.5567,
        ];
        let angles = euler_degrees(&matrix);
        assert!((angles[0] - 30.0).abs() < 0.01, "{angles:?}");
        assert!((angles[1] - 50.0).abs() < 0.01, "{angles:?}");
        assert!((angles[2] - 70.0).abs() < 0.01, "{angles:?}");
        let back = euler_matrix(angles);
        for (a, b) in matrix.iter().zip(&back) {
            assert!((a - *b as f32).abs() < 1e-4, "{matrix:?} vs {back:?}");
        }
    }

    /// The node TRS the writer stores must rebuild the same world bind pose the
    /// clusters state, or every reader deforms the mesh by the difference: the
    /// rest comes from the clusters, the pose comes from the nodes.
    #[test]
    fn the_node_trs_rebuilds_the_bind_pose() {
        let Some(path) = std::env::var_os("VELOMOCAP_CHARACTER") else {
            return;
        };
        let character = Character::load(std::path::Path::new(&path)).expect("load");
        let mut world: Vec<[f32; 16]> =
            vec![IDENTITY4; character.bones.len()];
        let mut worst = 0.0f32;
        let mut worst_bone = String::new();
        for &index in &character.order {
            let (translation, rotation, scale) = decompose(&local_bind(index, &character));
            let r = euler_matrix(euler_degrees(&rotation));
            let mut local = IDENTITY4;
            for row in 0..3 {
                for column in 0..3 {
                    local[row * 4 + column] = r[row * 3 + column] as f32 * scale as f32;
                }
                local[row * 4 + 3] = translation[row] as f32;
            }
            let parent = match character.bones[index].parent {
                Some(parent) => world[parent],
                None => IDENTITY4,
            };
            world[index] = crate::asset::multiply4(&parent, &local);
            let bind = bind_matrix(index, &character);
            let delta = world[index]
                .iter()
                .zip(bind.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            if delta > worst {
                worst = delta;
                worst_bone = character.bones[index].name.clone();
            }
        }
        assert!(
            worst < 5e-3,
            "the node TRS and the cluster bind disagree by {worst} at {worst_bone}"
        );
    }

    /// The file a game engine opens starts with the FBX signature and the 7.4
    /// version, and every combination of the three content switches writes one
    /// without panicking. The switches are what the export dialog offers, so
    /// each of the eight states has to be a file, not a crash.
    #[test]
    fn every_content_switch_writes_a_74_file() {
        let Some(path) = std::env::var_os("VELOMOCAP_CHARACTER") else {
            return;
        };
        let character = Character::load(std::path::Path::new(&path)).expect("load");
        let folder = std::env::temp_dir().join("velomocap-fbx-test");
        std::fs::create_dir_all(&folder).expect("temp folder");
        let signature = b"Kaydara FBX Binary  \0\x1a\0";
        for bits in 0..8 {
            let options = Options {
                skin: bits & 1 != 0,
                materials: bits & 2 != 0,
                animation: bits & 4 != 0,
            };
            let out = folder.join(format!("switch-{bits}.fbx"));
            write_fbx(&character, &[], options, 30.0, &out).expect("write");
            let bytes = std::fs::read(&out).expect("read back");
            assert!(
                bytes.starts_with(signature),
                "switch {bits}: the signature is missing"
            );
            assert_eq!(
                u32::from_le_bytes([bytes[23], bytes[24], bytes[25], bytes[26]]),
                7400,
                "switch {bits}: not a 7.4 file"
            );
            assert!(bytes.len() > 4096, "switch {bits}: {} bytes", bytes.len());
        }
        // A skinless, textureless write must not leave a texture folder behind.
        assert!(!folder.join("switch-0_textures").exists());
        std::fs::remove_dir_all(&folder).ok();
    }
}
