//! Binary FBX 7.4 skeleton/animation writer.
//!
//! Port of `mocap/fbx_writer.py`: no SDK, no Blender runtime.  Node defaults
//! carry the registered rest pose, `PreRotation` carries its axes, and only
//! Hips has animated translation.  End markers preserve terminal lengths.
//!
//! The Python version spends most of its time in per-property `struct.pack`
//! calls and in SciPy Euler conversions; this port streams into one buffer and
//! batches the Euler work through the C++ kernels.

use std::io::Write;
use std::path::Path;

use flate2::write::ZlibEncoder;
use flate2::Compression;

use crate::error::{Error, Result};
use crate::math::{self, mat3, mul3, tmul3, transpose3};

/// FBX key-time ticks per second.
pub const TICKS: f64 = 46_186_158_000.0;

/// Full transform applied to the root motion before writing.
pub const IDENTITY4: [f64; 16] = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
];

/// One FBX property. The binary encoding of a node is its list of these.
#[derive(Clone, Debug)]
pub enum Prop {
    Str(String),
    I32(i32),
    I64(i64),
    F64(f64),
    I32Array(Vec<i32>),
    I64Array(Vec<i64>),
    F32Array(Vec<f32>),
    F64Array(Vec<f64>),
}

/// One FBX element: a name, its properties and its children.
///
/// Public because the skinned-asset writer builds the same container: the
/// element encoding (13-byte records, zlib arrays, the end markers) is the part
/// that is easy to get wrong and hard to notice, so there is one copy of it.
#[derive(Clone, Debug)]
pub struct Node {
    pub name: String,
    pub props: Vec<Prop>,
    pub children: Vec<Node>,
}

impl Node {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            props: Vec::new(),
            children: Vec::new(),
        }
    }

    pub fn with(name: &str, props: Vec<Prop>) -> Self {
        Self {
            name: name.to_string(),
            props,
            children: Vec::new(),
        }
    }

    pub fn child(mut self, node: Node) -> Self {
        self.children.push(node);
        self
    }
}

/// One `P` property node as produced by `fbx_writer.prop`.
pub fn property(name: &str, kind: &str, flags: &str, values: Vec<Prop>) -> Node {
    let subtype = match kind {
        "int" => "Integer",
        "double" => "Number",
        _ => "",
    };
    let mut props = vec![
        Prop::Str(name.to_string()),
        Prop::Str(kind.to_string()),
        Prop::Str(subtype.to_string()),
        Prop::Str(flags.to_string()),
    ];
    props.extend(values);
    Node::with("P", props)
}

pub fn property_int(name: &str, kind: &str, value: i32) -> Node {
    property(name, kind, "", vec![Prop::I32(value)])
}

pub fn property_kint(name: &str, value: i64) -> Node {
    property(name, "KTime", "", vec![Prop::I64(value)])
}

pub fn property_double(name: &str, value: f64) -> Node {
    property(name, "double", "", vec![Prop::F64(value)])
}

pub fn property_vec(name: &str, kind: &str, flags: &str, values: [f64; 3]) -> Node {
    property(
        name,
        kind,
        flags,
        vec![Prop::F64(values[0]), Prop::F64(values[1]), Prop::F64(values[2])],
    )
}

pub fn properties(nodes: Vec<Node>) -> Node {
    Node {
        name: "Properties70".to_string(),
        props: Vec::new(),
        children: nodes,
    }
}

fn compress(data: &[u8]) -> Result<Vec<u8>> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(1));
    encoder.write_all(data)?;
    Ok(encoder.finish()?)
}

/// The 27-byte signature every binary FBX opens with.
pub fn begin(out: &mut Vec<u8>) {
    out.extend_from_slice(b"Kaydara FBX Binary  \0\x1a\0");
    out.extend_from_slice(&7400u32.to_le_bytes());
}

/// The footer a 7.4 file closes with: a null record, the file id, an alignment
/// pad, the version and the trailing id block.
pub fn finish(out: &mut Vec<u8>) {
    out.extend_from_slice(&[0u8; 13]);
    out.extend_from_slice(&hex("fabcab09d0c8d466b176fb831cf7267e"));
    let pad = (16 - (out.len() % 16)) % 16;
    out.extend(std::iter::repeat_n(0u8, pad));
    out.extend_from_slice(&7400u32.to_le_bytes());
    out.extend_from_slice(&[0u8; 120]);
    out.extend_from_slice(&hex("f85a8c6adef5d97eece90ce3758f290b"));
}

fn write_property(out: &mut Vec<u8>, prop: &Prop) -> Result<()> {
    match prop {
        Prop::Str(value) => {
            out.push(b'S');
            out.extend_from_slice(&(value.len() as u32).to_le_bytes());
            out.extend_from_slice(value.as_bytes());
        }

        Prop::I32(value) => {
            out.push(b'I');
            out.extend_from_slice(&value.to_le_bytes());
        }
        Prop::I64(value) => {
            out.push(b'L');
            out.extend_from_slice(&value.to_le_bytes());
        }
        Prop::F64(value) => {
            out.push(b'D');
            out.extend_from_slice(&value.to_le_bytes());
        }
        Prop::I32Array(values) => {
            let mut raw = Vec::with_capacity(values.len() * 4);
            for v in values {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            let packed = compress(&raw)?;
            out.push(b'i');
            out.extend_from_slice(&(values.len() as u32).to_le_bytes());
            out.extend_from_slice(&1u32.to_le_bytes());
            out.extend_from_slice(&(packed.len() as u32).to_le_bytes());
            out.extend_from_slice(&packed);
        }
        Prop::I64Array(values) => {
            let mut raw = Vec::with_capacity(values.len() * 8);
            for v in values {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            let packed = compress(&raw)?;
            out.push(b'l');
            out.extend_from_slice(&(values.len() as u32).to_le_bytes());
            out.extend_from_slice(&1u32.to_le_bytes());
            out.extend_from_slice(&(packed.len() as u32).to_le_bytes());
            out.extend_from_slice(&packed);
        }
        Prop::F32Array(values) => {
            let mut raw = Vec::with_capacity(values.len() * 4);
            for v in values {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            let packed = compress(&raw)?;
            out.push(b'f');
            out.extend_from_slice(&(values.len() as u32).to_le_bytes());
            out.extend_from_slice(&1u32.to_le_bytes());
            out.extend_from_slice(&(packed.len() as u32).to_le_bytes());
            out.extend_from_slice(&packed);
        }
        Prop::F64Array(values) => {
            let mut raw = Vec::with_capacity(values.len() * 8);
            for v in values {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            let packed = compress(&raw)?;
            out.push(b'd');
            out.extend_from_slice(&(values.len() as u32).to_le_bytes());
            out.extend_from_slice(&1u32.to_le_bytes());
            out.extend_from_slice(&(packed.len() as u32).to_le_bytes());
            out.extend_from_slice(&packed);
        }
    }
    Ok(())
}

pub fn write_element(out: &mut Vec<u8>, node: &Node) -> Result<()> {
    let start = out.len();
    out.extend_from_slice(&[0u8; 13]);
    out.extend_from_slice(node.name.as_bytes());
    let prop_start = out.len();
    for prop in &node.props {
        write_property(out, prop)?;
    }
    let prop_size = out.len() - prop_start;
    for child in &node.children {
        write_element(out, child)?;
    }
    // Every node's nested list is terminated by a NULL record, leaves included.
    // Writing it only for nodes with children left a leaf ending exactly at its
    // own end offset, which the FBX SDK tolerates but stricter importers read as
    // a child list that runs into the next sibling.
    out.extend_from_slice(&[0u8; 13]);
    let end = out.len();
    if end >= u32::MAX as usize {
        return Err(Error::new("FBX exceeds the 4 GiB format limit; split the clip"));
    }
    out[start..start + 4].copy_from_slice(&(end as u32).to_le_bytes());
    out[start + 4..start + 8].copy_from_slice(&(node.props.len() as u32).to_le_bytes());
    out[start + 8..start + 12].copy_from_slice(&(prop_size as u32).to_le_bytes());
    out[start + 12] = node.name.len() as u8;
    Ok(())
}

/// A connection record written into the trailing `Connections` node.
struct Connection {
    kind: &'static str,
    child: i32,
    parent: i32,
    property: Option<String>,
}

impl Connection {
    fn to_node(&self) -> Node {
        let mut props = vec![Prop::Str(self.kind.to_string()), Prop::I64(self.child as i64), Prop::I64(self.parent as i64)];
        if let Some(property) = &self.property {
            props.push(Prop::Str(property.clone()));
        }
        Node::with("C", props)
    }
}

/// Everything the writer needs, mirroring `fbx_writer.write`'s arguments.
pub struct FbxInput<'a> {
    pub names: &'a [String],
    pub parents: &'a [i32],
    /// `bones * 3` rest joint positions.
    pub rest: &'a [f64],
    /// `bones * 9` rest rotations used for node defaults.
    pub rest_frames: &'a [f64],
    /// `bones * 9` display basis used for animated channels.
    pub display: &'a [f64],
    /// `bones * 3` terminal offsets in local space.
    pub tip_local: &'a [f64],
    /// `frames * bones * 16` row-major 4x4 world matrices.
    pub frames: &'a [f64],
    pub fps: f64,
    pub transform: [f64; 16],
    pub prepend_tpose: bool,
    pub expected_bones: usize,
    pub rig_name: &'a str,
    pub creator: &'a str,
}

#[derive(Clone, Debug, Default)]
pub struct FbxReport {
    pub key_count: usize,
    pub stop_tick: i64,
    pub bones: usize,
}

/// Write one committed mapped take.
pub fn write(path: &Path, input: &FbxInput<'_>) -> Result<FbxReport> {
    validate(input)?;

    let bones = input.expected_bones;
    let frames = input.frames.len() / (bones * 16);
    let transform_rot = [
        input.transform[0], input.transform[1], input.transform[2],
        input.transform[4], input.transform[5], input.transform[6],
        input.transform[8], input.transform[9], input.transform[10],
    ];
    let transform_pos = [input.transform[3], input.transform[7], input.transform[11]];
    let transform_rot_t = transpose3(&transform_rot);

    // Source offsets use the pre-symmetric-bind rest data, as in the original.
    let source_offsets: Vec<[f64; 3]> = (0..bones)
        .map(|j| {
            let p = input.parents[j];
            if p < 0 {
                [
                    input.rest[j * 3],
                    input.rest[j * 3 + 1],
                    input.rest[j * 3 + 2],
                ]
            } else {
                let p = p as usize;
                let parent = transpose3(&input.rest_frames[p * 9..p * 9 + 9]);
                let delta = [
                    input.rest[j * 3] - input.rest[p * 3],
                    input.rest[j * 3 + 1] - input.rest[p * 3 + 1],
                    input.rest[j * 3 + 2] - input.rest[p * 3 + 2],
                ];
                let v = [
                    parent[0] * delta[0] + parent[1] * delta[1] + parent[2] * delta[2],
                    parent[3] * delta[0] + parent[4] * delta[1] + parent[5] * delta[2],
                    parent[6] * delta[0] + parent[7] * delta[1] + parent[8] * delta[2],
                ];
                v
            }
        })
        .collect();

    let ids: Vec<i32> = (0..bones).map(|j| 1000 + j as i32).collect();
    let mut connections: Vec<Connection> = Vec::with_capacity(bones * 8 + 16);
    let mut objects: Vec<Node> = Vec::with_capacity(bones * 4 + 16);

    connections.push(Connection {
        kind: "OO",
        child: 900,
        parent: 0,
        property: None,
    });
    objects.push(
        Node::with(
            "Model",
            vec![
                Prop::I64(900),
                Prop::Str(format!("{}\u{0}\u{1}Model", input.rig_name)),
                Prop::Str("Null".to_string()),
            ],
        )
        .child(Node::with("Version", vec![Prop::I64(232)]))
        .child(properties(Vec::new())),
    );

    let mut local_rest: Vec<[f64; 9]> = Vec::with_capacity(bones);
    for j in 0..bones {
        let p = input.parents[j];
        let (rotation, offset) = if p < 0 {
            (
                mat3(input.rest_frames, j).to_vec(),
                [
                    input.rest[j * 3],
                    input.rest[j * 3 + 1],
                    input.rest[j * 3 + 2],
                ],
            )
        } else {
            let p = p as usize;
            let parent = &input.rest_frames[p * 9..p * 9 + 9];
            let rotation = tmul3(parent, mat3(input.rest_frames, j)).to_vec();
            let delta = [
                input.rest[j * 3] - input.rest[p * 3],
                input.rest[j * 3 + 1] - input.rest[p * 3 + 1],
                input.rest[j * 3 + 2] - input.rest[p * 3 + 2],
            ];
            let offset = [
                parent[0] * delta[0] + parent[3] * delta[1] + parent[6] * delta[2],
                parent[1] * delta[0] + parent[4] * delta[1] + parent[7] * delta[2],
                parent[2] * delta[0] + parent[5] * delta[1] + parent[8] * delta[2],
            ];
            (rotation, offset)
        };
        let (angles, _) = math::mat_to_euler(&rotation, "XYZ");
        let prerotation = [
            angles[0].to_degrees(),
            angles[1].to_degrees(),
            angles[2].to_degrees(),
        ];
        let mut rotation9 = [0.0f64; 9];
        rotation9.copy_from_slice(&rotation);
        local_rest.push(rotation9);

        let name = &input.names[j];
        objects.push(
            Node::with(
                "Model",
                vec![
                    Prop::I64(ids[j] as i64),
                    Prop::Str(format!("{name}\u{0}\u{1}Model")),
                    Prop::Str("LimbNode".to_string()),
                ],
            )
            .child(Node::with("Version", vec![Prop::I64(232)]))
            .child(properties(vec![
                property_vec("Lcl Translation", "Lcl Translation", "A", offset),
                property_vec("Lcl Rotation", "Lcl Rotation", "A", [0.0, 0.0, 0.0]),
                property_vec("Lcl Scaling", "Lcl Scaling", "A", [1.0, 1.0, 1.0]),
                property_vec("PreRotation", "Vector3D", "", prerotation),
                // The reference writer stores bool-kind properties as int32.
                property_int("RotationActive", "bool", 1),
                property_int("RotationOrder", "enum", 0),
                property_int("InheritType", "enum", 1),
            ])),
        );
        objects.push(
            Node::with(
                "NodeAttribute",
                vec![
                    Prop::I64(2000 + j as i64),
                    Prop::Str(format!("{name}\u{0}\u{1}NodeAttribute")),
                    Prop::Str("LimbNode".to_string()),
                ],
            )
            .child(Node::with("TypeFlags", vec![Prop::Str("Skeleton".to_string())])),
        );
        connections.push(Connection {
            kind: "OO",
            child: 2000 + j as i32,
            parent: ids[j],
            property: None,
        });
        connections.push(Connection {
            kind: "OO",
            child: ids[j],
            parent: if p < 0 { 900 } else { ids[p as usize] },
            property: None,
        });
    }

    for j in 0..bones {
        if input.parents.contains(&(j as i32)) {
            continue;
        }
        let marker = 3000 + j as i32;
        let name = &input.names[j];
        let tip = [
            input.tip_local[j * 3],
            input.tip_local[j * 3 + 1],
            input.tip_local[j * 3 + 2],
        ];
        objects.push(
            Node::with(
                "Model",
                vec![
                    Prop::I64(marker as i64),
                    Prop::Str(format!("{name}_end\u{0}\u{1}Model")),
                    Prop::Str("LimbNode".to_string()),
                ],
            )
            .child(Node::with("Version", vec![Prop::I64(232)]))
            .child(properties(vec![property_vec(
                "Lcl Translation",
                "Lcl Translation",
                "A",
                tip,
            )])),
        );
        objects.push(
            Node::with(
                "NodeAttribute",
                vec![
                    Prop::I64(4000 + j as i64),
                    Prop::Str(format!("{name}_end\u{0}\u{1}NodeAttribute")),
                    Prop::Str("LimbNode".to_string()),
                ],
            )
            .child(Node::with("TypeFlags", vec![Prop::Str("Skeleton".to_string())])),
        );
        connections.push(Connection {
            kind: "OO",
            child: 4000 + j as i32,
            parent: marker,
            property: None,
        });
        connections.push(Connection {
            kind: "OO",
            child: marker,
            parent: ids[j],
            property: None,
        });
    }

    // Explicit bind-pose table in addition to the node defaults.
    let mut pose_nodes: Vec<Node> = Vec::with_capacity(bones + 1);
    pose_nodes.push(
        Node::with("PoseNode", Vec::new())
            .child(Node::with("Node", vec![Prop::I64(900)]))
            .child(Node::with(
                "Matrix",
                vec![Prop::F64Array(transpose_flatten(&IDENTITY4))],
            )),
    );
    for j in 0..bones {
        let mut matrix = [0.0f64; 16];
        matrix[15] = 1.0;
        for r in 0..3 {
            for c in 0..3 {
                matrix[r * 4 + c] = input.rest_frames[j * 9 + r * 3 + c];
            }
            matrix[r * 4 + 3] = input.rest[j * 3 + r];
        }
        pose_nodes.push(
            Node::with("PoseNode", Vec::new())
                .child(Node::with("Node", vec![Prop::I64(ids[j] as i64)]))
                .child(Node::with(
                    "Matrix",
                    vec![Prop::F64Array(transpose_flatten(&matrix))],
                )),
        );
    }
    objects.push(
        Node::with(
            "Pose",
            vec![
                Prop::I64(5000),
                Prop::Str("BindPose\u{0}\u{1}Pose".to_string()),
                Prop::Str("BindPose".to_string()),
            ],
        )
        .child(Node::with("Type", vec![Prop::Str("BindPose".to_string())]))
        .child(Node::with("Version", vec![Prop::I64(100)]))
        .child(Node::with(
            "NbPoseNodes",
            vec![Prop::I64(pose_nodes.len() as i64)],
        ))
        .children(pose_nodes),
    );

    let key_count = frames + usize::from(input.prepend_tpose);
    let times: Vec<i64> = (0..key_count)
        .map(|k| (k as f64 * TICKS / input.fps).round_ties_even() as i64)
        .collect();
    let stop = *times.last().unwrap_or(&0);

    objects.push(
        Node::with(
            "AnimationStack",
            vec![
                Prop::I64(6000),
                Prop::Str("Motion\u{0}\u{1}AnimStack".to_string()),
                Prop::Str(String::new()),
            ],
        )
        .child(properties(vec![
            property_kint("LocalStart", 0),
            property("LocalStop", "KTime", "", vec![Prop::I64(stop)]),
            property_kint("ReferenceStart", 0),
            property("ReferenceStop", "KTime", "", vec![Prop::I64(stop)]),
        ])),
    );
    objects.push(
        Node::with(
            "AnimationLayer",
            vec![
                Prop::I64(6001),
                Prop::Str("BaseLayer\u{0}\u{1}AnimLayer".to_string()),
                Prop::Str(String::new()),
            ],
        )
        .child(properties(Vec::new())),
    );
    connections.push(Connection {
        kind: "OO",
        child: 6001,
        parent: 6000,
        property: None,
    });

    // ---------------------------------------------------------------- file --
    let mut out: Vec<u8> = Vec::with_capacity(1 << 20);
    begin(&mut out);

    write_element(
        &mut out,
        &Node::new("FBXHeaderExtension")
            .child(Node::with("FBXHeaderVersion", vec![Prop::I64(1003)]))
            .child(Node::with("FBXVersion", vec![Prop::I64(7400)]))
            .child(Node::with("Creator", vec![Prop::Str(input.creator.to_string())])),
    )?;

    let mut settings: Vec<Node> = Vec::new();
    for (key, value) in [
        ("UpAxis", 1),
        ("UpAxisSign", 1),
        ("FrontAxis", 2),
        ("FrontAxisSign", 1),
        ("CoordAxis", 0),
        ("CoordAxisSign", 1),
    ] {
        settings.push(property_int(key, "int", value));
    }
    settings.push(property_double("UnitScaleFactor", 100.0));
    settings.push(property_double("OriginalUnitScaleFactor", 100.0));
    settings.push(property_int("TimeMode", "enum", 14));
    settings.push(property_double("CustomFrameRate", input.fps));
    settings.push(property_kint("TimeSpanStart", 0));
    settings.push(property("TimeSpanStop", "KTime", "", vec![Prop::I64(stop)]));
    write_element(
        &mut out,
        &Node::new("GlobalSettings")
            .child(Node::with("Version", vec![Prop::I64(1000)]))
            .child(properties(settings)),
    )?;

    write_element(
        &mut out,
        &Node::new("Documents")
            .child(Node::with("Count", vec![Prop::I64(1)]))
            .child(
                Node::with(
                    "Document",
                    vec![
                        Prop::I64(800),
                        Prop::Str("Scene".to_string()),
                        Prop::Str("Scene".to_string()),
                    ],
                )
                .child(properties(Vec::new()))
                .child(Node::with("RootNode", vec![Prop::I64(0)])),
            ),
    )?;
    write_element(&mut out, &Node::new("References"))?;
    write_element(
        &mut out,
        &Node::new("Definitions")
            .child(Node::with("Version", vec![Prop::I64(100)]))
            .child(Node::with(
                "Count",
                vec![Prop::I64(objects.len() as i64)],
            )),
    )?;

    // Objects is streamed: curve nodes are appended after the static objects.
    let objects_start = out.len();
    out.extend_from_slice(&[0u8; 13]);
    out.extend_from_slice(b"Objects");
    for object in &objects {
        write_element(&mut out, object)?;
    }
    let mut next_id = 10000i32;
    for j in 0..bones {
        let p = input.parents[j];
        let display = &input.display[j * 9..j * 9 + 9];
        let (local, translation) = if p < 0 {
            let mut positions = Vec::with_capacity(frames * 3);
            for t in 0..frames {
                let position = [
                    input.frames[(t * bones + j) * 16 + 3],
                    input.frames[(t * bones + j) * 16 + 7],
                    input.frames[(t * bones + j) * 16 + 11],
                ];
                positions.push(
                    position[0] * transform_rot_t[0]
                        + position[1] * transform_rot_t[1]
                        + position[2] * transform_rot_t[2]
                        + transform_pos[0],
                );
                positions.push(
                    position[0] * transform_rot_t[3]
                        + position[1] * transform_rot_t[4]
                        + position[2] * transform_rot_t[5]
                        + transform_pos[1],
                );
                positions.push(
                    position[0] * transform_rot_t[6]
                        + position[1] * transform_rot_t[7]
                        + position[2] * transform_rot_t[8]
                        + transform_pos[2],
                );
            }
            if !positions.iter().all(|v| v.is_finite()) {
                return Err(Error::new("Non-finite FBX root"));
            }
            if input.prepend_tpose {
                let head = positions[0..3].to_vec();
                let mut shifted = head;
                shifted.extend_from_slice(&positions);
                positions = shifted;
            }
            let mut rotations = Vec::with_capacity(frames * 9);
            for t in 0..frames {
                let world = mul3(&rot3(input.frames, (t * bones + j) * 16), display);
                let local = mul3(&transform_rot, &world);
                rotations.extend_from_slice(&local);
            }
            (rotations, Some(positions))
        } else {
            let p = p as usize;
            let parent_display = &input.display[p * 9..p * 9 + 9];
            let mut rotations = Vec::with_capacity(frames * 9);
            let mut worst = 0.0f64;
            for t in 0..frames {
                let world = mul3(&rot3(input.frames, (t * bones + j) * 16), display);
                let parent = mul3(&rot3(input.frames, (t * bones + p) * 16), parent_display);
                let local = tmul3(&parent, &world);
                rotations.extend_from_slice(&local);

                // Rigid-offset guard: the committed source animation must keep
                // every bone offset constant.
                let source_parent = mul3(
                    &rot3(input.frames, (t * bones + p) * 16),
                    &input.display[p * 9..p * 9 + 9],
                );
                let delta = [
                    input.frames[(t * bones + j) * 16 + 3] - input.frames[(t * bones + p) * 16 + 3],
                    input.frames[(t * bones + j) * 16 + 7] - input.frames[(t * bones + p) * 16 + 7],
                    input.frames[(t * bones + j) * 16 + 11] - input.frames[(t * bones + p) * 16 + 11],
                ];
                let actual = [
                    source_parent[0] * delta[0] + source_parent[3] * delta[1] + source_parent[6] * delta[2],
                    source_parent[1] * delta[0] + source_parent[4] * delta[1] + source_parent[7] * delta[2],
                    source_parent[2] * delta[0] + source_parent[5] * delta[1] + source_parent[8] * delta[2],
                ];
                for axis in 0..3 {
                    worst = worst.max((actual[axis] - source_offsets[j][axis]).abs());
                }
            }
            if !rotations.iter().all(|v| v.is_finite()) || worst > 2e-5 {
                return Err(Error::new(
                    "Non-rigid bone offset in committed source animation",
                ));
            }
            (rotations, None)
        };

        let mut local = local;
        if input.prepend_tpose {
            let pose = if p < 0 {
                mul3(&transform_rot, display)
            } else {
                let parent_display = &input.display[p as usize * 9..p as usize * 9 + 9];
                tmul3(parent_display, display)
            };
            let mut shifted = pose.to_vec();
            shifted.extend_from_slice(&local);
            local = shifted;
        }

        // angles = degrees(continuous(radians(euler_xyz(local_rest^T @ local))))
        let local_rest = &local_rest[j];
        let mut composed = vec![0.0; (local.len() / 9) * 9];
        for (t, chunk) in local.chunks_exact(9).enumerate() {
            let m = tmul3(local_rest, chunk);
            composed[t * 9..t * 9 + 9].copy_from_slice(&m);
        }
        // Reference chain: _euler() returns degrees, then the writer unwraps
        // radians and converts back. Each conversion is kept verbatim so the
        // rounding matches the original exactly.
        const DEG2RAD: f64 = std::f64::consts::PI / 180.0;
        const RAD2DEG: f64 = 180.0 / std::f64::consts::PI;
        let (mut angles, _) = math::mat_to_euler(&composed, "XYZ");
        for value in angles.iter_mut() {
            *value *= RAD2DEG; // scipy as_euler(degrees=True)
        }
        for value in angles.iter_mut() {
            *value *= DEG2RAD; // np.radians
        }
        math::continuous_angles(&mut angles);
        for value in angles.iter_mut() {
            *value *= RAD2DEG; // np.degrees
        }

        // The reference writer emits the root's translation curves before its
        // rotation curves; every other bone only has rotation channels.
        if let Some(positions) = &translation {
            write_curves(positions, &times, &mut next_id, &mut out, &mut connections, ids[j], "Lcl Translation")?;
        }
        write_curves(&angles, &times, &mut next_id, &mut out, &mut connections, ids[j], "Lcl Rotation")?;
    }
    out.extend_from_slice(&[0u8; 13]);
    let objects_end = out.len();
    out[objects_start..objects_start + 4].copy_from_slice(&(objects_end as u32).to_le_bytes());
    out[objects_start + 4..objects_start + 8].copy_from_slice(&0u32.to_le_bytes());
    out[objects_start + 8..objects_start + 12].copy_from_slice(&0u32.to_le_bytes());
    out[objects_start + 12] = 7;

    let connection_nodes: Vec<Node> = connections.iter().map(Connection::to_node).collect();
    write_element(
        &mut out,
        &Node {
            name: "Connections".to_string(),
            props: Vec::new(),
            children: connection_nodes,
        },
    )?;
    write_element(
        &mut out,
        &Node::new("Takes").child(Node::with("Current", vec![Prop::Str("Motion".to_string())])),
    )?;

    finish(&mut out);
    std::fs::write(path, &out)?;
    Ok(FbxReport {
        key_count,
        stop_tick: stop,
        bones,
    })
}

/// Stream one curve node plus its three axis curves into the output.
fn write_curves(
    values: &[f64],
    times: &[i64],
    next_id: &mut i32,
    out: &mut Vec<u8>,
    connections: &mut Vec<Connection>,
    bone_id: i32,
    kind: &'static str,
) -> Result<()> {
    let keys = values.len() / 3;
    let curve_id = *next_id;
    *next_id += 1;
    let node_name = if kind.ends_with("Translation") {
        "T\u{0}\u{1}AnimCurveNode"
    } else {
        "R\u{0}\u{1}AnimCurveNode"
    };
    let defaults: Vec<Node> = ["X", "Y", "Z"]
        .iter()
        .map(|axis| property(&format!("d|{axis}"), "Number", "", vec![Prop::F64(0.0)]))
        .collect();
    write_element(
        out,
        &Node::with(
            "AnimationCurveNode",
            vec![
                Prop::I64(curve_id as i64),
                Prop::Str(node_name.to_string()),
                Prop::Str(String::new()),
            ],
        )
        .child(properties(defaults)),
    )?;
    connections.push(Connection {
        kind: "OO",
        child: curve_id,
        parent: 6001,
        property: None,
    });
    connections.push(Connection {
        kind: "OP",
        child: curve_id,
        parent: bone_id,
        property: Some(kind.to_string()),
    });

    for (axis, letter) in ["X", "Y", "Z"].iter().enumerate() {
        let uid = *next_id;
        *next_id += 1;
        let axis_values: Vec<f32> = (0..keys).map(|k| values[k * 3 + axis] as f32).collect();
        write_element(
            out,
            &Node::with(
                "AnimationCurve",
                vec![Prop::I64(uid as i64), Prop::Str("\u{0}\u{1}AnimCurve".to_string()), Prop::Str(String::new())],
            )
            .child(Node::with("Default", vec![Prop::F64(values[axis])]))
            .child(Node::with("KeyVer", vec![Prop::I64(4008)]))
            .child(Node::with("KeyTime", vec![Prop::I64Array(times.to_vec())]))
            .child(Node::with("KeyValueFloat", vec![Prop::F32Array(axis_values)]))
            .child(Node::with("KeyAttrFlags", vec![Prop::I32Array(vec![4])]))
            .child(Node::with(
                "KeyAttrDataFloat",
                vec![Prop::F32Array(vec![0.0; 4])],
            ))
            .child(Node::with(
                "KeyAttrRefCount",
                vec![Prop::I32Array(vec![keys as i32])],
            )),
        )?;
        connections.push(Connection {
            kind: "OP",
            child: uid,
            parent: curve_id,
            property: Some(format!("d|{letter}")),
        });
    }
    Ok(())
}

/// Extract the upper-left 3x3 of a row-major 4x4 stored at `base`.
#[inline]
fn rot3(frames: &[f64], base: usize) -> [f64; 9] {
    [
        frames[base],
        frames[base + 1],
        frames[base + 2],
        frames[base + 4],
        frames[base + 5],
        frames[base + 6],
        frames[base + 8],
        frames[base + 9],
        frames[base + 10],
    ]
}

pub fn transpose_flatten(matrix: &[f64; 16]) -> Vec<f64> {    let mut out = Vec::with_capacity(16);
    for r in 0..4 {
        for c in 0..4 {
            out.push(matrix[c * 4 + r]);
        }
    }
    out
}

fn hex(text: &str) -> Vec<u8> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let mut i = 0;
    while i + 1 < bytes.len() {
        let hi = (bytes[i] as char).to_digit(16).unwrap_or(0) as u8;
        let lo = (bytes[i + 1] as char).to_digit(16).unwrap_or(0) as u8;
        out.push(hi * 16 + lo);
        i += 2;
    }
    out
}

fn validate(input: &FbxInput<'_>) -> Result<()> {
    let bones = input.expected_bones;
    let frames = input.frames.len() / (bones.max(1) * 16);
    if frames < 1
        || input.frames.len() != frames * bones * 16
        || !input.fps.is_finite()
        || input.fps <= 0.0
    {
        return Err(Error::new("Invalid FBX frame dimensions/timing"));
    }
    // The root is bone 0 and is the one that carries the animated translation;
    // its *name* is whatever the rig calls it (`Hips`, `pelvis`, `センター`), so
    // only the shape is checked here. The names themselves are validated below.
    if input.names.len() != bones
        || input.parents.len() != bones
        || input.parents[0] != -1
        || input.names.iter().collect::<std::collections::HashSet<_>>().len() != bones
    {
        return Err(Error::new(format!("Expected a {bones}-bone hierarchy")));
    }
    if input.rest.len() != bones * 3
        || input.rest_frames.len() != bones * 9
        || input.display.len() != bones * 9
        || input.tip_local.len() != bones * 3
    {
        return Err(Error::new("Invalid FBX rest dimensions"));
    }
    for j in 1..bones {
        let p = input.parents[j];
        if p < 0 || (p as usize) >= j {
            return Err(Error::new("Invalid FBX parents"));
        }
    }
    for slice in [input.rest, input.rest_frames, input.display, input.tip_local, input.frames] {
        if !slice.iter().all(|v| v.is_finite()) {
            return Err(Error::new("Invalid FBX rest data"));
        }
    }
    for j in 0..bones {
        let m = &input.rest_frames[j * 9..j * 9 + 9];
        let mt = transpose3(m);
        let product = mul3(&mt, m);
        for k in 0..9 {
            let expected = if k % 4 == 0 { 1.0 } else { 0.0 };
            if (product[k] - expected).abs() > 1e-5 {
                return Err(Error::new("Invalid FBX rest rotation"));
            }
        }
        let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6]);
        if det < 0.99999 {
            return Err(Error::new("Invalid FBX rest rotation"));
        }
    }
    // An FBX string is length-prefixed, so a `:` in a bone name is harmless —
    // Mixamo's own files write `mixamorig:LeftArm`. Non-ASCII names stay out:
    // the FBX importers this has been checked against are less predictable about
    // them, and the only rig that needs them (MMD) is a BVH/PMX target.
    if !input
        .names
        .iter()
        .all(|n| n.is_ascii() && !n.contains('\u{0}') && !n.contains('\u{1}'))
    {
        return Err(Error::new("Invalid FBX bone name"));
    }
    Ok(())
}

impl Node {
    pub fn children(mut self, nodes: Vec<Node>) -> Self {
        self.children.extend(nodes);
        self
    }
}
