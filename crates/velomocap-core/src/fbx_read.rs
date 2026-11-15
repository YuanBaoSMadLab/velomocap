//! Reading binary FBX, for importing a character.
//!
//! The writer in [`crate::fbx`] produces 7.4 files; a character from Mixamo, a
//! game engine or DCC is also a binary FBX, usually with compressed array
//! properties. This walks the node tree the way the FBX SDK does: a node's
//! children are read up to its own end offset, and a NULL record inside that
//! range ends the list.
//!
//! Only the container is read here. Turning `Objects` into a skeleton and a
//! skinned mesh is the importer's job (see the character module in the app).

use flate2::read::ZlibDecoder;
use std::io::Read;

#[derive(Clone, Debug)]
pub enum Prop {
    Bool(bool),
    I32(i32),
    F32(f32),
    F64(f64),
    I64(i64),
    Str(String),
    I32s(Vec<i32>),
    F32s(Vec<f32>),
    F64s(Vec<f64>),
    I64s(Vec<i64>),
    Bools(Vec<bool>),
    Raw(Vec<u8>),
}

impl Prop {
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Prop::I32(v) => Some(*v as i64),
            Prop::I64(v) => Some(*v),
            Prop::F32(v) => Some(*v as i64),
            Prop::F64(v) => Some(*v as i64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Prop::F32(v) => Some(*v as f64),
            Prop::F64(v) => Some(*v),
            Prop::I32(v) => Some(*v as f64),
            Prop::I64(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Prop::Str(text) => Some(text),
            _ => None,
        }
    }

    pub fn as_f64s(&self) -> &[f64] {
        match self {
            Prop::F64s(values) => values,
            _ => &[],
        }
    }

    pub fn as_i32s(&self) -> &[i32] {
        match self {
            Prop::I32s(values) => values,
            _ => &[],
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Node {
    pub name: String,
    pub props: Vec<Prop>,
    pub children: Vec<Node>,
}

impl Node {
    pub fn child(&self, name: &str) -> Option<&Node> {
        self.children.iter().find(|node| node.name == name)
    }

    pub fn children_named<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Node> + 'a {
        self.children.iter().filter(move |node| node.name == name)
    }

    pub fn i64_at(&self, index: usize) -> Option<i64> {
        self.props.get(index).and_then(Prop::as_i64)
    }

    pub fn f64_at(&self, index: usize) -> Option<f64> {
        self.props.get(index).and_then(Prop::as_f64)
    }

    pub fn str_at(&self, index: usize) -> Option<&str> {
        self.props.get(index).and_then(Prop::as_str)
    }
}

#[derive(Clone, Debug)]
pub struct Document {
    pub version: u32,
    pub nodes: Vec<Node>,
}

impl Document {
    pub fn child(&self, name: &str) -> Option<&Node> {
        self.nodes.iter().find(|node| node.name == name)
    }
}

/// Parse a binary FBX into its node tree.
pub fn parse(bytes: &[u8]) -> Result<Document, String> {
    if bytes.len() < 27 || !bytes.starts_with(b"Kaydara FBX Binary  \0\x1a\0") {
        return Err("not a binary FBX (the ASCII form is not supported)".into());
    }
    let version = u32::from_le_bytes([bytes[23], bytes[24], bytes[25], bytes[26]]);
    let mut reader = Reader {
        data: bytes,
        offset: 27,
    };
    let nodes = reader.list(bytes.len())?;
    Ok(Document { version, nodes })
}

struct Reader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    /// Read the node list of one parent, bounded by `end`.
    fn list(&mut self, end: usize) -> Result<Vec<Node>, String> {
        let mut nodes = Vec::new();
        while self.offset < end {
            if self.offset + 13 > self.data.len() {
                return Err(format!("truncated record at {}", self.offset));
            }
            let header = &self.data[self.offset..self.offset + 13];
            if header.iter().all(|byte| *byte == 0) {
                self.offset += 13;
                return Ok(nodes);
            }
            let node_end = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
            let properties = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
            let name_len = header[12] as usize;
            if node_end > end || node_end < self.offset + 13 {
                return Err(format!("node end {node_end} outside [{} , {end}]", self.offset));
            }
            let name_start = self.offset + 13;
            let name = String::from_utf8_lossy(&self.data[name_start..name_start + name_len])
                .to_string();
            self.offset = name_start + name_len;
            let mut props = Vec::with_capacity(properties);
            for _ in 0..properties {
                props.push(self.property()?);
            }
            let children = self.list(node_end)?;
            if self.offset != node_end {
                return Err(format!(
                    "{name}: children ended at {}, header says {node_end}",
                    self.offset
                ));
            }
            nodes.push(Node {
                name,
                props,
                children,
            });
        }
        Ok(nodes)
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], String> {
        if self.offset + count > self.data.len() {
            return Err(format!("truncated property at {}", self.offset));
        }
        let slice = &self.data[self.offset..self.offset + count];
        self.offset += count;
        Ok(slice)
    }

    fn property(&mut self) -> Result<Prop, String> {
        let kind = self.take(1)?[0];
        let prop = match kind {
            b'Y' => Prop::I32(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i32),
            b'C' => Prop::Bool(self.take(1)?[0] != 0),
            b'I' => Prop::I32(i32::from_le_bytes(self.take(4)?.try_into().unwrap())),
            b'F' => Prop::F32(f32::from_le_bytes(self.take(4)?.try_into().unwrap())),
            b'D' => Prop::F64(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            b'L' => Prop::I64(i64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            b'S' | b'R' => {
                let len = u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize;
                let raw = self.take(len)?;
                if kind == b'S' {
                    Prop::Str(String::from_utf8_lossy(raw).to_string())
                } else {
                    Prop::Raw(raw.to_vec())
                }
            }
            b'f' | b'd' | b'l' | b'i' | b'b' => {
                let count = u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize;
                let encoding = u32::from_le_bytes(self.take(4)?.try_into().unwrap());
                let stored = u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize;
                let raw = self.take(stored)?.to_vec();
                let data = if encoding == 1 {
                    let mut out = Vec::new();
                    ZlibDecoder::new(raw.as_slice())
                        .read_to_end(&mut out)
                        .map_err(|error| format!("zlib: {error}"))?;
                    out
                } else {
                    raw
                };
                array(kind, count, &data)
            }
            other => return Err(format!("unknown property type {:?}", other as char)),
        };
        Ok(prop)
    }
}

fn array(kind: u8, count: usize, data: &[u8]) -> Prop {
    let width = match kind {
        b'f' | b'i' => 4,
        b'd' | b'l' => 8,
        _ => 1,
    };
    let count = count.min(data.len() / width.max(1));
    match kind {
        b'f' => Prop::F32s(
            (0..count)
                .map(|index| f32::from_le_bytes(data[index * 4..index * 4 + 4].try_into().unwrap()))
                .collect(),
        ),
        b'd' => Prop::F64s(
            (0..count)
                .map(|index| f64::from_le_bytes(data[index * 8..index * 8 + 8].try_into().unwrap()))
                .collect(),
        ),
        b'i' => Prop::I32s(
            (0..count)
                .map(|index| i32::from_le_bytes(data[index * 4..index * 4 + 4].try_into().unwrap()))
                .collect(),
        ),
        b'l' => Prop::I64s(
            (0..count)
                .map(|index| i64::from_le_bytes(data[index * 8..index * 8 + 8].try_into().unwrap()))
                .collect(),
        ),
        _ => Prop::Bools((0..count).map(|index| data[index] != 0).collect()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The writer and the reader have to agree: write a two-bone clip, read it
    /// back, and check the parts the character importer needs.
    #[test]
    fn a_written_fbx_reads_back() {
        let names = vec!["Hips".to_string(), "Spine".to_string()];
        let parents = vec![-1i32, 0];
        let rest = vec![0.0, 0.0, 0.0, 0.1, 0.2, 0.0];
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let rest_frames: Vec<f64> = identity.iter().copied().chain(identity.iter().copied()).collect();
        let display = rest_frames.clone();
        let tip_local = vec![0.0f64; 6];
        let mut frames = Vec::with_capacity(3 * 2 * 16);
        for _ in 0..3 {
            for bone in 0..2 {
                let at = if bone == 0 { [0.0, 0.0, 0.0] } else { [0.1, 0.2, 0.0] };
                frames.extend_from_slice(&[
                    1.0, 0.0, 0.0, at[0],
                    0.0, 1.0, 0.0, at[1],
                    0.0, 0.0, 1.0, at[2],
                    0.0, 0.0, 0.0, 1.0,
                ]);
            }
        }
        let transform = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let input = crate::fbx::FbxInput {
            names: &names,
            parents: &parents,
            rest: &rest,
            rest_frames: &rest_frames,
            display: &display,
            tip_local: &tip_local,
            frames: &frames,
            fps: 30.0,
            transform,
            prepend_tpose: false,
            expected_bones: 2,
            rig_name: "Test",
            creator: "test",
        };
        let path = std::env::temp_dir().join("velomocap-readback.fbx");
        crate::fbx::write(&path, &input).expect("write");
        let bytes = std::fs::read(&path).expect("read");
        let document = parse(&bytes).expect("parse");
        assert_eq!(document.version, 7400);
        let objects = document.child("Objects").expect("Objects");
        // A model's name property is `name\0\x01Class`.
        let names: Vec<&str> = objects
            .children_named("Model")
            .filter_map(|node| node.str_at(1))
            .filter_map(|raw| raw.split('\0').next())
            .collect();
        assert!(names.contains(&"Hips"), "{names:?}");
        assert!(names.contains(&"Spine"), "{names:?}");
        assert!(document.child("Connections").is_some());
        assert!(document.child("Takes").is_some());
        let _ = std::fs::remove_file(&path);
    }
}
