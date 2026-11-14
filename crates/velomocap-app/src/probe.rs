//! Off-screen render probe.
//!
//! Renders the SMPL-H rest pose through the same `Canvas` the viewport uses and
//! writes a PPM. It exists because the window cannot always be brought to the
//! front for a screenshot, and shading bugs are much easier to see on a still
//! image than argued about in prose.

use std::io::Write;

use velomocap_body::{BodyModel, Pose};

use crate::render::{self, Camera, Canvas};

pub fn render_rest_pose(model_path: &str, out: &str) -> anyhow::Result<()> {
    let body = BodyModel::load(model_path, 16)?.with_posedirs()?;
    // `VELOMOCAP_PROBE_VERTS=<file>` renders an already-skinned mesh (f32 LE,
    // 6890 x 3) instead of posing the model here: this is what the viewport
    // actually shows for a generated clip.
    let from_file = std::env::var("VELOMOCAP_PROBE_VERTS").ok();
    if let Some(file) = &from_file {
        let bytes = std::fs::read(file)?;
        let verts: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        eprintln!("render-probe: {} vertices from {}", verts.len() / 3, file);
        return draw(&body, verts, out);
    }
    let mut pose = Pose::neutral(body.num_joints, 16);
    // `VELOMOCAP_PROBE_POSE=bend` puts a real bend on the elbows, knees and
    // fingers, so the skinning path is exercised and not just the rest pose.
    if std::env::var("VELOMOCAP_PROBE_POSE").as_deref() == Ok("bend") {
        let bend = |pose: &mut Pose, joint: usize, axis: usize, angle: f32| {
            let at = (joint - 1) * 3 + axis;
            if at < pose.body.len() {
                pose.body[at] = angle;
            }
        };
        bend(&mut pose, 18, 2, 1.4);
        bend(&mut pose, 19, 2, -1.4);
        bend(&mut pose, 4, 0, 0.9);
        bend(&mut pose, 5, 0, -0.6);
        for joint in 22..=36 {
            bend(&mut pose, joint, 2, 0.5);
        }
        for joint in 37..=51 {
            bend(&mut pose, joint, 2, -0.5);
        }
        eprintln!("render-probe: posed variant");
    }
    let (_skeleton, verts) = body.forward_mesh(&pose)?;
    draw(&body, verts, out)
}

fn draw(body: &BodyModel, verts: Vec<f32>, out: &str) -> anyhow::Result<()> {
    let faces: Vec<[u32; 3]> = body
        .faces
        .chunks_exact(3)
        .map(|face| [face[0], face[1], face[2]])
        .collect();
    eprintln!(
        "render-probe: {} verts, {} faces",
        verts.len() / 3,
        faces.len()
    );

    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for point in verts.chunks_exact(3) {
        for axis in 0..3 {
            lo[axis] = lo[axis].min(point[axis]);
            hi[axis] = hi[axis].max(point[axis]);
        }
    }
    let target = [
        (lo[0] + hi[0]) * 0.5,
        (lo[1] + hi[1]) * 0.5,
        (lo[2] + hi[2]) * 0.5,
    ];
    let radius = render::length(render::sub(hi, lo)).max(1.0);
    let size = [640.0f32, 800.0f32];
    let camera = Camera::orbit(target, 0.35, 0.05, radius * 1.3, size);
    let mut canvas = Canvas::new(size[0] as usize, size[1] as usize);
    canvas.clear_gradient(
        [0x22, 0x24, 0x28],
        [0x0e, 0x0f, 0x12],
    );

    let mut normals = vec![[0.0f32; 3]; verts.len() / 3];
    for face in &faces {
        let a = vertex(&verts, face[0]);
        let b = vertex(&verts, face[1]);
        let c = vertex(&verts, face[2]);
        let normal = render::cross(render::sub(b, a), render::sub(c, a));
        for index in face {
            let slot = &mut normals[*index as usize];
            slot[0] += normal[0];
            slot[1] += normal[1];
            slot[2] += normal[2];
        }
    }
    for normal in &mut normals {
        *normal = render::normalise(*normal);
    }

    let light = [0.35, 0.85, 0.4];
    for face in &faces {
        let a = vertex(&verts, face[0]);
        let b = vertex(&verts, face[1]);
        let c = vertex(&verts, face[2]);
        let n = [
            normals[face[0] as usize],
            normals[face[1] as usize],
            normals[face[2] as usize],
        ];
        canvas.triangle_gouraud(&camera, [a, b, c], n, [0x9c, 0xc2, 0xf0], light);
    }

    let image = canvas.to_image();
    let mut file = std::fs::File::create(out)?;
    write!(file, "P6\n{} {}\n255\n", image.width(), image.height())?;
    for pixel in &image.pixels {
        file.write_all(&[pixel.r(), pixel.g(), pixel.b()])?;
    }
    eprintln!("render-probe: wrote {out}");
    Ok(())
}

fn vertex(verts: &[f32], index: u32) -> [f32; 3] {
    let at = index as usize * 3;
    [verts[at], verts[at + 1], verts[at + 2]]
}
