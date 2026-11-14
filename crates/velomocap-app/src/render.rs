//! Software rasteriser for the 3D viewport.
//!
//! The Qt application draws its viewport with QOpenGLWidget. This build has no
//! GL context of its own, and the scene is a few shaded bodies, so the whole
//! frame is rasterised on the CPU into a colour + depth buffer and handed to
//! egui as one texture. That gives a real z-buffer (so a mesh is solid, not
//! painter-sorted), flat shading and hidden-line removal for free, and it runs
//! on every backend egui supports.

use eframe::egui;

/// A tiny camera: a pinhole at `eye` looking at `forward`, y-up.
#[derive(Clone, Debug)]
pub struct Camera {
    pub eye: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub forward: [f32; 3],
    /// Focal length in pixels.
    pub focal: f32,
    /// Principal point in pixels.
    pub centre: [f32; 2],
    pub near: f32,
    pub far: f32,
}

/// Cut a segment at the camera's near plane.
///
/// A floor line that runs from in front of the camera to behind it used to be
/// dropped whole (`project` returns `None` for any point behind the eye), which
/// is why the grid lost every line that ran away from the viewer and kept only
/// the ones parallel to the image plane.
fn clip_near(camera: &Camera, from: [f32; 3], to: [f32; 3]) -> Option<([f32; 3], [f32; 3])> {
    let depth = |point: [f32; 3]| dot(sub(point, camera.eye), camera.forward);
    let near = camera.near.max(1e-4);
    let (da, db) = (depth(from), depth(to));
    if da <= near && db <= near {
        return None;
    }
    let mix = |a: [f32; 3], b: [f32; 3], t: f32| -> [f32; 3] {
        [
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        ]
    };
    if da > near && db > near {
        Some((from, to))
    } else if da > near {
        let t = (da - near) / (da - db);
        Some((from, mix(from, to, t)))
    } else {
        let t = (db - near) / (db - da);
        Some((mix(to, from, t), to))
    }
}

impl Camera {
    /// Build an orbit camera around `target`.
    pub fn orbit(target: [f32; 3], yaw: f32, pitch: f32, distance: f32, size: [f32; 2]) -> Self {
        let (sin_yaw, cos_yaw) = yaw.sin_cos();
        let (sin_pitch, cos_pitch) = pitch.sin_cos();
        let direction = [
            cos_pitch * sin_yaw,
            sin_pitch,
            cos_pitch * cos_yaw,
        ];
        let eye = add(target, mul(direction, distance));
        let forward = normalise(sub(target, eye));
        // Right-handed basis: with `forward = -z` this gives `right = +x`, so the
        // screen is not mirrored. `cross(up, forward)` is the one that flips the
        // scene left-to-right (and makes every body face away).
        let right = normalise(cross(forward, [0.0, 1.0, 0.0]));
        let up = cross(right, forward);
        // 40° vertical field of view, like the Qt viewport's default.
        let focal = size[1] * 0.5 / (20.0f32.to_radians()).tan();
        Self {
            eye,
            right,
            up,
            forward,
            focal,
            centre: [size[0] * 0.5, size[1] * 0.5],
            near: 0.02,
            far: 250.0,
        }
    }

    /// Screen position and view depth, or `None` when behind the camera.
    pub fn project(&self, point: [f32; 3]) -> Option<([f32; 2], f32)> {
        let v = sub(point, self.eye);
        let depth = dot(v, self.forward);
        if depth < self.near || depth > self.far {
            return None;
        }
        let x = dot(v, self.right);
        let y = dot(v, self.up);
        Some((
            [
                self.centre[0] + self.focal * x / depth,
                self.centre[1] - self.focal * y / depth,
            ],
            depth,
        ))
    }
}

/// Colour + depth buffer.
pub struct Canvas {
    pub width: usize,
    pub height: usize,
    /// RGBA pixels, straight (unpremultiplied) alpha = 255.
    pub colour: Vec<[u8; 4]>,
    /// Inverse view depth per pixel; larger is nearer, `0.0` is empty.
    depth: Vec<f32>,
    clip: [usize; 4],
}

impl Canvas {
    pub fn new(width: usize, height: usize) -> Self {
        let mut canvas = Self {
            width: 0,
            height: 0,
            colour: Vec::new(),
            depth: Vec::new(),
            clip: [0, 0, 0, 0],
        };
        canvas.resize(width, height);
        canvas
    }

    pub fn resize(&mut self, width: usize, height: usize) {
        let (width, height) = (width.max(1), height.max(1));
        if width == self.width && height == self.height {
            return;
        }
        self.width = width;
        self.height = height;
        self.colour = vec![[0, 0, 0, 255]; width * height];
        self.depth = vec![0.0; width * height];
        self.clip = [0, 0, width, height];
    }

    /// Fill with a vertical gradient, the way the Qt viewport fades to the floor.
    pub fn clear_gradient(&mut self, top: [u8; 3], bottom: [u8; 3]) {
        for y in 0..self.height {
            let t = if self.height > 1 {
                y as f32 / (self.height - 1) as f32
            } else {
                0.0
            };
            let pixel = [
                lerp_u8(top[0], bottom[0], t),
                lerp_u8(top[1], bottom[1], t),
                lerp_u8(top[2], bottom[2], t),
                255,
            ];
            let row = &mut self.colour[y * self.width..(y + 1) * self.width];
            for value in row.iter_mut() {
                *value = pixel;
            }
            let depth = &mut self.depth[y * self.width..(y + 1) * self.width];
            depth.fill(0.0);
        }
    }

    #[inline]
    fn put(&mut self, x: i32, y: i32, colour: [u8; 3], inv_depth: f32) {
        if x < self.clip[0] as i32
            || y < self.clip[1] as i32
            || x >= self.clip[2] as i32
            || y >= self.clip[3] as i32
        {
            return;
        }
        let at = y as usize * self.width + x as usize;
        if inv_depth > self.depth[at] {
            self.depth[at] = inv_depth;
            self.colour[at] = [colour[0], colour[1], colour[2], 255];
        }
    }

    /// Rasterise one triangle with flat shading.
    ///
    /// `colour` is the base albedo; lighting is applied here so the caller only
    /// supplies the person's tint.
    pub fn triangle(
        &mut self,
        camera: &Camera,
        vertices: [[f32; 3]; 3],
        colour: [u8; 3],
        light: [f32; 3],
        _cull: bool,
    ) {
        let normal = normalise(cross(
            sub(vertices[1], vertices[0]),
            sub(vertices[2], vertices[0]),
        ));
        let shade = shading(normal, camera, light);
        self.fill(camera, vertices, [shade; 3], colour);
    }

    /// Rasterise one triangle with per-vertex shading (smooth bodies).
    pub fn triangle_gouraud(
        &mut self,
        camera: &Camera,
        vertices: [[f32; 3]; 3],
        normals: [[f32; 3]; 3],
        colour: [u8; 3],
        light: [f32; 3],
    ) {
        let shades = normals.map(|normal| shading(normal, camera, light));
        self.fill(camera, vertices, shades, colour);
    }

    /// Rasterise one triangle with a constant colour (the floor plane).
    pub fn triangle_flat(
        &mut self,
        camera: &Camera,
        vertices: [[f32; 3]; 3],
        shaded: [u8; 3],
        cull: bool,
    ) {
        let _ = cull;
        self.fill(camera, vertices, [1.0; 3], shaded);
    }

    fn fill(
        &mut self,
        camera: &Camera,
        vertices: [[f32; 3]; 3],
        shades: [f32; 3],
        colour: [u8; 3],
    ) {
        let Some((s0, w0)) = camera.project(vertices[0]) else {
            return;
        };
        let Some((s1, w1)) = camera.project(vertices[1]) else {
            return;
        };
        let Some((s2, w2)) = camera.project(vertices[2]) else {
            return;
        };
        let area = (s1[0] - s0[0]) * (s2[1] - s0[1]) - (s1[1] - s0[1]) * (s2[0] - s0[0]);
        if area.abs() < 1e-6 {
            return;
        }

        let inv0 = 1.0 / w0;
        let inv1 = 1.0 / w1;
        let inv2 = 1.0 / w2;

        let min_x = s0[0].min(s1[0]).min(s2[0]).floor().max(self.clip[0] as f32) as i32;
        let max_x = s0[0].max(s1[0]).max(s2[0]).ceil().min(self.clip[2] as f32 - 1.0) as i32;
        let min_y = s0[1].min(s1[1]).min(s2[1]).floor().max(self.clip[1] as f32) as i32;
        let max_y = s0[1].max(s1[1]).max(s2[1]).ceil().min(self.clip[3] as f32 - 1.0) as i32;
        if min_x > max_x || min_y > max_y {
            return;
        }

        let inverse = 1.0 / area;
        for y in min_y..=max_y {
            let py = y as f32 + 0.5;
            for x in min_x..=max_x {
                let px = x as f32 + 0.5;
                let w0b = ((s1[0] - px) * (s2[1] - py) - (s1[1] - py) * (s2[0] - px)) * inverse;
                if w0b < 0.0 {
                    continue;
                }
                let w1b = ((s2[0] - px) * (s0[1] - py) - (s2[1] - py) * (s0[0] - px)) * inverse;
                if w1b < 0.0 {
                    continue;
                }
                let w2b = 1.0 - w0b - w1b;
                if w2b < 0.0 {
                    continue;
                }
                let inv = w0b * inv0 + w1b * inv1 + w2b * inv2;
                let shade = w0b * shades[0] + w1b * shades[1] + w2b * shades[2];
                let shaded = [
                    (colour[0] as f32 * shade).clamp(0.0, 255.0) as u8,
                    (colour[1] as f32 * shade).clamp(0.0, 255.0) as u8,
                    (colour[2] as f32 * shade).clamp(0.0, 255.0) as u8,
                ];
                self.put(x, y, shaded, inv);
            }
        }
    }

    /// A depth-tested line, for the floor grid and the motion path.
    pub fn line(&mut self, camera: &Camera, from: [f32; 3], to: [f32; 3], colour: [u8; 3]) {
        let Some((from, to)) = clip_near(camera, from, to) else {
            return;
        };
        let (Some((a, wa)), Some((b, wb))) = (camera.project(from), camera.project(to)) else {
            return;
        };
        self.line_2d(a, wa, b, wb, colour);
    }

    /// A line that ignores depth, for world-space overlays drawn on top.
    pub fn line_overlay(&mut self, camera: &Camera, from: [f32; 3], to: [f32; 3], colour: [u8; 3]) {
        let Some((from, to)) = clip_near(camera, from, to) else {
            return;
        };
        let (Some((a, _)), Some((b, _))) = (camera.project(from), camera.project(to)) else {
            return;
        };
        let steps = (b[0] - a[0]).abs().max((b[1] - a[1]).abs()).max(1.0) as i32;
        for step in 0..=steps {
            let t = step as f32 / steps as f32;
            let x = a[0] + (b[0] - a[0]) * t;
            let y = a[1] + (b[1] - a[1]) * t;
            let xi = x as i32;
            let yi = y as i32;
            if xi >= 0 && yi >= 0 && xi < self.width as i32 && yi < self.height as i32 {
                let at = yi as usize * self.width + xi as usize;
                self.colour[at] = [colour[0], colour[1], colour[2], 255];
            }
        }
    }

    fn line_2d(
        &mut self,
        a: [f32; 2],
        wa: f32,
        b: [f32; 2],
        wb: f32,
        colour: [u8; 3],
    ) {
        let inv_a = 1.0 / wa;
        let inv_b = 1.0 / wb;
        let steps = (b[0] - a[0]).abs().max((b[1] - a[1]).abs()).max(1.0) as i32;
        for step in 0..=steps {
            let t = step as f32 / steps as f32;
            let x = a[0] + (b[0] - a[0]) * t;
            let y = a[1] + (b[1] - a[1]) * t;
            let inv = inv_a + (inv_b - inv_a) * t;
            self.put(x as i32, y as i32, colour, inv);
        }
    }

    /// Discard the depth buffer so an overlay pass can draw with its own test.
    pub fn begin_overlay(&mut self) {
        self.depth.fill(0.0);
    }

    pub fn to_image(&self) -> egui::ColorImage {
        let pixels = self
            .colour
            .iter()
            .map(|p| egui::Color32::from_rgba_premultiplied(p[0], p[1], p[2], p[3]))
            .collect();
        egui::ColorImage {
            size: [self.width, self.height],
            source_size: egui::vec2(self.width as f32, self.height as f32),
            pixels,
        }
    }
}

/// Blinn-ish lighting in `0.34..1`, never above 1 so a body does not clip white.
fn shading(normal: [f32; 3], camera: &Camera, light: [f32; 3]) -> f32 {
    let facing = -dot(normal, camera.forward).abs();
    let diffuse = dot(normal, normalise(light)).abs();
    let rim = (1.0 - facing).powf(2.5) * 0.18;
    (0.34 + 0.58 * diffuse + rim).min(1.0)
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t).clamp(0.0, 255.0) as u8
}

pub fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn mul(a: [f32; 3], factor: f32) -> [f32; 3] {
    [a[0] * factor, a[1] * factor, a[2] * factor]
}

pub fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn length(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}

pub fn normalise(a: [f32; 3]) -> [f32; 3] {
    let scale = length(a).max(1e-6);
    mul(a, 1.0 / scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real body, from the exported chain reference: 18 439 vertices shaped
    /// like an adult. Falls back to a cube when the fixture is absent.
    fn body_vertices() -> Vec<f32> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(std::path::Path::parent)
            .expect("workspace root");
        for candidate in [
            root.parent().unwrap().join("export/sam3d"),
            root.join("models/sam3d"),
        ] {
            let meta_path = candidate.join("sam3d_chain_ref.json");
            let bin_path = candidate.join("sam3d_chain_ref.bin");
            if !meta_path.is_file() || !bin_path.is_file() {
                continue;
            }
            let Ok(meta_text) = std::fs::read_to_string(&meta_path) else {
                continue;
            };
            let Ok(meta) = serde_json::from_str::<serde_json::Value>(&meta_text) else {
                continue;
            };
            let Ok(bytes) = std::fs::read(&bin_path) else {
                continue;
            };
            let floats: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let entry = &meta["tensors"]["verts"];
            let offset = entry["offset"].as_u64().unwrap_or(0) as usize;
            let len = entry["len"].as_u64().unwrap_or(0) as usize;
            if len > 0 {
                return floats[offset..offset + len]
                    .chunks_exact(3)
                    .flat_map(|p| [p[0] * 0.01, p[1] * 0.01, p[2] * 0.01])
                    .collect();
            }
        }
        Vec::new()
    }

    /// Cost of one viewport frame with four bodies: the number that decides
    /// whether playback stays interactive.
    #[test]
    fn rasterises_four_bodies_in_reasonable_time() {
        let body = body_vertices();
        if body.len() < 3 {
            eprintln!("skipping: no chain reference for vertex data");
            return;
        }
        let shapes = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(std::path::Path::parent)
            .expect("workspace root")
            .join("models/sam3d/sam3d_faces.bin");
        let Ok(bytes) = std::fs::read(&shapes) else {
            eprintln!("skipping: no mesh topology");
            return;
        };
        let faces: Vec<[u32; 3]> = bytes
            .chunks_exact(12)
            .map(|c| {
                let get = |at: usize| u32::from_le_bytes([c[at], c[at + 1], c[at + 2], c[at + 3]]);
                [get(0), get(4), get(8)]
            })
            .collect();

        let mut canvas = Canvas::new(1100, 800);
        let camera = Camera::orbit([0.0, 0.9, 8.0], std::f32::consts::PI, 0.08, 3.6, [1100.0, 800.0]);
        let started = std::time::Instant::now();
        let mut triangles = 0usize;
        for person in 0..4 {
            let offset = [person as f32 * 0.8 - 1.2, 0.0, 0.0];
            for face in &faces {
                let vertex = |index: u32| {
                    let at = index as usize * 3;
                    [
                        body[at] + offset[0],
                        body[at + 1] + offset[1],
                        body[at + 2] + offset[2],
                    ]
                };
                canvas.triangle(
                    &camera,
                    [vertex(face[0]), vertex(face[1]), vertex(face[2])],
                    [190, 196, 204],
                    [0.3, 0.8, 0.4],
                    false,
                );
                triangles += 1;
            }
        }
        let elapsed = started.elapsed().as_secs_f64();
        eprintln!(
            "{triangles} triangles at 1100x800: {:.1} ms",
            elapsed * 1000.0
        );
        assert!(elapsed < 0.25, "one frame took {elapsed:.3}s");
    }
}
