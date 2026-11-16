//! Software rasteriser for the 3D viewport.
//!
//! The Qt application draws its viewport with QOpenGLWidget. This build has no
//! GL context of its own, and the scene is a few shaded bodies, so the whole
//! frame is rasterised on the CPU into a colour + depth buffer and handed to
//! egui as one texture. That gives a real z-buffer (so a mesh is solid, not
//! painter-sorted), flat shading and hidden-line removal for free, and it runs
//! on every backend egui supports.

use eframe::egui;

use crate::character::Texture;

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

impl Camera {
    /// Build an orbit camera around `target`.
    pub fn orbit(target: [f32; 3], yaw: f32, pitch: f32, distance: f32, size: [f32; 2]) -> Self {
        let (sin_yaw, cos_yaw) = yaw.sin_cos();
        let (sin_pitch, cos_pitch) = pitch.sin_cos();
        let direction = [cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw];
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
    ///
    /// Only used for labels and name anchors; geometry goes through the
    /// near-plane clipping in [`Canvas`], which never produces a point the
    /// rasteriser cannot project.
    pub fn project(&self, point: [f32; 3]) -> Option<([f32; 2], f32)> {
        let v = to_view(self, point);
        // A point cut to exactly `near` must not be rejected by rounding, or a
        // caller that clips first loses it again here.
        if v[2] < self.near * 0.999 || v[2] > self.far {
            return None;
        }
        Some((to_screen(self, v), v[2]))
    }

    /// Metres per device pixel at `depth`. Used to keep the floor grid legible.
    pub fn metres_per_pixel(&self, depth: f32) -> f32 {
        depth.max(1e-3) / self.focal.max(1e-3)
    }
}

/// View space: x right, y up, z forward (the view depth).
fn to_view(camera: &Camera, point: [f32; 3]) -> [f32; 3] {
    let v = sub(point, camera.eye);
    [dot(v, camera.right), dot(v, camera.up), dot(v, camera.forward)]
}

/// Screen position of a view-space point; requires `z > 0`.
fn to_screen(camera: &Camera, v: [f32; 3]) -> [f32; 2] {
    let inv = 1.0 / v[2];
    [
        camera.centre[0] + camera.focal * v[0] * inv,
        camera.centre[1] - camera.focal * v[1] * inv,
    ]
}

/// Cut a view-space segment at the near plane.
///
/// A floor line that runs from in front of the camera to behind it used to be
/// dropped whole (`project` returns `None` for any point behind the eye), which
/// is why the grid lost every line that ran away from the viewer and kept only
/// the ones parallel to the image plane.
fn clip_segment(a: [f32; 3], b: [f32; 3], near: f32) -> Option<([f32; 3], [f32; 3])> {
    if a[2] < near && b[2] < near {
        return None;
    }
    if a[2] >= near && b[2] >= near {
        return Some((a, b));
    }
    let t = (near - a[2]) / (b[2] - a[2]);
    let cut = [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        near,
    ];
    if a[2] >= near {
        Some((a, cut))
    } else {
        Some((cut, b))
    }
}

/// Liang–Barsky clip of a screen-space segment to `[left, top, right, bottom]`.
///
/// Near-plane clipping leaves endpoints thousands of pixels away from the
/// canvas, and the sampler walks one step per pixel: without this the grid could
/// spend tens of millions of iterations on a single line and freeze the frame.
fn clip_to_rect(
    a: [f32; 2],
    b: [f32; 2],
    rect: [f32; 4],
) -> Option<(f32, f32)> {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for (p, q) in [
        (-dx, a[0] - rect[0]),
        (dx, rect[2] - a[0]),
        (-dy, a[1] - rect[1]),
        (dy, rect[3] - a[1]),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
            continue;
        }
        let r = q / p;
        if p < 0.0 {
            if r > t1 {
                return None;
            }
            if r > t0 {
                t0 = r;
            }
        } else {
            if r < t0 {
                return None;
            }
            if r < t1 {
                t1 = r;
            }
        }
    }
    (t0 < t1).then_some((t0, t1))
}

/// Sutherland–Hodgman against `z >= near` for one triangle.
///
/// One triangle cut by one plane is a triangle or a quad, so four slots are
/// enough. Vertices carry their shading so the clip can interpolate it.
fn clip_triangle_near(
    near: f32,
    triangle: [([f32; 3], f32); 3],
) -> ([([f32; 3], f32); 4], usize) {
    let mut out = [([0.0f32; 3], 0.0f32); 4];
    let mut count = 0usize;
    for index in 0..3 {
        let (a, shade_a) = triangle[index];
        let (b, shade_b) = triangle[(index + 1) % 3];
        let inside_a = a[2] >= near;
        let inside_b = b[2] >= near;
        if inside_a && count < 4 {
            out[count] = (a, shade_a);
            count += 1;
        }
        if inside_a != inside_b && count < 4 {
            let t = (near - a[2]) / (b[2] - a[2]);
            out[count] = (
                [
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                    near,
                ],
                shade_a + (shade_b - shade_a) * t,
            );
            count += 1;
        }
    }
    (out, count)
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

    #[inline]
    fn paint(&mut self, x: i32, y: i32, colour: [u8; 3]) {
        if x < self.clip[0] as i32
            || y < self.clip[1] as i32
            || x >= self.clip[2] as i32
            || y >= self.clip[3] as i32
        {
            return;
        }
        let at = y as usize * self.width + x as usize;
        self.colour[at] = [colour[0], colour[1], colour[2], 255];
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
        self.fill(camera, vertices, [shade; 3], colour, None);
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
        self.fill(camera, vertices, shades, colour, None);
    }

    /// Rasterise one triangle sampling a model's own texture: the image times
    /// the same lighting the clay bodies get.
    pub fn triangle_textured(
        &mut self,
        camera: &Camera,
        vertices: [[f32; 3]; 3],
        normals: [[f32; 3]; 3],
        uvs: [[f32; 2]; 3],
        texture: &Texture,
        light: [f32; 3],
    ) {
        let shades = normals.map(|normal| shading(normal, camera, light));
        self.fill(camera, vertices, shades, [255, 255, 255], Some((texture, uvs)));
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
        self.fill(camera, vertices, [1.0; 3], shaded, None);
    }

    fn fill(
        &mut self,
        camera: &Camera,
        vertices: [[f32; 3]; 3],
        shades: [f32; 3],
        colour: [u8; 3],
        texture: Option<(&Texture, [[f32; 2]; 3])>,
    ) {
        // Clip against the near plane *before* projecting. A corner behind the
        // eye used to drop the whole triangle, which is why the floor plane
        // popped out of the viewport whenever the camera got close to it.
        let view = vertices.map(|vertex| to_view(camera, vertex));
        let behind = view.iter().any(|point| point[2] <= camera.near);
        if let Some((texture, uvs)) = texture {
            if !behind {
                // Nothing to clip, so the texture coordinates stay valid.
                self.raster(
                    camera,
                    [
                        (view[0], shades[0]),
                        (view[1], shades[1]),
                        (view[2], shades[2]),
                    ],
                    colour,
                    Some((texture, uvs)),
                );
                return;
            }
            // A textured triangle crossing the near plane loses its texture for
            // that one frame rather than being dropped.
        }
        let triangle = [
            (view[0], shades[0]),
            (view[1], shades[1]),
            (view[2], shades[2]),
        ];
        let (polygon, count) = clip_triangle_near(camera.near, triangle);
        if count < 3 {
            return;
        }
        for index in 1..count - 1 {
            self.raster(
                camera,
                [polygon[0], polygon[index], polygon[index + 1]],
                colour,
                None,
            );
        }
    }

    /// Rasterise one already-clipped view-space triangle.
    fn raster(
        &mut self,
        camera: &Camera,
        triangle: [([f32; 3], f32); 3],
        colour: [u8; 3],
        texture: Option<(&Texture, [[f32; 2]; 3])>,
    ) {
        let s = [
            to_screen(camera, triangle[0].0),
            to_screen(camera, triangle[1].0),
            to_screen(camera, triangle[2].0),
        ];
        if !s.iter().all(|point| point[0].is_finite() && point[1].is_finite()) {
            return;
        }
        let area = (s[1][0] - s[0][0]) * (s[2][1] - s[0][1])
            - (s[1][1] - s[0][1]) * (s[2][0] - s[0][0]);
        if area.abs() < 1e-6 {
            return;
        }

        let inv = [
            1.0 / triangle[0].0[2],
            1.0 / triangle[1].0[2],
            1.0 / triangle[2].0[2],
        ];
        let shade = [triangle[0].1, triangle[1].1, triangle[2].1];

        let min_x = s[0][0].min(s[1][0]).min(s[2][0]).floor().max(self.clip[0] as f32) as i32;
        let max_x = s[0][0].max(s[1][0]).max(s[2][0]).ceil().min(self.clip[2] as f32 - 1.0) as i32;
        let min_y = s[0][1].min(s[1][1]).min(s[2][1]).floor().max(self.clip[1] as f32) as i32;
        let max_y = s[0][1].max(s[1][1]).max(s[2][1]).ceil().min(self.clip[3] as f32 - 1.0) as i32;
        if min_x > max_x || min_y > max_y {
            return;
        }

        let inverse = 1.0 / area;
        for y in min_y..=max_y {
            let py = y as f32 + 0.5;
            for x in min_x..=max_x {
                let px = x as f32 + 0.5;
                let w0 = ((s[1][0] - px) * (s[2][1] - py) - (s[1][1] - py) * (s[2][0] - px))
                    * inverse;
                if w0 < 0.0 {
                    continue;
                }
                let w1 = ((s[2][0] - px) * (s[0][1] - py) - (s[2][1] - py) * (s[0][0] - px))
                    * inverse;
                if w1 < 0.0 {
                    continue;
                }
                let w2 = 1.0 - w0 - w1;
                if w2 < 0.0 {
                    continue;
                }
                let depth = w0 * inv[0] + w1 * inv[1] + w2 * inv[2];
                let level = w0 * shade[0] + w1 * shade[1] + w2 * shade[2];
                let shaded = match &texture {
                    Some((texture, uvs)) => {
                        // Perspective-correct: interpolate uv/z, divide by 1/z.
                        let weight = [
                            w0 * inv[0] / depth,
                            w1 * inv[1] / depth,
                            w2 * inv[2] / depth,
                        ];
                        let u = weight[0] * uvs[0][0]
                            + weight[1] * uvs[1][0]
                            + weight[2] * uvs[2][0];
                        let v = weight[0] * uvs[0][1]
                            + weight[1] * uvs[1][1]
                            + weight[2] * uvs[2][1];
                        let texel = texture.sample(u, v);
                        [
                            (texel[0] * level * 255.0).clamp(0.0, 255.0) as u8,
                            (texel[1] * level * 255.0).clamp(0.0, 255.0) as u8,
                            (texel[2] * level * 255.0).clamp(0.0, 255.0) as u8,
                        ]
                    }
                    None => [
                        (colour[0] as f32 * level).clamp(0.0, 255.0) as u8,
                        (colour[1] as f32 * level).clamp(0.0, 255.0) as u8,
                        (colour[2] as f32 * level).clamp(0.0, 255.0) as u8,
                    ],
                };
                self.put(x, y, shaded, depth);
            }
        }
    }

    /// A depth-tested line, for the floor grid and the motion path.
    pub fn line(&mut self, camera: &Camera, from: [f32; 3], to: [f32; 3], colour: [u8; 3]) {
        let a = to_view(camera, from);
        let b = to_view(camera, to);
        let Some((a, b)) = clip_segment(a, b, camera.near) else {
            return;
        };
        let (sa, sb) = (to_screen(camera, a), to_screen(camera, b));
        let rect = [
            0.0,
            0.0,
            self.width as f32 - 1.0,
            self.height as f32 - 1.0,
        ];
        let Some((t0, t1)) = clip_to_rect(sa, sb, rect) else {
            return;
        };
        self.line_2d(sa, 1.0 / a[2], sb, 1.0 / b[2], t0, t1, colour);
    }

    /// A line that ignores depth, for world-space overlays drawn on top.
    pub fn line_overlay(&mut self, camera: &Camera, from: [f32; 3], to: [f32; 3], colour: [u8; 3]) {
        let a = to_view(camera, from);
        let b = to_view(camera, to);
        let Some((a, b)) = clip_segment(a, b, camera.near) else {
            return;
        };
        let (sa, sb) = (to_screen(camera, a), to_screen(camera, b));
        let rect = [
            0.0,
            0.0,
            self.width as f32 - 1.0,
            self.height as f32 - 1.0,
        ];
        let Some((t0, t1)) = clip_to_rect(sa, sb, rect) else {
            return;
        };
        let (x0, y0) = (
            sa[0] + (sb[0] - sa[0]) * t0,
            sa[1] + (sb[1] - sa[1]) * t0,
        );
        let (x1, y1) = (
            sa[0] + (sb[0] - sa[0]) * t1,
            sa[1] + (sb[1] - sa[1]) * t1,
        );
        let steps = (x1 - x0).abs().max((y1 - y0).abs()).max(1.0) as i32;
        for step in 0..=steps {
            let t = step as f32 / steps as f32;
            self.paint(
                (x0 + (x1 - x0) * t) as i32,
                (y0 + (y1 - y0) * t) as i32,
                colour,
            );
        }
    }

    /// Sample a screen-space segment between two parameters, depth-tested.
    ///
    /// `t0`/`t1` are the clipped range along `a..b`, so the walk never leaves
    /// the canvas no matter how far the near plane pushed the endpoints.
    #[allow(clippy::too_many_arguments)]
    fn line_2d(
        &mut self,
        a: [f32; 2],
        inv_a: f32,
        b: [f32; 2],
        inv_b: f32,
        t0: f32,
        t1: f32,
        colour: [u8; 3],
    ) {
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let (x0, y0) = (a[0] + dx * t0, a[1] + dy * t0);
        let (x1, y1) = (a[0] + dx * t1, a[1] + dy * t1);
        let steps = (x1 - x0).abs().max((y1 - y0).abs()).max(1.0) as i32;
        for step in 0..=steps {
            let u = step as f32 / steps as f32;
            let t = t0 + (t1 - t0) * u;
            let inv = inv_a + (inv_b - inv_a) * t;
            self.put(
                (x0 + (x1 - x0) * u) as i32,
                (y0 + (y1 - y0) * u) as i32,
                colour,
                inv,
            );
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

/// Blinn-ish lighting in `0.3..1`, never above 1 so a body does not clip white.
///
/// A key light plus a cool fill from the camera's far side: one light alone
/// crushes a textured model's shadow side to black, which is what made an
/// imported avatar read as a silhouette rather than as a lit surface.
fn shading(normal: [f32; 3], camera: &Camera, light: [f32; 3]) -> f32 {
    let facing = -dot(normal, camera.forward).abs();
    let key = dot(normal, normalise(light)).abs();
    let fill = dot(normal, normalise([-0.65, 0.15, -0.75])).max(0.0);
    let rim = (1.0 - facing).powf(2.5) * 0.16;
    (0.30 + 0.52 * key + 0.16 * fill + rim).min(1.0)
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

    fn painted(canvas: &Canvas) -> usize {
        // Every drawn pixel differs from the gradient only by chance, so count
        // the depth buffer instead: `put` always writes depth.
        canvas.depth.iter().filter(|value| **value > 0.0).count()
    }

    /// The camera stands inside the floor's footprint and looks slightly down,
    /// which is the normal viewing position: some floor lines run from behind
    /// the eye to in front of it. Walk each 3D line independently and demand a
    /// painted pixel wherever it should land on screen — a line that crosses the
    /// near plane used to be dropped whole, which left a *band* of missing grid
    /// lines across the floor wherever the near plane cut the ground.
    #[test]
    fn lines_crossing_the_near_plane_are_drawn() {
        let (width, height) = (642usize, 434usize);
        let colour = [200u8, 210, 220];
        let mut checked = 0usize;
        let mut clipped_lines = 0usize;
        for pose in 0..16 {
            let yaw = 0.28 + pose as f32 * 0.041;
            let pitch = -0.35 + (pose % 9) as f32 * 0.04;
            let distance = [3.6f32, 5.6, 9.0, 14.0][pose % 4];
            let camera = Camera::orbit(
                [0.0, 0.9, 0.0],
                yaw,
                pitch,
                distance,
                [642.0, 434.0],
            );
            let mut canvas = Canvas::new(width, height);
            // A fan of floor lines: some run under the camera (and are cut by the
            // near plane), the rest are ordinary grid lines further out.
            let eye = camera.eye;
            let mut lines = Vec::new();
            for offset in [-6.0f32, -4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5, 6.0] {
                lines.push((
                    [eye[0] - 9.0, 0.0, eye[2] + offset],
                    [eye[0] + 9.0, 0.0, eye[2] + offset],
                ));
                lines.push((
                    [eye[0] + offset, 0.0, eye[2] - 9.0],
                    [eye[0] + offset, 0.0, eye[2] + 9.0],
                ));
            }
            for (from, to) in &lines {
                canvas.line(&camera, *from, *to, colour);
                let a = to_view(&camera, *from);
                let b = to_view(&camera, *to);
                if a[2] < camera.near || b[2] < camera.near {
                    clipped_lines += 1;
                }
                for sample in 0..=900 {
                    let t = sample as f32 / 900.0;
                    let point = [
                        from[0] + (to[0] - from[0]) * t,
                        from[1] + (to[1] - from[1]) * t,
                        from[2] + (to[2] - from[2]) * t,
                    ];
                    let v = [
                        dot(sub(point, camera.eye), camera.right),
                        dot(sub(point, camera.eye), camera.up),
                        dot(sub(point, camera.eye), camera.forward),
                    ];
                    // Stay clear of the clip tolerance at the near plane.
                    if v[2] < camera.near * 4.0 {
                        continue;
                    }
                    let screen = [
                        camera.centre[0] + camera.focal * v[0] / v[2],
                        camera.centre[1] - camera.focal * v[1] / v[2],
                    ];
                    let (x, y) = (screen[0] as i32, screen[1] as i32);
                    if x < 1 || y < 1 || x >= width as i32 - 1 || y >= height as i32 - 1 {
                        continue;
                    }
                    checked += 1;
                    // The sampler rounds to a pixel, so accept the 3×3 around the
                    // exact position; a dropped line has nothing anywhere near it.
                    let mut hit = false;
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let at = (y + dy) as usize * width + (x + dx) as usize;
                            let pixel = canvas.colour[at];
                            if pixel[0] == colour[0]
                                && pixel[1] == colour[1]
                                && pixel[2] == colour[2]
                            {
                                hit = true;
                            }
                        }
                    }
                    assert!(
                        hit,
                        "no line pixel near ({x}, {y}) at yaw {yaw} pitch {pitch} d {distance}"
                    );
                }
            }
        }
        assert!(clipped_lines > 10, "no line crossed the near plane at all");
        assert!(checked > 5000, "the test only checked {checked} points");
    }

    /// A floor quad with one corner behind the eye used to vanish whole, which
    /// looked like the ground plane disappearing while the camera moved.
    #[test]
    fn floor_plane_survives_a_corner_behind_the_camera() {
        let size = [480.0, 320.0];
        for step in 0..60 {
            let yaw = step as f32 * 0.11;
            let pitch = 0.05 + (step % 11) as f32 * 0.06;
            let camera = Camera::orbit([0.0, 1.0, 0.0], yaw, pitch, 1.2, size);
            let mut canvas = Canvas::new(480, 320);
            let half = 8.0;
            let quad = [
                [-half, 0.0, -half],
                [half, 0.0, -half],
                [half, 0.0, half],
                [-half, 0.0, half],
            ];
            canvas.triangle_flat(&camera, [quad[0], quad[1], quad[2]], [30, 32, 36], false);
            canvas.triangle_flat(&camera, [quad[0], quad[2], quad[3]], [30, 32, 36], false);
            assert!(
                painted(&canvas) > 0,
                "the floor plane vanished at yaw {yaw} pitch {pitch}"
            );
        }
    }

    /// The near-plane clip must not turn a partly visible line into a walk over
    /// millions of pixels: the sampled range stays inside the canvas.
    #[test]
    fn a_line_is_sampled_only_inside_the_canvas() {
        let size = [320.0, 240.0];
        let camera = Camera::orbit([0.0, 1.0, 0.0], 0.0, 0.0, 0.5, size);
        let mut canvas = Canvas::new(320, 240);
        // Runs from just behind the eye to far behind it sideways.
        let from = [camera.eye[0] - 0.001, camera.eye[1], camera.eye[2] - 0.001];
        let to = [camera.eye[0] + 500.0, camera.eye[1], camera.eye[2] - 0.05];
        let started = std::time::Instant::now();
        canvas.line(&camera, from, to, [255, 0, 0]);
        assert!(
            started.elapsed().as_millis() < 100,
            "a single line took {:?}",
            started.elapsed()
        );
    }

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
