//! The 3D viewport: shaded MHR meshes over a floor grid, drawn by the software
//! rasteriser in [`crate::render`].
//!
//! The Qt application gives this panel a QOpenGLWidget with shaded SMPL-class
//! bodies, a ground grid, a bone overlay, a motion path and an axis gizmo. This
//! keeps the same content and interaction, with a CPU rasteriser instead of GL:
//! one texture per frame, plus egui text for the labels.

use std::sync::Arc;

use eframe::egui;

use crate::i18n;
use crate::render::{self, Canvas, Camera};

/// Orbit camera, in metres and radians.
#[derive(Clone, Debug)]
pub struct Orbit {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: [f32; 3],
    /// Set once the view has been fitted to the current subject.
    pub framed: bool,
}

impl Default for Orbit {
    fn default() -> Self {
        Self {
            // The source camera sits at the world origin and the bodies face +z,
            // so a small positive yaw puts the eye in front of the actor at a
            // three-quarter angle (was behind them).
            yaw: 0.28,
            pitch: 0.12,
            distance: 3.6,
            target: [0.0, 0.0, 0.0],
            framed: false,
        }
    }
}

/// The standard viewing directions of the original combo box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewMode {
    Perspective,
    Front,
    Back,
    Right,
    Left,
    Top,
    Bottom,
}

impl ViewMode {
    pub const LABELS: [&'static str; 7] = [
        "Perspective",
        "Front",
        "Back",
        "Right",
        "Left",
        "Top",
        "Bottom",
    ];

    pub fn from_index(index: usize) -> Self {
        match index {
            1 => ViewMode::Front,
            2 => ViewMode::Back,
            3 => ViewMode::Right,
            4 => ViewMode::Left,
            5 => ViewMode::Top,
            6 => ViewMode::Bottom,
            _ => ViewMode::Perspective,
        }
    }

    /// Fixed yaw/pitch for an axis-aligned view.
    fn angles(self) -> Option<(f32, f32)> {
        match self {
            ViewMode::Perspective => None,
            ViewMode::Front => Some((std::f32::consts::PI, 0.0)),
            ViewMode::Back => Some((0.0, 0.0)),
            ViewMode::Right => Some((std::f32::consts::FRAC_PI_2, 0.0)),
            ViewMode::Left => Some((-std::f32::consts::FRAC_PI_2, 0.0)),
            ViewMode::Top => Some((0.0, 1.4)),
            ViewMode::Bottom => Some((0.0, -1.4)),
        }
    }
}

/// Which display layers are on. Names follow the Qt footer.
#[derive(Clone, Debug)]
pub struct Layers {
    pub bones: bool,
    pub grid: bool,
    pub motion_path: bool,
    pub names: bool,
    pub follow: bool,
    /// Show tracked people over the source frame in the Movie Clip panel.
    pub overlays: bool,
}

impl Default for Layers {
    fn default() -> Self {
        Self {
            bones: false,
            grid: true,
            motion_path: true,
            names: true,
            follow: true,
            overlays: true,
        }
    }
}

/// Colours the original gives its people list, in order.
pub const PERSON_COLOURS: [egui::Color32; 6] = [
    egui::Color32::from_rgb(96, 178, 255),
    egui::Color32::from_rgb(255, 170, 96),
    egui::Color32::from_rgb(140, 220, 160),
    egui::Color32::from_rgb(230, 140, 220),
    egui::Color32::from_rgb(235, 220, 120),
    egui::Color32::from_rgb(150, 205, 225),
];

/// Sky-to-floor gradient of the Qt viewport.
const SKY_TOP: [u8; 3] = [38, 40, 44];
const SKY_BOTTOM: [u8; 3] = [24, 25, 28];
const GRID_MINOR: [u8; 3] = [52, 54, 58];
const GRID_MAJOR: [u8; 3] = [68, 71, 76];
const AXIS_X: [u8; 3] = [150, 74, 74];
const AXIS_Z: [u8; 3] = [74, 96, 150];
const BONE: [u8; 3] = [235, 240, 246];
const TRAIL: [u8; 3] = [120, 170, 235];
const HINT: egui::Color32 = egui::Color32::from_rgb(150, 150, 150);

/// Cached texture and canvas, so a repaint only re-rasterises.
#[derive(Default)]
pub struct ViewportCache {
    pub texture: Option<egui::TextureHandle>,
    pub canvas: Option<Canvas>,
}

/// One person to draw for the frame on screen.
pub struct PersonDraw<'a> {
    pub name: &'a str,
    pub colour: egui::Color32,
    pub verts: &'a [f32],
    pub keypoints: &'a [f32],
    /// Triangles for these vertices. Each body carries its own: the MHR mesh
    /// and the FlowHMR SMPL-H body have different topologies.
    pub faces: &'a [[u32; 3]],
    /// Texture coordinates, the model's images and which one each face uses.
    pub uv: Option<&'a [[f32; 2]]>,
    pub textures: &'a [Arc<crate::character::Texture>],
    pub face_texture: &'a [u16],
    pub selected: bool,
}

/// The scene to draw for one frame.
pub struct Scene<'a> {
    pub people: &'a [PersonDraw<'a>],
    /// Root path of the selected person over the clip.
    pub trail: &'a [[f32; 3]],
    pub layers: &'a Layers,
    pub view: ViewMode,
    /// Ground height in world metres.
    pub floor: f32,
    /// Placeholder shown when there is nothing to draw.
    pub empty_hint: (&'static str, &'static str),
}

/// Bones of the 70-keypoint rig, as index pairs into `PersonDraw::keypoints`.
const BONES: [(usize, usize); 26] = [
    (69, 0),   // neck -> nose
    (0, 1),    // nose -> left eye
    (0, 2),    // nose -> right eye
    (1, 3),    // left eye -> left ear
    (2, 4),    // right eye -> right ear
    (69, 5),   // neck -> left shoulder
    (69, 6),   // neck -> right shoulder
    (5, 7),    // left shoulder -> left elbow
    (6, 8),    // right shoulder -> right elbow
    (7, 62),   // left elbow -> left wrist
    (8, 41),   // right elbow -> right wrist
    (5, 9),    // left shoulder -> left hip
    (6, 10),   // right shoulder -> right hip
    (9, 10),   // hips
    (5, 6),    // shoulders
    (9, 11),   // left hip -> left knee
    (10, 12),  // right hip -> right knee
    (11, 13),  // left knee -> left ankle
    (12, 14),  // right knee -> right ankle
    (13, 15),  // left ankle -> left big toe
    (13, 17),  // left ankle -> left heel
    (14, 18),  // right ankle -> right big toe
    (14, 20),  // right ankle -> right heel
    (67, 5),   // left acromion -> left shoulder
    (68, 6),   // right acromion -> right shoulder
    (0, 69),   // nose -> neck (head axis)
];

/// Draw the viewport into `rect`, handling orbit/zoom/frame-all.
pub fn show(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    scene: &Scene<'_>,
    orbit: &mut Orbit,
    cache: &mut ViewportCache,
) -> bool {
    let response = ui.interact(
        rect,
        ui.id().with("motion-viewport"),
        egui::Sense::click_and_drag(),
    );

    if let Some((yaw, pitch)) = scene.view.angles() {
        orbit.yaw = yaw;
        orbit.pitch = pitch;
        orbit.framed = true;
    }

    // Left drag orbits, middle/right drag pans, wheel zooms, double click
    // reframes - the four gestures the footer advertises.
    if response.dragged_by(egui::PointerButton::Primary) {
        let delta = response.drag_delta();
        // Convention (measured, not guessed): dragging left moves the picture to
        // the right, dragging right moves it left, and the pivot stays put. The
        // Qt viewport (`mocap/viewport.py`) uses the same `-=` sign.
        orbit.yaw -= delta.x * 0.008;
        orbit.pitch = (orbit.pitch + delta.y * 0.008).clamp(-1.45, 1.45);
        orbit.framed = true;
    }
    if response.dragged_by(egui::PointerButton::Middle)
        || response.dragged_by(egui::PointerButton::Secondary)
    {
        let delta = response.drag_delta();
        let (sin_yaw, cos_yaw) = orbit.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = orbit.pitch.sin_cos();
        let forward = [
            -cos_pitch * sin_yaw,
            -sin_pitch,
            -cos_pitch * cos_yaw,
        ];
        let right = [cos_yaw, 0.0, -sin_yaw];
        let up = render::cross(right, forward);
        let scale = orbit.distance * 0.0016;
        for axis in 0..3 {
            orbit.target[axis] +=
                (-right[axis] * delta.x + up[axis] * delta.y) * scale;
        }
        orbit.framed = true;
    }
    if response.hovered() {
        let scroll = ui.input(|i| i.raw_scroll_delta.y);
        if scroll.abs() > 0.01 {
            orbit.distance = (orbit.distance * (-scroll * 0.0022).exp()).clamp(0.3, 120.0);
            orbit.framed = true;
        }
    }
    let mut changed = false;
    if response.double_clicked() {
        orbit.framed = false;
        changed = true;
    }

    // The pivot is taken once, when the clip is framed, and then left alone:
    // re-centring every frame made the camera chase the root and the whole view
    // juddered while the animation played.
    if !orbit.framed {
        // Only latch once a body actually exists: the first frames of a capture
        // have no geometry, and latching then left the camera at the origin
        // looking at nothing for the rest of the session.
        if let Some((min, max)) = scene_bounds(scene) {
            orbit.target = [
                (min[0] + max[0]) * 0.5,
                (min[1] + max[1]) * 0.5,
                (min[2] + max[2]) * 0.5,
            ];
            let radius = render::length(render::sub(max, min)).max(1.0);
            orbit.distance = (radius * 1.25).clamp(0.8, 60.0);
            orbit.framed = true;
        }
    }

    // Rasterise at device pixels, capped so four bodies stay interactive.
    let pixels_per_point = ui.ctx().pixels_per_point();
    let width = ((rect.width() * pixels_per_point).round() as usize).clamp(160, 1100);
    let height = ((rect.height() * pixels_per_point).round() as usize).clamp(120, 800);
    let size = [width as f32, height as f32];
    let camera = Camera::orbit(orbit.target, orbit.yaw, orbit.pitch, orbit.distance, size);

    let canvas = cache.canvas.get_or_insert_with(|| Canvas::new(width, height));
    canvas.resize(width, height);
    canvas.clear_gradient(SKY_TOP, SKY_BOTTOM);

    let grid = if scene.layers.grid {
        Some(draw_grid(canvas, &camera, orbit.target, scene.floor))
    } else {
        None
    };
    if scene.layers.motion_path && scene.trail.len() > 1 {
        // A trail, not a roadmap: the newest segment is at full strength and the
        // older ones fade out, so the path is drawn behind the walk as the take
        // plays instead of appearing whole from the first frame.
        let count = scene.trail.len();
        for (index, pair) in scene.trail.windows(2).enumerate() {
            let age = 1.0 - (index + 1) as f32 / count as f32;
            let alpha = (1.0 - age * age) * 0.85;
            canvas.line_faded(&camera, pair[0], pair[1], TRAIL, alpha);
        }
    }

    let light = [0.35f32, 0.85, 0.4];

    for person in scene.people {
        if person.verts.len() < 9 {
            continue;
        }
        // A light clay body tinted by the person colour: the Qt render is a
        // shaded figure, not a saturated fill.
        let tint = [
            mix_channel(person.colour.r(), 214, 0.62),
            mix_channel(person.colour.g(), 220, 0.62),
            mix_channel(person.colour.b(), 226, 0.62),
        ];
        // Vertex normals make the body read as a smooth surface instead of a
        // faceted shell, which is what the Qt renderer shows.
        let normals = vertex_normals(person.verts, person.faces);
        // Both sides are rasterised: the MHR surface is not perfectly
        // consistently wound, so a back-face test drops real geometry. The
        // z-buffer already resolves what is visible, and 147k triangles for
        // four bodies cost ~3 ms.
        for (face, triangle) in person.faces.iter().enumerate() {
            let (a, b, c) = (
                vertex(person.verts, triangle[0]),
                vertex(person.verts, triangle[1]),
                vertex(person.verts, triangle[2]),
            );
            let n = [
                normals[triangle[0] as usize],
                normals[triangle[1] as usize],
                normals[triangle[2] as usize],
            ];
            // An imported character brings its own images and coordinates; the
            // built-in bodies are clay, tinted by the person's colour. A face
            // with no material texture falls back to the clay path.
            let slot = person.face_texture.get(face).copied().unwrap_or(u16::MAX);
            let image = if slot == u16::MAX {
                None
            } else {
                person.textures.get(slot as usize)
            };
            match (person.uv, image) {
                (Some(uv), Some(texture)) if uv.len() >= person.verts.len() / 3 =>
                {
                    let coordinates = [
                        uv[triangle[0] as usize],
                        uv[triangle[1] as usize],
                        uv[triangle[2] as usize],
                    ];
                    canvas.triangle_textured(&camera, [a, b, c], n, coordinates, texture, light);
                }
                _ => canvas.triangle_gouraud(&camera, [a, b, c], n, tint, light),
            }
        }
        if scene.layers.bones && person.selected {
            draw_bones(canvas, &camera, person.keypoints);
        }
    }

    let image = canvas.to_image();
    match &mut cache.texture {
        Some(texture) => texture.set(image, egui::TextureOptions::LINEAR),
        None => {
            cache.texture = Some(ui.ctx().load_texture(
                "viewport",
                image,
                egui::TextureOptions::LINEAR,
            ));
        }
    }
    if let Some(texture) = &cache.texture {
        let painter = ui.painter_at(rect);
        painter.image(
            texture.id(),
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
    }

    painted_labels(ui, rect, scene, orbit, &camera, size, grid);

    changed || response.dragged() || response.double_clicked()
}

fn vertex(verts: &[f32], index: u32) -> [f32; 3] {
    let at = index as usize * 3;
    [verts[at], verts[at + 1], verts[at + 2]]
}

/// Area-weighted vertex normals for one body.
fn vertex_normals(verts: &[f32], faces: &[[u32; 3]]) -> Vec<[f32; 3]> {
    let count = verts.len() / 3;
    let mut normals = vec![[0.0f32; 3]; count];
    for triangle in faces {
        let (a, b, c) = (
            vertex(verts, triangle[0]),
            vertex(verts, triangle[1]),
            vertex(verts, triangle[2]),
        );
        let normal = render::cross(render::sub(b, a), render::sub(c, a));
        for index in triangle {
            let slot = &mut normals[*index as usize];
            slot[0] += normal[0];
            slot[1] += normal[1];
            slot[2] += normal[2];
        }
    }
    for normal in &mut normals {
        *normal = render::normalise(*normal);
    }
    normals
}

fn mix_channel(from: u8, to: u8, t: f32) -> u8 {
    (from as f32 + (to as f32 - from as f32) * t).clamp(0.0, 255.0) as u8
}

/// Spacing and half-extent of the floor grid for one camera pose, in metres.
#[derive(Clone, Copy)]
struct GridGeometry {
    /// Distance between two lines.
    step: f32,
    /// Half the width of the patch that is drawn.
    half: f32,
}

impl GridGeometry {
    /// Every second line is the heavier one: 1 m over a 0.5 m grid, like the Qt
    /// viewport.
    fn major(self) -> f32 {
        self.step * 2.0
    }

    fn extent(self) -> f32 {
        self.half * 2.0
    }
}

/// Pick a spacing and an extent that keep the floor readable at any zoom.
///
/// The old grid was 16 m across with a fixed 0.5 m step, so pulling back packed
/// the lines below one pixel and the floor turned into a moiré that appeared to
/// gain and lose lines as the camera moved, while pushing in wasted work on
/// hundreds of lines the view could not resolve.
fn grid_geometry(camera: &Camera, distance: f32) -> GridGeometry {
    const MIN_CELL_PIXELS: f32 = 12.0;
    let wanted = camera.metres_per_pixel(distance) * MIN_CELL_PIXELS;
    let mut step = 0.5f32;
    while step < wanted && step < 512.0 {
        step *= 2.0;
    }
    let mut half = (distance * 1.6).max(8.0);
    let mut count = (half / step).round().max(1.0);
    // Bound the work per frame: about a hundred lines each way is already more
    // than a viewport can resolve.
    while count > 100.0 && step < 512.0 {
        step *= 2.0;
        count = (half / step).round().max(1.0);
    }
    half = count * step;
    GridGeometry { step, half }
}

/// Whether a world coordinate lands on a heavy line.
fn on_major(value: f32, major: f32) -> bool {
    let nearest = (value / major).round();
    (value - nearest * major).abs() < major * 1e-3
}

fn draw_grid(canvas: &mut Canvas, camera: &Camera, target: [f32; 3], floor: f32) -> GridGeometry {
    let distance = render::length(render::sub(target, camera.eye));
    let geometry = grid_geometry(camera, distance);
    let (step, half) = (geometry.step, geometry.half);
    // Anchor the patch to whole steps of the world grid, so every line stays on
    // a fixed world coordinate: panning moves the edge of the patch, never the
    // position or the shade of a line.
    let centre_x = (target[0] / step).round() * step;
    let centre_z = (target[2] / step).round() * step;
    // A slightly lighter plane under the lines, so the ground reads as a surface
    // instead of loose stripes.
    let plane = [
        [centre_x - half, floor, centre_z - half],
        [centre_x + half, floor, centre_z - half],
        [centre_x + half, floor, centre_z + half],
        [centre_x - half, floor, centre_z + half],
    ];
    canvas.triangle_flat(camera, [plane[0], plane[1], plane[2]], [30, 32, 36], false);
    canvas.triangle_flat(camera, [plane[0], plane[2], plane[3]], [30, 32, 36], false);
    // The lines sit exactly on the plane they just painted, so drop the depth
    // they wrote; the bodies that follow still occlude themselves correctly.
    canvas.begin_overlay();

    let major = geometry.major();
    let count = (half / step).round() as i32;
    for index in -count..=count {
        let x = centre_x + index as f32 * step;
        let z = centre_z + index as f32 * step;
        // Shade from the world coordinate rather than the loop index, so a heavy
        // line keeps its colour when the patch re-centres.
        let along_z = if on_major(x, major) { GRID_MAJOR } else { GRID_MINOR };
        let along_x = if on_major(z, major) { GRID_MAJOR } else { GRID_MINOR };
        canvas.line(
            camera,
            [x, floor, centre_z - half],
            [x, floor, centre_z + half],
            along_z,
        );
        canvas.line(
            camera,
            [centre_x - half, floor, z],
            [centre_x + half, floor, z],
            along_x,
        );
    }
    // The axes mark the *world* origin. Drawing them where the patch re-centred
    // made them wander across the floor while panning, and because they were
    // depth-tested against the grid line they sit on, the two fought over the
    // same pixels and the red/blue lines blinked in and out.
    if (0.0 - centre_z).abs() <= half {
        canvas.line_overlay(
            camera,
            [centre_x - half, floor, 0.0],
            [centre_x + half, floor, 0.0],
            AXIS_X,
        );
    }
    if (0.0 - centre_x).abs() <= half {
        canvas.line_overlay(
            camera,
            [0.0, floor, centre_z - half],
            [0.0, floor, centre_z + half],
            AXIS_Z,
        );
    }
    geometry
}

fn draw_bones(canvas: &mut Canvas, camera: &Camera, keypoints: &[f32]) {
    if keypoints.len() < 70 * 3 {
        return;
    }
    let point = |index: usize| -> [f32; 3] {
        [
            keypoints[index * 3],
            keypoints[index * 3 + 1],
            keypoints[index * 3 + 2],
        ]
    };
    for (from, to) in BONES {
        canvas.line_overlay(camera, point(from), point(to), BONE);
    }
}

fn painted_labels(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    scene: &Scene<'_>,
    orbit: &Orbit,
    _camera: &Camera,
    _size: [f32; 2],
    grid: Option<GridGeometry>,
) {
    let painter = ui.painter_at(rect);
    painter.text(
        rect.left_top() + egui::vec2(10.0, 8.0),
        egui::Align2::LEFT_TOP,
        i18n::t(match scene.view {
            ViewMode::Perspective => "User Perspective",
            ViewMode::Front => "Front",
            ViewMode::Back => "Back",
            ViewMode::Right => "Right",
            ViewMode::Left => "Left",
            ViewMode::Top => "Top",
            ViewMode::Bottom => "Bottom",
        }),
        egui::FontId::proportional(12.0),
        egui::Color32::from_gray(205),
    );

    if scene.people.is_empty() {
        painter.text(
            rect.center() - egui::vec2(0.0, 10.0),
            egui::Align2::CENTER_CENTER,
            scene.empty_hint.0,
            egui::FontId::proportional(14.0),
            egui::Color32::from_gray(190),
        );
        painter.text(
            rect.center() + egui::vec2(0.0, 16.0),
            egui::Align2::CENTER_CENTER,
            scene.empty_hint.1,
            egui::FontId::proportional(12.0),
            HINT,
        );
    } else if scene.layers.names {
        for person in scene.people {
            let anchor = person
                .keypoints
                .chunks_exact(3)
                .take(1)
                .next()
                .map(|p| [p[0], p[1] + 0.28, p[2]]);
            if let Some(anchor) = anchor {
                let camera_ok = scene.people.len() <= 8;
                if camera_ok {
                    if let Some(screen) = project_person_name(rect, orbit, anchor) {
                        painter.text(
                            screen,
                            egui::Align2::CENTER_BOTTOM,
                            person.name,
                            egui::FontId::proportional(11.0),
                            person.colour,
                        );
                    }
                }
            }
        }
    }

    painter.text(
        rect.left_bottom() + egui::vec2(10.0, -10.0),
        egui::Align2::LEFT_BOTTOM,
        i18n::t("LMB: Orbit | MMB/RMB: Pan | Wheel: Zoom | Double-click: Frame"),
        egui::FontId::proportional(11.0),
        HINT,
    );
    let grid_info = match grid {
        Some(grid) => format!("{:.0} m | {} / {} m", grid.extent(), grid.step, grid.major()),
        None => i18n::zh_en("关闭", "off"),
    };
    painter.text(
        rect.right_bottom() + egui::vec2(-10.0, -10.0),
        egui::Align2::RIGHT_BOTTOM,
        match i18n::lang() {
            i18n::Lang::Zh => format!(
                "地面 Y {:+.2} m | {} | Y 向上   ·   {:+.1} m",
                scene.floor, grid_info, orbit.distance
            ),
            i18n::Lang::En => format!(
                "Grid Y {:+.2} m | {} | Y Up   ·   {:+.1} m",
                scene.floor, grid_info, orbit.distance
            ),
        },
        egui::FontId::proportional(11.0),
        HINT,
    );

    // Axis gizmo, top right.
    let centre = egui::pos2(rect.right() - 62.0, rect.top() + 54.0);
    let radius = 30.0f32;
    painter.circle_filled(
        centre,
        radius + 6.0,
        egui::Color32::from_rgba_unmultiplied(28, 29, 31, 180),
    );
    let (sin_yaw, cos_yaw) = orbit.yaw.sin_cos();
    let (sin_pitch, cos_pitch) = orbit.pitch.sin_cos();
    let forward = [
        -cos_pitch * sin_yaw,
        -sin_pitch,
        -cos_pitch * cos_yaw,
    ];
    // Same basis as `Camera::orbit`. Building the gizmo from `cross(up, forward)`
    // instead pointed the whole triad the other way, so the widget turned against
    // the scene: Y showed downwards and X pointed left.
    let right = render::normalise(render::cross(forward, [0.0, 1.0, 0.0]));
    let up = render::cross(right, forward);
    for (axis, colour, label) in [
        ([1.0, 0.0, 0.0], egui::Color32::from_rgb(214, 92, 92), "X"),
        ([0.0, 1.0, 0.0], egui::Color32::from_rgb(122, 200, 122), "Y"),
        ([0.0, 0.0, 1.0], egui::Color32::from_rgb(104, 148, 230), "Z"),
    ] {
        let x = render::dot(axis, right);
        let y = render::dot(axis, up);
        let tip = centre + egui::vec2(x, -y) * radius;
        painter.line_segment([centre, tip], egui::Stroke::new(1.4_f32, colour));
        painter.circle_filled(tip, 7.0, colour);
        painter.text(
            tip + egui::vec2(0.0, -0.5),
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::proportional(9.0),
            egui::Color32::from_rgb(20, 20, 20),
        );
    }
}

/// Where to put a person's name: the same orbit camera, in UI points.
fn project_person_name(
    rect: egui::Rect,
    orbit: &Orbit,
    point: [f32; 3],
) -> Option<egui::Pos2> {
    let size = [rect.width(), rect.height()];
    let camera = Camera::orbit(orbit.target, orbit.yaw, orbit.pitch, orbit.distance, size);
    let (screen, _) = camera.project(point)?;
    Some(egui::pos2(rect.left() + screen[0], rect.top() + screen[1]))
}

fn scene_bounds(scene: &Scene<'_>) -> Option<([f32; 3], [f32; 3])> {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut any = false;
    // Frame the selected actor. Fitting every person at once makes the actor
    // tiny whenever a distant spectator was also reconstructed.
    let selected = scene.people.iter().filter(|person| person.selected).count() > 0;
    for person in scene.people.iter().filter(|person| !selected || person.selected) {
        for point in person.verts.chunks_exact(3) {
            if !point.iter().all(|value| value.is_finite()) {
                continue;
            }
            any = true;
            for axis in 0..3 {
                min[axis] = min[axis].min(point[axis]);
                max[axis] = max[axis].max(point[axis]);
            }
        }
    }
    any.then_some((min, max))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The grid must stay legible at every zoom: cells wide enough to resolve,
    /// lines spaced on a power-of-two multiple of the reference 0.5 m, and a
    /// patch that reaches past the pivot. A fixed step over a fixed 16 m patch
    /// is what made distant lines crowd into a moiré that seemed to gain and
    /// lose lines as the camera moved.
    #[test]
    fn grid_spacing_stays_readable_at_every_zoom() {
        for distance in [0.3f32, 1.0, 3.6, 10.0, 30.0, 80.0, 120.0] {
            for size in [[480.0f32, 320.0], [1100.0, 800.0]] {
                let camera = Camera::orbit([0.0, 0.9, 0.0], 0.28, 0.12, distance, size);
                let grid = grid_geometry(&camera, distance);
                let cell = grid.step / camera.metres_per_pixel(distance);
                assert!(
                    cell >= 10.0,
                    "a {:.2} m cell is only {cell:.1} px wide at {distance} m",
                    grid.step
                );
                let multiple = grid.step / 0.5;
                assert!(
                    (multiple.log2().fract()).abs() < 1e-4,
                    "step {} is not a power-of-two multiple of 0.5 m",
                    grid.step
                );
                let lines = grid.half / grid.step;
                assert!(lines <= 100.0 + 0.5, "{lines} lines each way at {distance} m");
                assert!(
                    grid.half + 1e-3 >= (distance * 1.6).max(8.0),
                    "the patch stops at {:.1} m when the camera is {distance} m out",
                    grid.half
                );
            }
        }
    }

    /// Zooming a little must not re-space the grid: the step is a function of
    /// the distance, but it may only change when a cell would otherwise become
    /// too small to read.
    #[test]
    fn grid_does_not_respace_over_a_small_zoom() {
        let size = [1100.0f32, 800.0];
        let camera = Camera::orbit([0.0, 0.9, 0.0], 0.28, 0.12, 3.6, size);
        let baseline = grid_geometry(&camera, 3.6).step;
        for step in 0..100 {
            let distance = 3.0 + step as f32 * 0.02;
            let camera = Camera::orbit([0.0, 0.9, 0.0], 0.28, 0.12, distance, size);
            assert_eq!(
                grid_geometry(&camera, distance).step,
                baseline,
                "the grid re-spaced at {distance} m"
            );
        }
    }
}

#[cfg(test)]
mod visual {
    use super::*;

    /// Visual harness: rasterise the floor grid at a given pose into a raw RGB
    /// file, so the result can be looked at without opening the GUI.
    ///
    ///     VELOMOCAP_GRID_DUMP=out.rgb cargo test -p velomocap-app dump_grid_frame -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dump_grid_frame() {
        let (width, height) = (642usize, 434usize);
        let distance: f32 = std::env::var("VELOMOCAP_GRID_DISTANCE")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(5.6);
        let camera = Camera::orbit(
            [0.0, 0.9, 0.0],
            0.28,
            0.12,
            distance,
            [width as f32, height as f32],
        );
        let mut canvas = Canvas::new(width, height);
        canvas.clear_gradient(SKY_TOP, SKY_BOTTOM);
        let grid = draw_grid(&mut canvas, &camera, [0.0, 0.9, 0.0], 0.0);
        eprintln!(
            "grid: step {} m, extent {:.0} m at {distance} m",
            grid.step,
            grid.extent()
        );
        let mut out = Vec::with_capacity(width * height * 3);
        for pixel in &canvas.colour {
            out.extend_from_slice(&pixel[0..3]);
        }
        let path = std::env::var("VELOMOCAP_GRID_DUMP")
            .unwrap_or_else(|_| std::env::temp_dir().join("grid_frame.rgb").display().to_string());
        std::fs::write(&path, &out).expect("writing the raw frame");
        eprintln!("grid frame -> {path}");
    }
}

#[cfg(test)]
mod orbit_tests {
    use crate::render::Camera;

    /// The orbit convention, pinned numerically: a left drag turns the subject's
    /// front to the left (the camera orbits to the right around the pivot),
    /// which is what `mocap/viewport.py` does with `yaw -= delta.x`. The sign is
    /// one character and a screenshot cannot show which way the scene turned.
    #[test]
    fn dragging_left_turns_the_subject_left() {
        let size = [642.0f32, 434.0];
        let target = [0.0, 0.9, 0.0];
        // A point on the front of the subject, which is what a user watches.
        let marker = [0.0, 0.9, 1.0];
        let base_yaw = 0.28f32;
        let screen_x = |yaw: f32| {
            Camera::orbit(target, yaw, 0.12, 5.6, size)
                .project(marker)
                .expect("the marker stays in front of the camera")
                .0[0]
        };
        // `show` applies `yaw -= delta.x * 0.008`.
        let base = screen_x(base_yaw);
        let left = screen_x(base_yaw - (-40.0 * 0.008));
        let right = screen_x(base_yaw - (40.0 * 0.008));
        assert!(
            left < base,
            "a left drag must turn the subject left ({base:.1} -> {left:.1})"
        );
        assert!(
            right > base,
            "a right drag must turn the subject right ({base:.1} -> {right:.1})"
        );
    }
}

#[cfg(test)]
mod orbit_dump {
    use super::*;

    /// Render the view before and after a 40 px drag to the left, so the orbit
    /// direction can be looked at instead of argued about.
    ///
    ///     VELOMOCAP_ORBIT_DUMP=dir cargo test -p velomocap-app dump_orbit_pair -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dump_orbit_pair() {
        let (width, height) = (642usize, 434usize);
        let target = [0.0, 0.9, 0.0];
        let dir = std::env::var("VELOMOCAP_ORBIT_DUMP")
            .unwrap_or_else(|_| std::env::temp_dir().display().to_string());
        for (name, drag) in [("before", 0.0f32), ("after", -40.0f32)] {
            // `show` applies `yaw -= delta.x * 0.008`.
            let yaw = 0.28 - drag * 0.008;
            let camera = Camera::orbit(target, yaw, 0.12, 5.6, [width as f32, height as f32]);
            let mut canvas = Canvas::new(width, height);
            canvas.clear_gradient(SKY_TOP, SKY_BOTTOM);
            draw_grid(&mut canvas, &camera, target, 0.0);
            // Two upright posts so a horizontal shift is obvious at a glance.
            for (x, colour) in [(2.0f32, [220, 90, 90]), (-2.0f32, [90, 160, 230])] {
                for step in -2..=2 {
                    let offset = step as f32 * 0.02;
                    canvas.line(
                        &camera,
                        [x + offset, 0.0, 0.0],
                        [x + offset, 2.0, 0.0],
                        colour,
                    );
                }
            }
            let mut out = Vec::with_capacity(width * height * 3);
            for pixel in &canvas.colour {
                out.extend_from_slice(&pixel[0..3]);
            }
            let path = format!("{dir}/orbit_{name}.rgb");
            std::fs::write(&path, &out).expect("writing the raw frame");
            eprintln!("{name}: yaw {yaw:.3} -> {path}");
        }
    }
}

