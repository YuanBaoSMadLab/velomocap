//! The 3D viewport: shaded MHR meshes over a floor grid, drawn by the software
//! rasteriser in [`crate::render`].
//!
//! The Qt application gives this panel a QOpenGLWidget with shaded SMPL-class
//! bodies, a ground grid, a bone overlay, a motion path and an axis gizmo. This
//! keeps the same content and interaction, with a CPU rasteriser instead of GL:
//! one texture per frame, plus egui text for the labels.

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
        // The scene follows the cursor: drag right and the subject turns to the
        // right, which is what "grab and turn it" means to a user.
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

    if scene.layers.grid {
        draw_grid(canvas, &camera, orbit.target, scene.floor);
    }
    if scene.layers.motion_path && scene.trail.len() > 1 {
        for pair in scene.trail.windows(2) {
            canvas.line(&camera, pair[0], pair[1], TRAIL);
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
        for triangle in person.faces {
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
            canvas.triangle_gouraud(&camera, [a, b, c], n, tint, light);
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

    painted_labels(ui, rect, scene, orbit, &camera, size);

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

fn draw_grid(canvas: &mut Canvas, camera: &Camera, target: [f32; 3], floor: f32) {
    let half = 8.0f32;
    let step = 0.5f32;
    let centre_x = (target[0] / 1.0).round();
    let centre_z = (target[2] / 1.0).round();
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

    let count = (half / step) as i32;
    for index in -count..=count {
        let offset = index as f32 * step;
        let major = (offset % 1.0).abs() < 1e-3;
        let colour = if major { GRID_MAJOR } else { GRID_MINOR };
        canvas.line(
            camera,
            [centre_x + offset, floor, centre_z - half],
            [centre_x + offset, floor, centre_z + half],
            colour,
        );
        canvas.line(
            camera,
            [centre_x - half, floor, centre_z + offset],
            [centre_x + half, floor, centre_z + offset],
            colour,
        );
    }
    canvas.line(
        camera,
        [centre_x - half, floor, centre_z],
        [centre_x + half, floor, centre_z],
        AXIS_X,
    );
    canvas.line(
        camera,
        [centre_x, floor, centre_z - half],
        [centre_x, floor, centre_z + half],
        AXIS_Z,
    );
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
    painter.text(
        rect.right_bottom() + egui::vec2(-10.0, -10.0),
        egui::Align2::RIGHT_BOTTOM,
        match i18n::lang() {
            i18n::Lang::Zh => format!(
                "地面 Y {:+.2} m | 20 m | 0.5 / 1 m | Y 向上   ·   {:+.1} m",
                scene.floor, orbit.distance
            ),
            i18n::Lang::En => format!(
                "Grid Y {:+.2} m | 20 m | 0.5 / 1 m | Y Up   ·   {:+.1} m",
                scene.floor, orbit.distance
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
    let right = render::normalise(render::cross([0.0, 1.0, 0.0], forward));
    let up = render::cross(forward, right);
    for (axis, colour, label) in [
        ([1.0, 0.0, 0.0], egui::Color32::from_rgb(214, 92, 92), "X"),
        ([0.0, 1.0, 0.0], egui::Color32::from_rgb(122, 200, 122), "Y"),
        ([0.0, 0.0, 1.0], egui::Color32::from_rgb(104, 148, 230), "Z"),
    ] {
        let x = render::dot(axis, right);
        let y = render::dot(axis, up);
        let tip = centre + egui::vec2(x, -y) * radius;
        painter.line_segment([centre, tip], egui::Stroke::new(1.4, colour));
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
