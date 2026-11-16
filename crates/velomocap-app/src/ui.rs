//! Widgets and layout for the VeloMocap window.
//!
//! This mirrors the original RGB MoCap Qt workspace: the same panel order, the
//! same headers and footer strips, the same names and the same dark palette
//! (`mocap/theme.py`), so the two applications read as one product.

use eframe::egui;

use crate::app::{LockDrag, LockGrab, VeloMocap};
use crate::i18n;
use crate::viewport::{self, Scene, ViewMode};

// Palette: light, quiet, and close to the other tools on this desktop — white
// Palette: the neutral dark grey the other tools on this desktop use — a near
// black canvas, panels a shade above it, one hairline between them, quiet grey
// text and one blue accent. The panel order is still the Qt build's, so nothing
// that works is rearranged; only the skin is its own.
const WINDOW_BG: egui::Color32 = egui::Color32::from_rgb(0x17, 0x17, 0x18);
const PANEL_BG: egui::Color32 = egui::Color32::from_rgb(0x1e, 0x1e, 0x20);
const HEADER_BG: egui::Color32 = egui::Color32::from_rgb(0x24, 0x24, 0x26);
const BORDER: egui::Color32 = egui::Color32::from_rgb(0x31, 0x31, 0x34);
const MENU_BG: egui::Color32 = egui::Color32::from_rgb(0x1b, 0x1b, 0x1d);
const TEXT: egui::Color32 = egui::Color32::from_rgb(0xe4, 0xe4, 0xe6);
const MUTED: egui::Color32 = egui::Color32::from_rgb(0x9a, 0x9a, 0x9f);
const ACCENT: egui::Color32 = egui::Color32::from_rgb(0x4c, 0x8d, 0xff);
const ACCENT_DIM: egui::Color32 = egui::Color32::from_rgb(0x2f, 0x62, 0xc4);
const ERR: egui::Color32 = egui::Color32::from_rgb(0xf2, 0x6d, 0x6a);
const OK: egui::Color32 = egui::Color32::from_rgb(0x5f, 0xc9, 0x92);
/// Cards and list rows: a shade above the panel, the way the other tools lay a
/// card on a page.
const CARD_BG: egui::Color32 = egui::Color32::from_rgb(0x26, 0x26, 0x28);
/// The row that is selected: the neutral highlight, not the accent.
const SELECT_BG: egui::Color32 = egui::Color32::from_rgb(0x35, 0x35, 0x38);

/// Where a missing model comes from, and where the project lives.
/// Where a missing SMPL-H model comes from. This is the official Max Planck
/// download page (registration required), not a mirror: the body model is not
/// redistributed with the app.
const MODEL_DOWNLOAD_URL: &str = "https://download.is.tue.mpg.de/";
const REPOSITORY_URL: &str = "https://github.com/YuanBaoSMadLab/velomocap";
const SPONSOR_URL: &str = "https://afdian.com/a/YuanBaoSMadLab";

const HEADER_HEIGHT: f32 = 26.0;
const FOOTER_HEIGHT: f32 = 24.0;
const STATUS_HEIGHT: f32 = 24.0;
const TIMELINE_HEIGHT: f32 = 232.0;
const SIDEBAR_WIDTH: f32 = 276.0;
const CLIP_PANEL_WIDTH: f32 = 460.0;

pub fn configure(ctx: &egui::Context) {
    install_cjk_font(ctx);
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = PANEL_BG;
    visuals.window_fill = PANEL_BG;
    visuals.extreme_bg_color = egui::Color32::from_rgb(0x13, 0x13, 0x15);
    visuals.faint_bg_color = egui::Color32::from_rgb(0x28, 0x28, 0x2b);
    visuals.override_text_color = Some(TEXT);
    visuals.selection.bg_fill = SELECT_BG;
    visuals.selection.stroke = egui::Stroke::new(1.0_f32, ACCENT);
    visuals.window_stroke = egui::Stroke::new(1.0_f32, BORDER);
    visuals.widgets.noninteractive.bg_fill = PANEL_BG;
    visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0_f32, BORDER);
    visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(0x2a, 0x2a, 0x2d);
    visuals.widgets.inactive.weak_bg_fill = egui::Color32::from_rgb(0x26, 0x26, 0x29);
    visuals.widgets.inactive.bg_stroke =
        egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(0x36, 0x36, 0x3a));
    visuals.widgets.inactive.corner_radius = 6.into();
    visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(0x34, 0x34, 0x38);
    visuals.widgets.hovered.weak_bg_fill = egui::Color32::from_rgb(0x30, 0x30, 0x34);
    visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(0x48, 0x48, 0x4d));
    visuals.widgets.hovered.corner_radius = 6.into();
    visuals.widgets.active.bg_fill = egui::Color32::from_rgb(0x3a, 0x3a, 0x3f);
    visuals.widgets.active.weak_bg_fill = SELECT_BG;
    visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0_f32, ACCENT_DIM);
    visuals.widgets.active.corner_radius = 6.into();
    visuals.widgets.open.bg_fill = egui::Color32::from_rgb(0x2a, 0x2a, 0x2d);
    visuals.window_corner_radius = 10.into();
    visuals.menu_corner_radius = 8.into();
    visuals.popup_shadow = egui::epaint::Shadow {
        offset: [0, 6],
        blur: 16,
        spread: 0,
        color: egui::Color32::from_black_alpha(140),
    };
    visuals.window_shadow = egui::epaint::Shadow {
        offset: [0, 10],
        blur: 28,
        spread: 0,
        color: egui::Color32::from_black_alpha(170),
    };
    ctx.set_visuals(visuals);

    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(7.0, 6.0);
    style.spacing.button_padding = egui::vec2(10.0, 4.0);
    style.spacing.interact_size.y = 21.0;
    style.spacing.slider_width = 110.0;
    style.spacing.combo_width = 110.0;
    style.spacing.indent = 12.0;
    use egui::{FontFamily, FontId, TextStyle};
    style.text_styles = [
        (TextStyle::Small, FontId::new(10.5, FontFamily::Proportional)),
        (TextStyle::Body, FontId::new(12.0, FontFamily::Proportional)),
        (TextStyle::Button, FontId::new(12.0, FontFamily::Proportional)),
        (TextStyle::Heading, FontId::new(13.5, FontFamily::Proportional)),
        (TextStyle::Monospace, FontId::new(11.0, FontFamily::Monospace)),
    ]
    .into();
    ctx.set_style(style);
}

/// egui's bundled fonts have no CJK coverage, so a Chinese label renders as
/// nothing at all. Pull a system CJK face in, preferring the UI fonts Windows
/// ships, and fall back to English-only text if none is present.
fn install_cjk_font(ctx: &egui::Context) {
    let candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhl.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\Deng.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ];
    let Some((path, bytes)) = candidates
        .iter()
        .find_map(|path| std::fs::read(path).ok().map(|bytes| (*path, bytes)))
    else {
        return;
    };
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "cjk".to_owned(),
        std::sync::Arc::new(egui::FontData::from_owned(bytes)),
    );
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts.families.entry(family).or_default().push("cjk".to_owned());
    }
    ctx.set_fonts(fonts);
    let _ = path;
}

/// The whole window, in the order the Qt build lays it out.
pub fn draw(app: &mut VeloMocap, ctx: &egui::Context) {
    menu_bar(app, ctx);
    // The status bar spans the window; the timeline sits inside the workspace,
    // between the clip column and the sidebar, exactly as in Qt.
    status_bar(app, ctx);
    movie_clip(app, ctx);
    sidebar(app, ctx);
    timeline(app, ctx);
    viewport_panel(app, ctx);
    console_window(app, ctx);
    about_window(app, ctx);
    guide_window(app, ctx);
    settings_window(app, ctx);
    about_app_window(app, ctx);
    export_window(app, ctx);
    person_window(app, ctx);
    clear_scene_prompt(app, ctx);
    drop_overlay(ctx);
}

/// While files hover over the window, say what releasing them would do. Without
/// it a drop is a guess: the same gesture means "this is the clip", "this is a
/// character" and "this is the project".
fn drop_overlay(ctx: &egui::Context) {
    let Some(hint) = VeloMocap::drop_hint(ctx) else {
        return;
    };
    let screen = ctx.screen_rect();
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Tooltip,
        egui::Id::new("drop-overlay"),
    ));
    painter.rect_filled(screen, 0.0, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 140));
    egui::Area::new(egui::Id::new("drop-card"))
        .order(egui::Order::Tooltip)
        .fixed_pos(screen.center() - egui::vec2(150.0, 40.0))
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(PANEL_BG)
                .stroke(egui::Stroke::new(1.0, ACCENT_DIM))
                .inner_margin(egui::Margin::symmetric(14, 10))
                .show(ui, |ui| {
                    ui.set_max_width(320.0);
                    ui.label(
                        egui::RichText::new(i18n::t("Drop to import"))
                            .size(13.0)
                            .color(egui::Color32::WHITE),
                    );
                    for line in hint.lines() {
                        ui.label(egui::RichText::new(line).size(11.0).color(MUTED));
                    }
                });
        });
}

/// Ask what to do with the scene that is already on screen before an import.
fn clear_scene_prompt(app: &mut VeloMocap, ctx: &egui::Context) {
    if !app.ask_clear {
        return;
    }
    let people = app.people.len();
    let clips = app.flow.len();
    // `Some(true)` clears, `Some(false)` keeps, `None` cancels.
    let mut choice: Option<Option<bool>> = None;
    egui::Window::new(i18n::t("Import video"))
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ctx, |ui| {
            ui.label(
                egui::RichText::new(i18n::zh_en(
                    &format!("当前场景有 {people} 人、{clips} 段动作。导入新视频后是否清空？"),
                    &format!(
                        "The scene has {people} people and {clips} motion clips. Clear it when importing?"
                    ),
                ))
                .size(12.0),
            );
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button(i18n::zh_en("清空并导入", "Clear and import")).clicked() {
                    choice = Some(Some(true));
                }
                if ui.button(i18n::zh_en("保留并导入", "Keep and import")).clicked() {
                    choice = Some(Some(false));
                }
                if ui.button(i18n::t("Cancel")).clicked() {
                    choice = Some(None);
                }
            });
        });
    if let Some(choice) = choice {
        app.ask_clear = false;
        let pending = app.pending_import.take();
        match (pending, choice) {
            (Some(path), Some(clear)) => app.finish_import(path, clear),
            _ => {}
        }
    }
}

fn menu_bar(app: &mut VeloMocap, ctx: &egui::Context) {
    egui::TopBottomPanel::top("menubar")
        .exact_height(22.0)
        .frame(
            egui::Frame::new()
                .fill(MENU_BG)
                .inner_margin(egui::Margin::symmetric(6, 2)),
        )
        .show(ctx, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button(i18n::t("File"), |ui| {
                    // The workspace goes first: a project is the document this
                    // app edits, and everything else in this menu acts on it.
                    if ui
                        .add(
                            egui::Button::new(i18n::t("Open project…"))
                                .shortcut_text("Ctrl+O"),
                        )
                        .on_hover_text(i18n::t(
                            "Load a .vp project: the video, the people, the motion, the avatars and the locks",
                        ))
                        .clicked()
                    {
                        ui.close();
                        app.open_project_dialog();
                    }
                    if ui
                        .add(
                            egui::Button::new(i18n::t("Save project…"))
                                .shortcut_text("Ctrl+S"),
                        )
                        .on_hover_text(i18n::t(
                            "Write the whole workspace to a .vp file, so reopening it does not run the reconstruction again",
                        ))
                        .clicked()
                    {
                        ui.close();
                        app.save_project_dialog();
                    }
                    ui.separator();
                    if ui.button(i18n::t("Import video…")).clicked() {
                        ui.close();
                        app.open_video_dialog();
                    }
                    let can_run = app.video.is_some();
                    let selected = app.candidates.iter().filter(|c| c.selected).count();
                    let label = if selected > 0 {
                        i18n::zh_en(
                            &format!("重建选中人物（{selected}）"),
                            &format!("Reconstruct {selected} selected"),
                        )
                    } else {
                        i18n::t("Reconstruct all people").to_string()
                    };
                    if ui
                        .add_enabled(
                            can_run,
                            egui::Button::new(label),
                        )
                        .on_hover_text(i18n::t("Detect every person, then run SAM 3D Body on each"))
                        .clicked()
                    {
                        ui.close();
                        app.start_capture();
                    }
                    if ui
                        .add_enabled(
                            app.capture.as_ref().map(|j| j.is_running()).unwrap_or(false),
                            egui::Button::new(i18n::t("Cancel reconstruction")),
                        )
                        .clicked()
                    {
                        ui.close();
                        app.cancel_capture();
                    }
                    ui.separator();
                    // The reference's "Import Character": a VRM/glTF humanoid
                    // replaces the built-in body and is driven by the clip.
                    if ui
                        .button(i18n::t("Import character…"))
                        .on_hover_text(i18n::t(
                            "Import a VRM / glTF humanoid and play the motion on it",
                        ))
                        .clicked()
                    {
                        ui.close();
                        app.import_character_dialog(Some(app.selected));
                    }
                    ui.separator();
                    // Everything a written file needs — its format, the parts to
                    // include, the names and the folder — lives in one export
                    // window, like a DCC's export dialog. It opens even with
                    // nothing to export: the window says what is missing, which
                    // a greyed-out menu item cannot.
                    if ui
                        .button(i18n::t("Export character assets…"))
                        .on_hover_text(i18n::t(
                            "Mesh, skeleton, materials and animation as one .glb",
                        ))
                        .clicked()
                    {
                        ui.close();
                        app.show_export = true;
                    }
                    ui.separator();
                    // Preferences are a property of the application, not of the
                    // panel that happened to hold them.
                    if ui
                        .add(
                            egui::Button::new(i18n::t("Settings…"))
                                .shortcut_text("Ctrl+,"),
                        )
                        .clicked()
                    {
                        ui.close();
                        app.show_settings = true;
                    }
                    ui.separator();
                    if ui.button(i18n::t("Quit")).clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                ui.menu_button(i18n::t("Help"), |ui| {
                    ui.menu_button(i18n::t("Language / 语言"), |ui| {
                        let mut language = i18n::lang();
                        if ui
                            .selectable_value(&mut language, i18n::Lang::Zh, "中文")
                            .clicked()
                        {
                            i18n::set_lang(i18n::Lang::Zh);
                            ctx.send_viewport_cmd(egui::ViewportCommand::Title(window_title()));
                            ui.close();
                        }
                        if ui
                            .selectable_value(&mut language, i18n::Lang::En, "English")
                            .clicked()
                        {
                            i18n::set_lang(i18n::Lang::En);
                            ctx.send_viewport_cmd(egui::ViewportCommand::Title(window_title()));
                            ui.close();
                        }
                    });
                    ui.separator();
                    if ui.button(i18n::t("Workspace guide")).clicked() {
                        ui.close();
                        app.show_guide = true;
                    }
                    if ui.button(i18n::t("Keyboard shortcuts")).clicked() {
                        ui.close();
                        app.show_help = true;
                    }
                    if ui.button(i18n::t("Console")).clicked() {
                        ui.close();
                        app.show_console = true;
                    }
                    ui.separator();
                    if ui
                        .button(i18n::zh_en("关于 VeloMocap", "About VeloMocap"))
                        .clicked()
                    {
                        ui.close();
                        app.show_about = true;
                    }
                });
            });
        });
}


/// A panel: raised above the gutters, one hairline border, rounded corners.
fn panel(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::new()
        .fill(PANEL_BG)
        .stroke(egui::Stroke::new(1.0_f32, BORDER))
        .corner_radius(6.0)
        .inner_margin(egui::Margin::same(6))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            // Fill the column instead of shrinking to the content: a panel that
            // stops halfway leaves a slab of empty gutter under it.
            ui.spacing_mut().item_spacing = egui::vec2(7.0, 5.0);
            add(ui);
            let spare = ui.available_height();
            if spare.is_finite() && spare > 0.0 {
                ui.allocate_space(egui::vec2(ui.available_width(), spare));
            }
        });
}

/// The strip that titles every panel.
fn header(ui: &mut egui::Ui, title: &str, right: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::new()
        .fill(HEADER_BG)
        .corner_radius(4.0)
        .inner_margin(egui::Margin::symmetric(9, 4))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            // `horizontal_centered` would stretch the strip to the panel height.
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(title).size(11.5).strong().color(TEXT));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), right);
            });
        });
}

/// The matching footer strip (viewport layers, action bars).
fn footer(ui: &mut egui::Ui, left: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::new()
        .fill(HEADER_BG)
        .corner_radius(4.0)
        .inner_margin(egui::Margin::symmetric(9, 3))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(left);
        });
}

fn movie_clip(app: &mut VeloMocap, ctx: &egui::Context) {
    egui::SidePanel::left("movie-clip")
        // Rubber-banded like the reference's workspace splitter: the source pane
        // and the person list are the parts a user resizes most.
        .resizable(true)
        .default_width(CLIP_PANEL_WIDTH)
        .min_width(300.0)
        .max_width(760.0)
        .frame(egui::Frame::new().fill(WINDOW_BG).inner_margin(egui::Margin::same(6)))
        .show(ctx, |ui| {
            let available = ui.available_size();
            ui.allocate_ui(available, |ui| {
                panel(ui, |ui| {
                    header(ui, i18n::t("Movie Clip"), |ui| {
                        let fps = if app.video.is_some() {
                            format!("{:.2} fps", app.current_fps())
                        } else {
                            i18n::t("No clip loaded").to_string()
                        };
                        ui.checkbox(
                            &mut app.layers.overlays,
                            egui::RichText::new(i18n::t("Overlays")).size(11.0),
                        )
                        .on_hover_text(i18n::t("Show tracked people over the source frame"));
                        ui.label(egui::RichText::new(fps).size(11.0).color(MUTED));
                    });
                    let body = ui.available_rect_before_wrap();
                    if body.height() > 40.0 {
                        let split = body.left() + 168.0;
                        let people = egui::Rect::from_min_max(
                            body.min,
                            egui::pos2(split, body.bottom()),
                        );
                        let video = egui::Rect::from_min_max(
                            egui::pos2(split + 14.0, body.min.y),
                            body.max,
                        );
                        people_column(app, ui, people);
                        video_view(app, ui, video);
                        ui.advance_cursor_after_rect(body);
                    }
                });
            });
        });
}

fn people_column(app: &mut VeloMocap, ui: &mut egui::Ui, rect: egui::Rect) {
    let mut ui = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(rect.shrink2(egui::vec2(6.0, 4.0)))
            .layout(egui::Layout::top_down(egui::Align::Min)),
    );
    let ui = &mut ui;
    // People | Characters | Stream. These used to be two labels with the second
    // one disabled, which read as a broken tab. All three are pages now, and the
    // character page is where imported models live so one can be switched onto a
    // person without finding the file again.
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        // The three tabs share the column's width. A fixed 54 px each ran past
        // the edge of a narrow column and the third one was clipped, which read
        // as a missing page.
        let spacing = ui.spacing().item_spacing.x;
        let width = ((ui.available_width() - spacing * 2.0) / 3.0).max(38.0);
        for (index, label) in [i18n::t("People"), i18n::t("Characters"), i18n::t("Stream")]
            .iter()
            .enumerate()
        {
            let selected = app.left_tab == index;
            let text = egui::RichText::new(*label).size(11.5).color(if selected {
                egui::Color32::WHITE
            } else {
                TEXT
            });
            let button = egui::Button::new(text)
                .fill(if selected { ACCENT } else { HEADER_BG })
                .stroke(egui::Stroke::new(
                    1.0_f32,
                    if selected { ACCENT_DIM } else { BORDER },
                ))
                .corner_radius(4.0)
                .min_size(egui::vec2(width, 21.0));
            if ui.add(button).clicked() {
                app.left_tab = index;
            }
        }
    });
    ui.add_space(4.0);
    match app.left_tab {
        1 => characters_column(app, ui),
        2 => stream_column(app, ui),
        _ => people_list(app, ui),
    }
}

/// The imported characters, one card each. Clicking a card opens its details on
/// the right, where it can be put on the selected person.
fn characters_column(app: &mut VeloMocap, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(i18n::t("Characters")).size(11.5));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .add(egui::Button::new("?").min_size(egui::vec2(22.0, 18.0)))
                .on_hover_text(i18n::t(
                    "Import a VRM / glTF / FBX / PMX humanoid and it joins this list",
                ))
                .clicked()
            {
                app.show_guide = true;
            }
        });
    });
    let cards: Vec<(usize, String, String, usize, usize)> = app
        .characters
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            (
                index,
                entry
                    .path
                    .file_stem()
                    .map(|stem| stem.to_string_lossy().into_owned())
                    .unwrap_or_else(|| entry.path.display().to_string()),
                entry
                    .path
                    .extension()
                    .map(|extension| extension.to_string_lossy().to_uppercase())
                    .unwrap_or_default(),
                entry.character.matched(),
                entry.character.bones.len(),
            )
        })
        .collect();
    let mut remove: Option<usize> = None;
    let mut select: Option<usize> = None;
    // The list of cards, then the import button: the scroll area is given the
    // height that is left *after* the button, otherwise a long list pushes the
    // button under the status bar.
    let list_height = (ui.available_height() - 36.0).max(80.0);
    egui::Frame::new()
        .fill(CARD_BG)
        .stroke(egui::Stroke::new(1.0_f32, BORDER))
        .inner_margin(egui::Margin::same(3))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            egui::ScrollArea::vertical()
                .max_height(list_height)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if cards.is_empty() {
                        ui.add_space(6.0);
                        ui.label(
                            egui::RichText::new(i18n::t("No characters imported yet"))
                                .size(11.0)
                                .color(MUTED),
                        );
                    }
                    // Two shelves: what ships with the app, and what the user
                    // brought in. Built-ins come from this install's folder, so
                    // the same avatar arriving from another path is the same
                    // entry, and it is listed once.
                    for (builtin, title) in [
                        (true, i18n::zh_en("内置模型", "Built in")),
                        (false, i18n::zh_en("导入模型", "Imported")),
                    ] {
                        let group: Vec<_> = cards
                            .iter()
                            .filter(|(index, ..)| app.characters[*index].builtin == builtin)
                            .collect();
                        if group.is_empty() {
                            continue;
                        }
                        ui.add_space(3.0);
                        ui.label(
                            egui::RichText::new(title)
                                .size(10.0)
                                .strong()
                                .color(MUTED),
                        );
                        for (index, name, format, matched, bones) in group {
                            // PMX is experimental: the skeleton and 付与 work, but the
                            // rigid-body physics that moves hair and cloth in MMD is
                            // not simulated here yet.
                            let tag = if format == "PMX" {
                                format!("{} · ", i18n::t("Experimental"))
                            } else {
                                String::new()
                            };
                            let selected = app.character_panel == Some(*index);
                            let worn_by = app
                                .people
                                .iter()
                                .find(|person| {
                                    person
                                        .character
                                        .as_ref()
                                        .map(|assigned| assigned.path == app.characters[*index].path)
                                        .unwrap_or(false)
                                })
                                .map(|person| person.name.clone());
                            let caption = match &worn_by {
                                Some(person) => {
                                    format!("{}{format} · {matched}/{bones} · {person}", tag)
                                }
                                None => format!("{}{format} · {matched}/{bones}", tag),
                            };
                            let row = egui::Frame::new()
                                .fill(if selected {
                                    SELECT_BG
                                } else {
                                    egui::Color32::TRANSPARENT
                                })
                                .stroke(egui::Stroke::new(
                                    1.0_f32,
                                    if selected {
                                        ACCENT_DIM
                                    } else {
                                        egui::Color32::TRANSPARENT
                                    },
                                ))
                                .inner_margin(egui::Margin::symmetric(4, 3))
                                .show(ui, |ui| {
                                    ui.set_width(ui.available_width());
                                    ui.horizontal(|ui| {
                                        ui.vertical(|ui| {
                                            ui.label(
                                                egui::RichText::new(name).size(11.5).color(
                                                    if selected {
                                                        egui::Color32::WHITE
                                                    } else {
                                                        TEXT
                                                    },
                                                ),
                                            );
                                            ui.label(
                                                egui::RichText::new(caption).size(10.0).color(MUTED),
                                            );
                                        });
                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                if ui
                                                    .add(
                                                        egui::Button::new("×")
                                                            .min_size(egui::vec2(16.0, 16.0)),
                                                    )
                                                    .on_hover_text(i18n::t("Remove from list"))
                                                    .clicked()
                                                {
                                                    remove = Some(*index);
                                                }
                                            },
                                        );
                                    });
                                })
                                .response;
                            let row = row.interact(egui::Sense::click());
                            if row.clicked() {
                                select = Some(*index);
                            }
                            row.on_hover_text(i18n::t(
                                "Click for this model's details, then put it on the selected person",
                            ));
                        }
                    }
                });
        });
    if let Some(index) = remove {
        app.remove_character(index);
    } else if let Some(index) = select {
        app.character_panel = Some(index);
    }
    ui.add_space(4.0);
    if ui
        .add(
            egui::Button::new(i18n::t("Import character…"))
                .min_size(egui::vec2(ui.available_width(), 22.0)),
        )
        .clicked()
    {
        let person = if app.people.is_empty() {
            None
        } else {
            Some(app.selected.min(app.people.len() - 1))
        };
        app.import_character_dialog(person);
    }
}

/// The stream page: what the loaded clip actually is.
fn stream_column(app: &mut VeloMocap, ui: &mut egui::Ui) {
    ui.label(egui::RichText::new(i18n::t("Stream")).size(11.5));
    ui.add_space(4.0);
    let name = app
        .video
        .as_ref()
        .and_then(|path| path.file_name())
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| i18n::t("No clip loaded").to_string());
    egui::Frame::new()
        .fill(CARD_BG)
        .stroke(egui::Stroke::new(1.0_f32, BORDER))
        .inner_margin(egui::Margin::symmetric(6, 5))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(egui::RichText::new(name).size(11.0).color(TEXT));
            for (label, value) in [
                (
                    i18n::t("Resolution"),
                    format!("{} × {}", app.video_size.0, app.video_size.1),
                ),
                (
                    i18n::t("Frame rate"),
                    format!("{:.2} fps", app.current_fps()),
                ),
                (i18n::t("Frames"), format!("{}", app.frame_count())),
                (
                    i18n::t("Position"),
                    format!("{} / {}", app.playhead + 1, app.frame_count().max(1)),
                ),
            ] {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(label).size(10.5).color(MUTED));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(egui::RichText::new(value).size(10.5).color(TEXT));
                    });
                });
            }
        });
}

fn people_list(app: &mut VeloMocap, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(i18n::t("People")).size(11.5));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .add(egui::Button::new("?").min_size(egui::vec2(22.0, 18.0)))
                .on_hover_text(i18n::t("How to choose, rename and display people"))
                .clicked()
            {
                app.show_guide = true;
            }
        });
    });

    let people = app
        .people
        .iter()
        .map(|person| (person.id, person.name.clone(), person.hits(), person.colour))
        .collect::<Vec<_>>();

    // The list gets what is left after the button row below it. Measuring this
    // *inside* the frame reads the column's whole height, which made the frame
    // tall enough to push "Find person" out through the bottom of the panel,
    // half-hidden behind the status bar.
    let list_height = (ui.available_height() - 34.0).max(60.0);
    egui::Frame::new()
        .fill(CARD_BG)
        .stroke(egui::Stroke::new(1.0_f32, BORDER))
        .inner_margin(egui::Margin::same(3))
        .show(ui, |ui| {
            ui.set_min_height(list_height);
            ui.set_width(ui.available_width());
            egui::ScrollArea::vertical()
                .max_height(list_height)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if people.is_empty() {
                        if app.detecting {
                            ui.add_space(6.0);
                            ui.label(
                                egui::RichText::new(i18n::t("Detecting people…"))
                                    .size(11.0)
                                    .color(MUTED),
                            );
                            return;
                        }
                        if !app.candidates.is_empty() {
                            let mut build = false;
                            let mut pick: Option<usize> = None;
                            for index in 0..app.candidates.len() {
                                let colour = crate::viewport::PERSON_COLOURS
                                    [index % crate::viewport::PERSON_COLOURS.len()];
                                let selected = app.selected == index;
                                let mut checked = app.candidates[index].selected;
                                let width = app.candidates[index].bbox[2]
                                    - app.candidates[index].bbox[0];
                                let height = app.candidates[index].bbox[3]
                                    - app.candidates[index].bbox[1];
                                let score = app.candidates[index].score;
                                ui.horizontal(|ui| {
                                    if ui
                                        .add(egui::Checkbox::without_text(&mut checked))
                                        .on_hover_text(i18n::t(
                                            "Reconstruct this person when the run starts",
                                        ))
                                        .changed()
                                    {
                                        app.candidates[index].selected = checked;
                                    }
                                    let (rect, _) = ui.allocate_exact_size(
                                        egui::vec2(10.0, 10.0),
                                        egui::Sense::hover(),
                                    );
                                    ui.painter().circle_filled(rect.center(), 4.0, colour);
                                    let label = egui::RichText::new(format!(
                                        "{} {}   {width:.0}×{height:.0} · {score:.2}",
                                        i18n::t("Person"),
                                        index + 1
                                    ))
                                    .size(11.0)
                                    .color(if selected { egui::Color32::WHITE } else { TEXT });
                                    if ui
                                        .add(egui::Label::new(label).sense(egui::Sense::click()))
                                        .on_hover_text(i18n::t(
                                            "Select this person: the box over the clip is drawn in this colour",
                                        ))
                                        .clicked()
                                    {
                                        pick = Some(index);
                                    }
                                });
                            }
                            if let Some(index) = pick {
                                app.selected = index;
                            }
                            let selected =
                                app.candidates.iter().filter(|c| c.selected).count();
                            ui.add_space(4.0);
                            if ui
                                .add_enabled(
                                    selected > 0,
                                    egui::Button::new(i18n::zh_en(
                                        &format!("重建选中人物（{selected}）"),
                                        &format!("Reconstruct {selected} selected"),
                                    )),
                                )
                                .clicked()
                            {
                                build = true;
                            }
                            if build {
                                app.start_capture();
                            }
                            return;
                        }
                        ui.add_space(6.0);
                        ui.label(
                            egui::RichText::new(i18n::t("No people yet.\nReconstruct a clip to find them."))
                                .size(11.0)
                                .color(MUTED),
                        );
                        return;
                    }
                    for (index, (id, name, frames, colour)) in people.iter().enumerate() {
                        let mut checked = app
                            .people
                            .get(index)
                            .map(|person| person.visible)
                            .unwrap_or(true);
                        let selected = app.selected == index;
                        ui.horizontal(|ui| {
                            if ui
                                .add(egui::Checkbox::without_text(&mut checked))
                                .on_hover_text(i18n::t("Show this person in the 3D viewport"))
                                .changed()
                            {
                                if let Some(person) = app.people.get_mut(index) {
                                    person.visible = checked;
                                }
                            }
                            let (rect, _) = ui.allocate_exact_size(
                                egui::vec2(10.0, 10.0),
                                egui::Sense::hover(),
                            );
                            ui.painter().circle_filled(rect.center(), 4.0, *colour);
                            let label = egui::RichText::new(i18n::person_row(name, *frames))
                                .size(11.0)
                                .color(if selected { egui::Color32::WHITE } else { TEXT });
                            if ui
                                .add(egui::Label::new(label).sense(egui::Sense::click()))
                                .clicked()
                            {
                                app.selected = index;
                            }
                            let _ = id;
                        });
                    }
                });
        });

    ui.add_space(4.0);
    ui.horizontal(|ui| {
        if ui
            .add_enabled(app.video.is_some(), egui::Button::new(i18n::t("Find person")))
            .on_hover_text(i18n::t("Reconstruct every person in the clip"))
            .clicked()
        {
            app.start_capture();
        }
        // Not a control: this build has one take per video. The reference's
        // take selector lists several clips of a scene, and rendering an option
        // list with one dead entry only invites the question of why it does
        // nothing.
        ui.label(egui::RichText::new(i18n::t("Default")).size(11.0))
            .on_hover_text(i18n::t(
                "One take per video in this build; the reference's multi-take scene is not implemented",
            ));
    });
}

fn video_view(app: &mut VeloMocap, ui: &mut egui::Ui, rect: egui::Rect) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(0x11, 0x11, 0x13));

    let Some(texture) = &app.preview else {
        painter.text(
            rect.center() - egui::vec2(0.0, 10.0),
            egui::Align2::CENTER_CENTER,
            i18n::t("Open a video to begin"),
            egui::FontId::proportional(13.0),
            egui::Color32::from_gray(205),
        );
        painter.text(
            rect.center() + egui::vec2(0.0, 12.0),
            egui::Align2::CENTER_CENTER,
            "MP4 · MOV · AVI",
            egui::FontId::proportional(11.0),
            MUTED,
        );
        return;
    };

    let size = texture.size_vec2();
    let scale = (rect.width() / size.x).min(rect.height() / size.y);
    let drawn = size * scale;
    let origin = rect.center() - drawn * 0.5;
    let image_rect = egui::Rect::from_min_size(origin, drawn);
    painter.image(
        texture.id(),
        image_rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );

    // Boxes over the source frame, one colour per person, with the selected one
    // drawn heavier. Before a reconstruction exists these come from the
    // detection pass; afterwards from the tracked boxes.
    if app.layers.overlays {
        if !app.people.is_empty() {
            let frame = app.playhead.min(app.preview_frame);
            // Capture decodes at `work_width`, which may be narrower than the
            // source, so the boxes are in *that* pixel space, not the video's.
            let source = app.video_size.0.min(app.scale_width).max(1) as f32;
            let source_h = app.video_size.1.max(1) as f32 * source / app.video_size.0.max(1) as f32;
            let sx = drawn.x / source;
            let sy = drawn.y / source_h;
            for (index, person) in app.people.iter().enumerate() {
                if !person.visible {
                    continue;
                }
                // Only this frame's box: falling back to the nearest stored
                // frame drew people who had already left the shot.
                let Some(frame) = person.frame(frame) else {
                    continue;
                };
                let bbox = frame.bbox;
                let min = origin + egui::vec2(bbox[0] * sx, bbox[1] * sy);
                let max = origin + egui::vec2(bbox[2] * sx, bbox[3] * sy);
                let selected = index == app.selected;
                painter.rect_stroke(
                    egui::Rect::from_min_max(min, max),
                    2.0,
                    egui::Stroke::new(if selected { 2.6_f32 } else { 1.4 }, person.colour),
                    egui::StrokeKind::Inside,
                );
                painter.text(
                    min + egui::vec2(3.0, -3.0),
                    egui::Align2::LEFT_BOTTOM,
                    &person.name,
                    egui::FontId::proportional(if selected { 12.0 } else { 10.0 }),
                    person.colour,
                );
            }
        } else if !app.candidates.is_empty() && app.playhead == 0 {
            for (index, candidate) in app.candidates.iter().enumerate() {
                let colour = crate::viewport::PERSON_COLOURS[index % crate::viewport::PERSON_COLOURS.len()];
                let min = origin
                    + egui::vec2(candidate.norm[0] * drawn.x, candidate.norm[1] * drawn.y);
                let max = origin
                    + egui::vec2(candidate.norm[2] * drawn.x, candidate.norm[3] * drawn.y);
                let selected = index == app.selected;
                painter.rect_stroke(
                    egui::Rect::from_min_max(min, max),
                    2.0,
                    egui::Stroke::new(if selected { 2.6_f32 } else { 1.4 }, colour),
                    egui::StrokeKind::Inside,
                );
                painter.text(
                    min + egui::vec2(3.0, -3.0),
                    egui::Align2::LEFT_BOTTOM,
                    format!("{}", index + 1),
                    egui::FontId::proportional(if selected { 12.0 } else { 10.0 }),
                    colour,
                );
            }
        }
    }
}

fn viewport_panel(app: &mut VeloMocap, ctx: &egui::Context) {    egui::CentralPanel::default()
        .frame(egui::Frame::new().fill(WINDOW_BG).inner_margin(egui::Margin::same(6)))
        .show(ctx, |ui| {
            panel(ui, |ui| {
                header(ui, i18n::t("3D Viewport"), |ui| {
                    // Right-to-left: Frame All, view combo, Bone Overlay.
                    if ui
                        .add(egui::Button::new(i18n::t("Frame All")).min_size(egui::vec2(64.0, 18.0)))
                        .on_hover_text(i18n::t("Fit all visible people in the 3D view"))
                        .clicked()
                    {
                        app.orbit.framed = false;
                    }
                    egui::ComboBox::from_id_salt("view-mode")
                        .selected_text(i18n::t(mode_label(ViewMode::from_index(app.view_mode))))
                        .width(104.0)
                        .show_ui(ui, |ui| {
                            for (index, label) in ViewMode::LABELS.iter().enumerate() {
                                if ui
                                    .selectable_label(
                                        app.view_mode == index,
                                        egui::RichText::new(i18n::t(*label)).size(11.0),
                                    )
                                    .clicked()
                                {
                                    app.view_mode = index;
                                }
                            }
                        });
                    ui.checkbox(
                        &mut app.layers.bones,
                        egui::RichText::new(i18n::t("Bone Overlay")).size(11.0),
                    )
                    .on_hover_text(i18n::t("Show the skeleton used for FBX export over the motion"));
                });
                // Reserve the layer strip first, then give the rest to the canvas.
                let body = ui.available_rect_before_wrap();
                let rect = egui::Rect::from_min_size(
                    body.min,
                    egui::vec2(body.width(), (body.height() - FOOTER_HEIGHT).max(1.0)),
                );
                // Borrow the fields separately: the scene keeps references into
                // `people`/`mesh` while the camera is updated in place.
                let selected = app.selected;
                let people = &app.people;
                let mesh = app.mesh.as_deref();
                let layers = &app.layers;
                let floor = app.floor;
                let flow = &app.flow;
                let smplh_faces = app.smplh_faces.as_deref();
                let empty: &[[u32; 3]] = &[];
                let mhr_faces = mesh.map(|mesh| mesh.triangles.as_slice()).unwrap_or(empty);
                // Imported characters replace their person's body: each one is
                // posed by the clip's joint rotations, with its root put on the
                // clip's pelvis so it stands where the motion stands.
                let character_rest = app.character_rest.as_deref();
                let character_rest_positions = app.character_rest_positions.as_deref();
                let posed: Vec<Option<Vec<f32>>> = people
                    .iter()
                    .enumerate()
                    .map(|(slot, person)| {
                        let assigned = person.character.as_ref()?;
                        let rest = character_rest?;
                        let rest_positions = character_rest_positions?;
                        let clip = flow.get(slot).filter(|clip| !clip.skeletons.is_empty())?;
                        let index = app
                            .playhead
                            .saturating_sub(clip.first_frame)
                            .min(clip.skeletons.len() - 1);
                        let ours = &clip.skeletons[index];
                        let rotations = rotations_of(ours);
                        let bind = assigned
                            .character
                            .bones
                            .iter()
                            .find(|bone| bone.canonical == Some(0))
                            .or_else(|| assigned.character.bones.first())
                            .map(|bone| bone.bind)
                            .unwrap_or([0.0; 3]);
                        let translation = [
                            ours.positions[0] - bind[0],
                            ours.positions[1] - bind[1],
                            ours.positions[2] - bind[2],
                        ];
                        Some(assigned.character.pose(
                            &rotations,
                            rest,
                            rest_positions,
                            &clip.bvh.names,
                            translation,
                        ))
                    })
                    .collect();
                let draws: Vec<viewport::PersonDraw<'_>> = people
                    .iter()
                    .enumerate()
                    .filter(|(_, person)| person.visible)
                    .filter_map(|(slot, person)| {
                        // An imported character wins over the built-in body.
                        if let Some(Some(verts)) = posed.get(slot) {
                            if let Some(assigned) = person.character.as_ref() {
                                return Some(viewport::PersonDraw {
                                    name: &person.name,
                                    colour: person.colour,
                                    verts,
                                    keypoints: &[],
                                    faces: &assigned.character.faces,
                                    uv: Some(&assigned.character.uv),
                                    textures: &assigned.character.textures,
                                    face_texture: &assigned.character.face_texture,
                                    selected: slot == selected,
                                });
                            }
                        }
                        // Prefer the FlowHMR body **inside the clip's own frame
                        // range**: it is the temporally coherent motion the export
                        // uses. A clip can cover only part of a long capture, and
                        // holding its last pose for the rest of the take is what
                        // read as "the animation stops halfway": outside the range
                        // the per-frame reconstructed body takes over, and only
                        // when that frame is missing too does the clip hold.
                        if let (Some(clip), Some(faces)) = (flow.get(slot), smplh_faces) {
                            let inside = app.playhead >= clip.first_frame
                                && app.playhead < clip.first_frame + clip.frames.len();
                            if inside {
                                if let Some(verts) =
                                    clip.frame(app.playhead).filter(|verts| !verts.is_empty())
                                {
                                    return Some(viewport::PersonDraw {
                                        name: &person.name,
                                        colour: person.colour,
                                        verts,
                                        keypoints: &[],
                                        faces,
                                        uv: None,
                                        textures: &[],
                                        face_texture: &[],
                                        selected: slot == selected,
                                    });
                                }
                            }
                        }
                        if let Some(frame) = person.frame(app.playhead) {
                            return Some(viewport::PersonDraw {
                                name: &person.name,
                                colour: person.colour,
                                verts: &frame.verts,
                                keypoints: &frame.keypoints,
                                faces: mhr_faces,
                                uv: None,
                                textures: &[],
                                face_texture: &[],
                                selected: slot == selected,
                            });
                        }
                        // Neither: hold the clip's nearest pose so the person does
                        // not vanish at the very edges of the capture.
                        let (Some(clip), Some(faces)) = (flow.get(slot), smplh_faces) else {
                            return None;
                        };
                        clip.frame_clamped(app.playhead)
                            .map(|verts| viewport::PersonDraw {
                                name: &person.name,
                                colour: person.colour,
                                verts,
                                keypoints: &[],
                                faces,
                                uv: None,
                                textures: &[],
                                face_texture: &[],
                                selected: slot == selected,
                            })
                    })
                    .collect();
                if std::env::var_os("VELOMOCAP_DEBUG_MESH").is_some() {
                    let mut line = format!("[scene] playhead {0} draws {1}:", app.playhead, draws.len());
                    for draw in &draws {
                        let count = draw.verts.len() / 3;
                        let mut lo = [f32::INFINITY; 3];
                        let mut hi = [f32::NEG_INFINITY; 3];
                        let mut broken = 0usize;
                        for point in draw.verts.chunks_exact(3) {
                            if !point.iter().all(|value| value.is_finite()) {
                                broken += 1;
                                continue;
                            }
                            for axis in 0..3 {
                                lo[axis] = lo[axis].min(point[axis]);
                                hi[axis] = hi[axis].max(point[axis]);
                            }
                        }
                        line.push_str(&format!(
                            " {}[{count} v, {} f, bbox {:?}..{:?}, bad {}]",
                            draw.name,
                            draw.faces.len(),
                            lo,
                            hi,
                            broken
                        ));
                    }
                    line.push_str(&format!(
                        " | orbit target {:?} dist {:.2} framed {}",
                        app.orbit.target, app.orbit.distance, app.orbit.framed
                    ));
                    eprintln!("{line}");
                }
                // The trail belongs to the body that is on screen, and it is a
                // walk *on the floor*: the point under each frame's root at the
                // height of the lowest joint — the sole of the supporting foot.
                // A path through the root itself reads as a line through the
                // character's hips, which is what it looked like before. It is
                // also a trail and not a roadmap: only the frames that have
                // already played, and only the last few seconds of them, so it
                // grows with the take and fades out behind.
                let lowest_of = |points: &[f32]| -> f32 {
                    points
                        .chunks_exact(3)
                        .map(|point| point[1])
                        .fold(f32::INFINITY, f32::min)
                };
                let path: Vec<[f32; 3]> = flow
                    .get(selected)
                    .filter(|clip| !clip.skeletons.is_empty())
                    .map(|clip| {
                        clip.skeletons
                            .iter()
                            .map(|skeleton| {
                                let sole = lowest_of(&skeleton.positions);
                                [
                                    skeleton.positions[0],
                                    if sole.is_finite() { sole } else { floor },
                                    skeleton.positions[2],
                                ]
                            })
                            .collect()
                    })
                    .or_else(|| {
                        people.get(selected).map(|person| {
                            person
                                .frames
                                .iter()
                                .flatten()
                                .filter(|frame| frame.joints.len() >= 3)
                                .map(|frame| {
                                    let sole = lowest_of(&frame.joints);
                                    [
                                        frame.joints[0],
                                        if sole.is_finite() { sole } else { floor },
                                        frame.joints[2],
                                    ]
                                })
                                .collect()
                        })
                    })
                    .unwrap_or_default();
                let trail: Vec<[f32; 3]> = {
                    // Three seconds at 30 fps: long enough to read a stride,
                    // short enough that the path never becomes a second drawing.
                    const TRAIL_FRAMES: usize = 90;
                    trail_window(&path, app.playhead, TRAIL_FRAMES).to_vec()
                };
                let scene = Scene {
                    people: &draws,
                    trail: &trail,
                    layers,
                    view: ViewMode::from_index(app.view_mode),
                    floor,
                    empty_hint: if app.video.is_none() {
                        (
                            i18n::t("No video open"),
                            i18n::t("File → Import video… to start; the viewport fills once a clip is open."),
                        )
                    } else if app.people.is_empty() {
                        (
                            i18n::t("No people yet"),
                            i18n::t("File → Reconstruct all people runs the detector and the body model."),
                        )
                    } else {
                        (
                            i18n::t("No motion yet"),
                            i18n::t("The motion stage is still running, or it produced no clip."),
                        )
                    },
                };
                if viewport::show(ui, rect, &scene, &mut app.orbit, &mut app.view) {
                    ctx.request_repaint();
                }
                let footer_rect = egui::Rect::from_min_max(
                    egui::pos2(body.left(), rect.bottom()),
                    body.max,
                );
                let mut footer_ui = ui.new_child(
                    egui::UiBuilder::new()
                        .max_rect(footer_rect)
                        .layout(egui::Layout::left_to_right(egui::Align::Center)),
                );
                footer_checks(app, &mut footer_ui);
                ui.advance_cursor_after_rect(body);
            });
        });
}

fn mode_label(mode: ViewMode) -> &'static str {
    match mode {
        ViewMode::Perspective => i18n::t("Perspective"),
        ViewMode::Front => i18n::t("Front"),
        ViewMode::Back => i18n::t("Back"),
        ViewMode::Right => i18n::t("Right"),
        ViewMode::Left => i18n::t("Left"),
        ViewMode::Top => i18n::t("Top"),
        ViewMode::Bottom => i18n::t("Bottom"),
    }
}

fn footer_checks(app: &mut VeloMocap, ui: &mut egui::Ui) {
    ui.spacing_mut().item_spacing.x = 8.0;
    ui.checkbox(&mut app.layers.grid, egui::RichText::new(i18n::t("Floor Grid")).size(11.0));
    ui.checkbox(
        &mut app.layers.motion_path,
        egui::RichText::new(i18n::t("Motion Path")).size(11.0),
    );
    ui.checkbox(&mut app.layers.names, egui::RichText::new(i18n::t("Name")).size(11.0));
    ui.checkbox(
        &mut app.layers.follow,
        egui::RichText::new(i18n::t("Follow Actor")).size(11.0),
    );
    // Three dead checkboxes used to sit here (correction floor, locks, camera
    // frustums) with "not part of this build" on hover. Two of those features do
    // exist — as the floor field in the sidebar and the lock rows under the
    // timeline — so the controls read as broken rather than absent.
}

fn timeline(app: &mut VeloMocap, ctx: &egui::Context) {
    // How tall the panel has to be for every row to be visible: header, clip
    // label, ruler, one row per person and the four lock rows. The panel can be
    // dragged taller, but it may not be dragged short enough to cut a row — a
    // take with one person was already losing its fourth lock row.
    let lock_rows = usize::from(!app.flow.is_empty() || !app.people.is_empty()) * 4;
    let rows = 1 + app.people.len() + lock_rows;
    let needed = (140.0 + rows as f32 * 19.0).min(380.0);
    egui::TopBottomPanel::bottom("timeline")
        // Resizable, like the reference's splitter: a take with several people
        // needs more rows than the default, and the lock rows used to be clipped
        // away entirely at the bottom of a fixed-height panel.
        .resizable(true)
        .default_height(TIMELINE_HEIGHT)
        .min_height(needed)
        .max_height(ctx.available_rect().height() * 0.7)
        .frame(egui::Frame::new().fill(WINDOW_BG).inner_margin(egui::Margin::same(6)))
        .show(ctx, |ui| {
            panel(ui, |ui| {
                // Header: title and help on the left, transport in the middle,
                // frame/time/speed on the right — as in the Qt timeline.
                egui::Frame::new()
                    .fill(HEADER_BG)
                    .inner_margin(egui::Margin::symmetric(9, 3))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(i18n::t("Timeline")).color(TEXT));
                            if ui
                                .add(egui::Button::new("?").min_size(egui::vec2(22.0, 18.0)))
                                .on_hover_text(i18n::t("How to seek, trim and edit contacts"))
                                .clicked()
                            {
                                app.show_help = true;
                            }
                            ui.add_space(12.0);
                            transport(app, ui);
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    // Right-to-left: speed, time, frame number.
                                    egui::ComboBox::from_id_salt("play-speed")
                                        .selected_text(format!("{}x", app.play_speed))
                                        .width(58.0)
                                        .show_ui(ui, |ui| {
                                            for speed in [0.25f32, 0.5, 1.0, 1.5, 2.0] {
                                                if ui
                                                    .selectable_label(
                                                        (app.play_speed - speed).abs() < 0.01,
                                                        format!("{speed}x"),
                                                    )
                                                    .clicked()
                                                {
                                                    app.play_speed = speed;
                                                }
                                            }
                                        });
                                    let frames = app.frame_count();
                                    let fps = app.current_fps();
                                    let current =
                                        app.playhead.min(frames.saturating_sub(1));
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "{:.2} / {:.2} s",
                                            current as f64 / fps.max(1e-6),
                                            frames.max(1) as f64 / fps.max(1e-6)
                                        ))
                                        .size(11.0)
                                        .color(MUTED),
                                    );
                                    let mut frame_number = current + 1;
                                    if ui
                                        .add(
                                            egui::DragValue::new(&mut frame_number)
                                                .range(1..=frames.max(1))
                                                .speed(0.2),
                                        )
                                        .changed()
                                    {
                                        app.seek_to(frame_number.saturating_sub(1));
                                        app.playing = false;
                                    }
                                },
                            );
                        });
                    });
                clip_label(app, ui);
                ui.add_space(2.0);
                frame_ruler(app, ui);
                ui.add_space(4.0);
                // The rows scroll instead of being cut off: a person row per actor
                // plus the four lock rows already overflowed the old fixed height,
                // and a take with several people overflowed it a lot.
                egui::ScrollArea::vertical()
                    .id_salt("timeline-tracks")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        tracks(app, ui);
                    });
            });
        });
}

fn transport(app: &mut VeloMocap, ui: &mut egui::Ui) {
    let frames = app.frame_count();
    let button = |ui: &mut egui::Ui, label: &str, width: f32| -> bool {
        ui.add(egui::Button::new(egui::RichText::new(label).size(11.0)).min_size(egui::vec2(width, 20.0)))
            .clicked()
    };
    if button(ui, "|<", 26.0) {
        app.seek_to(0);
    }
    if button(ui, "<", 26.0) {
        app.seek_to(app.playhead.saturating_sub(1));
    }
    let play_label = if app.playing { "||" } else { ">" };
    if ui
        .add(
            egui::Button::new(egui::RichText::new(play_label).size(12.0))
                .fill(ACCENT)
                .stroke(egui::Stroke::new(1.0_f32, ACCENT_DIM))
                .min_size(egui::vec2(34.0, 20.0)),
        )
        .on_hover_text(i18n::t("Play / pause (Space)"))
        .clicked()
    {
        app.playing = !app.playing;
    }
    if button(ui, ">", 26.0) {
        app.seek_to((app.playhead + 1).min(frames.saturating_sub(1)));
    }
    if button(ui, ">|", 26.0) {
        app.seek_to(frames.saturating_sub(1));
    }
    // Reconstruction progress with a stop button, so a long run is visible and
    // interruptible without hunting through the menu.
    let running = app
        .capture
        .as_ref()
        .map(|job| job.is_running())
        .unwrap_or(false);
    if running {
        ui.add_space(8.0);
        let (frame, total, stage) = app.progress.clone();
        // Two bars side by side: the whole run (which stage of six) and the work
        // inside the current stage. The old single bar kept showing the SAM frame
        // count, so the long camera and motion stages looked hung at 100 %.
        let overall = (app.progress_stage as f32 + stage_fraction(frame, total))
            / crate::capture::STAGE_COUNT as f32;
        ui.add(
            egui::ProgressBar::new(overall.clamp(0.0, 1.0))
                .desired_width(200.0)
                .text(stage),
        )
        .on_hover_text(i18n::t("Overall progress"));
        let detail = if total > 0 {
            format!("{frame}/{total}  {:.0}%", 100.0 * frame as f32 / total as f32)
        } else {
            i18n::t("working…").to_string()
        };
        ui.add(
            egui::ProgressBar::new(stage_fraction(frame, total))
                .desired_width(130.0)
                .text(detail),
        )
        .on_hover_text(crate::capture::stage_name(app.progress_stage));
        if ui
            .add(
                egui::Button::new(egui::RichText::new(i18n::t("Stop")).size(11.0))
                    .min_size(egui::vec2(46.0, 20.0)),
            )
            .clicked()
        {
            app.cancel_capture();
        }
    }
}

/// Fraction of a stage's own work that is done; `0` while it is unknown.
fn stage_fraction(frame: usize, total: usize) -> f32 {
    if total > 0 {
        (frame as f32 / total as f32).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn clip_label(app: &mut VeloMocap, ui: &mut egui::Ui) {
    egui::Frame::new()
        .fill(HEADER_BG)
        .inner_margin(egui::Margin::symmetric(9, 3))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                let label = app
                    .video
                    .as_ref()
                    .map(|p| i18n::clip_line(&p.display().to_string()))
                    .unwrap_or_else(|| "Clip: no video".to_string());
                ui.add(
                    egui::Label::new(egui::RichText::new(label).size(11.0).color(TEXT))
                        .truncate(),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // "Reset Trim" was a permanently disabled button: trimming
                    // does not exist in this build, so a control that never
                    // works only reads as broken. The two that do exist are here.
                    let has_locks = !app.locks.is_automatic();
                    if ui
                        .add_enabled(
                            has_locks,
                            egui::Button::new(i18n::t("Reset Locks"))
                                .min_size(egui::vec2(76.0, 18.0)),
                        )
                        .on_hover_text(i18n::t("Put all four lock tracks back on automatic"))
                        .clicked()
                    {
                        app.reset_locks(None);
                    }
                    if ui
                        .add(egui::Button::new(i18n::t("Frame All")).min_size(egui::vec2(64.0, 18.0)))
                        .clicked()
                    {
                        app.orbit.framed = false;
                    }
                });
            });
        });
}

/// A clip pose's per-joint world rotations, in the layout a character is posed
/// with.
fn rotations_of(skeleton: &velomocap_body::Skeleton) -> Vec<[f32; 9]> {
    (0..skeleton.rotations.len() / 9)
        .map(|joint| {
            let mut rotation = [0.0f32; 9];
            rotation.copy_from_slice(&skeleton.rotations[joint * 9..joint * 9 + 9]);
            rotation
        })
        .collect()
}

fn frame_ruler(app: &mut VeloMocap, ui: &mut egui::Ui) {    let frames = app.frame_count();
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(i18n::t("Frame")).size(11.0));
        let mut value = app.playhead.min(frames.saturating_sub(1)) as f64;
        // egui sliders are `spacing.slider_width` wide whatever `add_sized` asks
        // for, which left the seek bar as a 100 px stub in the corner of an
        // otherwise empty row. Give it the whole row.
        ui.spacing_mut().slider_width = (ui.available_width() - 46.0).max(120.0);
        let slider = egui::Slider::new(&mut value, 0.0..=(frames.saturating_sub(1)) as f64)
            .show_value(false)
            .trailing_fill(true);
        let response = ui.add(slider);
        if response.changed() {
            app.seek_to(value.round() as usize);
            app.playing = false;
        }
        ui.label(
            egui::RichText::new(format!("{}", frames.max(1)))
                .size(11.0)
                .color(MUTED),
        );
    });
}

/// Frame under a horizontal position inside a track row.
fn track_frame(position: f32, rect: egui::Rect, frames: usize) -> usize {
    let fraction = ((position - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0);
    (fraction * frames.max(2).saturating_sub(1) as f32).round() as usize
}

fn tracks(app: &mut VeloMocap, ui: &mut egui::Ui) {
    // Rows sit close together, like the reference's 26 px track height: the
    // default spacing turned six rows into 24 px each, which pushed the fourth
    // lock row out of the panel even at its default height.
    ui.spacing_mut().item_spacing.y = 1.0;
    let frames = app.frame_count().max(1);
    // One row per person, plus the source video, like the Qt track list.
    let mut rows: Vec<(String, Option<egui::Color32>, Vec<bool>)> = Vec::new();
    rows.push((
        i18n::t("Video").to_string(),
        if app.video.is_some() { Some(OK) } else { None },
        Vec::new(),
    ));
    for person in &app.people {
        let present: Vec<bool> = person.frames.iter().map(|slot| slot.is_some()).collect();
        rows.push((
            person.name.clone(),
            if person.visible {
                Some(person.colour)
            } else {
                None
            },
            present,
        ));
    }
    for (label, colour, present) in rows {
        ui.horizontal(|ui| {
            ui.add_sized(
                [64.0, 16.0],
                egui::Label::new(egui::RichText::new(label).size(11.0).color(TEXT)),
            );
            let width = ui.available_width();
            // Clicking or dragging a track seeks to that frame: the rows are the
            // most obvious place to scrub, and they used to be hover-only, so a
            // click did nothing at all.
            let (rect, response) = ui.allocate_exact_size(
                egui::vec2(width, 17.0),
                egui::Sense::click_and_drag(),
            );
            let response = response.on_hover_cursor(egui::CursorIcon::PointingHand);
            if response.clicked() || response.dragged() {
                if let Some(position) = response.interact_pointer_pos() {
                    app.seek_to(track_frame(position.x, rect, frames));
                    app.playing = false;
                }
            }
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 2.0, CARD_BG);
            match colour {
                Some(colour) => {
                    if present.is_empty() {
                        painter.rect_filled(rect.shrink(2.0), 1.0, colour.gamma_multiply(0.55));
                    } else {
                        // One segment per observed frame.
                        let step = rect.width() / present.len().max(1) as f32;
                        for (index, present) in present.iter().enumerate() {
                            if !*present {
                                continue;
                            }
                            let x = rect.left() + index as f32 * step;
                            painter.rect_filled(
                                egui::Rect::from_min_max(
                                    egui::pos2(x, rect.top() + 2.0),
                                    egui::pos2((x + step.max(1.0)).min(rect.right()), rect.bottom() - 2.0),
                                ),
                                1.0,
                                colour.gamma_multiply(0.65),
                            );
                        }
                    }
                    // Playhead.
                    let x = rect.left() + rect.width() * (app.playhead.min(frames - 1) as f32
                        / (frames.max(2) - 1) as f32);
                    painter.line_segment(
                        [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                        egui::Stroke::new(1.4_f32, egui::Color32::WHITE),
                    );
                }
                None => {
                    painter.text(
                        rect.left_center() + egui::vec2(6.0, 0.0),
                        egui::Align2::LEFT_CENTER,
                        if present.is_empty() { i18n::t("No data") } else { i18n::t("Hidden") },
                        egui::FontId::proportional(10.5),
                        MUTED,
                    );
                }
            }
        });
    }
    // Contact locks: four sparse tracks per clip, drawn under the person rows so
    // a lock reads against the video above it (the reference's `footlock_editor`).
    if !app.flow.is_empty() || !app.people.is_empty() {
        app.load_locks_if_needed();
        lock_rows(app, ui, frames);
    }
}

/// Colours of the four lock tracks, in the reference's order.
const LOCK_COLOURS: [egui::Color32; 4] = [
    egui::Color32::from_rgb(0x56, 0xb6, 0xa7),
    egui::Color32::from_rgb(0xc7, 0x8f, 0x5f),
    egui::Color32::from_rgb(0x79, 0xa9, 0xdb),
    egui::Color32::from_rgb(0xbf, 0x8d, 0xcf),
];

/// Edge grab tolerance in pixels, like the reference's six-pixel edges.
const LOCK_EDGE: f32 = 6.0;

/// Where one drag wants a lock's interval to be.
///
/// Pure on purpose: the whole editing model (grow from the press, drag an edge
/// without crossing the other one, slide a lock without leaving the clip) is
/// testable without a window.
fn lock_frame_edit(
    grab: LockGrab,
    anchor: usize,
    interval: [usize; 2],
    frame: usize,
    frames: usize,
) -> [usize; 2] {
    let [start, end] = interval;
    let limit = frames.max(1);
    match grab {
        LockGrab::Start => [frame.min(end.saturating_sub(1)), end],
        LockGrab::End => [start, (frame + 1).max(start + 1).min(limit)],
        LockGrab::New => {
            let (low, high) = (anchor.min(frame), anchor.max(frame) + 1);
            [low, high.min(limit).max(low + 1)]
        }
        LockGrab::Move => {
            let span = end.saturating_sub(start).max(1);
            let delta = frame as i64 - anchor as i64;
            let mut moved = (start as i64 + delta).max(0) as usize;
            if moved + span > limit {
                moved = limit.saturating_sub(span);
            }
            [moved, (moved + span).min(limit)]
        }
    }
}

fn lock_rows(app: &mut VeloMocap, ui: &mut egui::Ui, frames: usize) {
    let frame_of = |rect: egui::Rect, x: f32| track_frame(x, rect, frames);
    let x_of = |rect: egui::Rect, frame: usize| {
        let last = frames.max(2) - 1;
        rect.left() + rect.width() * (frame.min(last) as f32 / last as f32)
    };

    // Delete the selection, undo/redo the edits.
    if let Some((track, index)) = app.selected_lock {
        let delete = ui.input(|state| {
            state.key_pressed(egui::Key::Delete) || state.key_pressed(egui::Key::Backspace)
        });
        if delete && app.locks.track(track).is_some_and(|list| index < list.len()) {
            app.delete_selected_lock();
        }
    }
    let undo = ui.input(|state| state.modifiers.command && state.key_pressed(egui::Key::Z));
    let redo_y = ui.input(|state| state.modifiers.command && state.key_pressed(egui::Key::Y));
    let shift = ui.input(|state| state.modifiers.shift);
    if undo {
        if shift {
            app.redo_lock();
        } else {
            app.undo_lock();
        }
    }
    if redo_y {
        app.redo_lock();
    }

    for track in 0..velomocap_motion::TRACK_COUNT {
        ui.horizontal(|ui| {
            ui.add_sized(
                [64.0, 15.0],
                egui::Label::new(
                    egui::RichText::new(i18n::t(velomocap_motion::TRACK_LABELS[track]))
                        .size(10.5)
                        .color(TEXT),
                ),
            );
            let width = ui.available_width();
            let (rect, response) =
                ui.allocate_exact_size(egui::vec2(width, 15.0), egui::Sense::click_and_drag());
            let colour = LOCK_COLOURS[track];

            // Press: grab an edge, the body of a lock, or empty space (a new
            // lock grows from there, like the reference's editor).
            if response.is_pointer_button_down_on() && app.lock_drag.is_none() {
                if let Some(position) = response.interact_pointer_pos() {
                    let frame = frame_of(rect, position.x);
                    let before = app.locks.track(track).map(<[_]>::to_vec);
                    let mut grab = (LockGrab::New, None);
                    // A one-frame lock is only a few pixels wide on a long clip,
                    // and the six-pixel edge tolerance then covers the whole
                    // bar: every press landed on an edge and the bar slid or
                    // shrank instead of being dragged where it was pointed at.
                    // Edges are only grabbable when there is room for a body.
                    let per_frame = rect.width() / (frames.max(2) - 1) as f32;
                    let tolerance = LOCK_EDGE.min(per_frame * 0.35);
                    if let Some(intervals) = app.locks.track(track) {
                        for (index, [start, end]) in intervals.iter().enumerate() {
                            let (left, right) = (x_of(rect, *start), x_of(rect, *end));
                            let roomy = (*end - *start) as f32 * per_frame >= 12.0;
                            if roomy && (position.x - left).abs() <= tolerance {
                                grab = (LockGrab::Start, Some(index));
                                break;
                            }
                            if roomy && (position.x - right).abs() <= tolerance {
                                grab = (LockGrab::End, Some(index));
                                break;
                            }
                            if left <= position.x && position.x <= right {
                                grab = (LockGrab::Move, Some(index));
                                break;
                            }
                        }
                    }
                    if grab.0 == LockGrab::New {
                        let mut list = before.clone().unwrap_or_default();
                        list.push([frame, (frame + 1).min(frames)]);
                        app.locks.set_track(track, list);
                        app.locks.normalise(frames);
                    }
                    let index = match grab.1 {
                        Some(index) => Some(index),
                        None => app
                            .locks
                            .track(track)
                            .and_then(|list| list.iter().position(|block| block.contains(&frame))),
                    };
                    app.selected_lock = index.map(|index| (track, index));
                    app.lock_drag = Some(LockDrag {
                        track,
                        grab: grab.0,
                        anchor: frame,
                        index,
                        before,
                    });
                }
            }

            // Drag: recompute the grabbed lock; the release turns it into one
            // undo step (a press that never moved leaves the one-frame lock it
            // just placed).
            if app.lock_drag.as_ref().is_some_and(|drag| drag.track == track) {
                let pointer = ui.ctx().pointer_latest_pos();
                if let Some(position) = pointer {
                    let mut drag = app.lock_drag.clone().expect("checked above");
                    let frame = frame_of(rect, position.x);
                    if let Some(index) = drag.index {
                        if let Some(interval) = app
                            .locks
                            .track(track)
                            .and_then(|list| list.get(index).copied())
                        {
                            let next = lock_frame_edit(drag.grab, drag.anchor, interval, frame, frames);
                            if next != interval {
                                let mut list =
                                    app.locks.track(track).map(<[_]>::to_vec).unwrap_or_default();
                                if let Some(slot) = list.get_mut(index) {
                                    *slot = next;
                                }
                                app.locks.set_track(track, list);
                                app.locks.normalise(frames);
                                // Normalising can merge or reorder, so the
                                // selection follows the frame it was made at.
                                if let Some(now) = app.locks.track(track).and_then(|list| {
                                    list.iter()
                                        .position(|block| block[0] <= next[0] && next[0] < block[1])
                                }) {
                                    app.selected_lock = Some((track, now));
                                    drag.index = Some(now);
                                }
                            }
                        }
                    }
                    app.lock_drag = Some(drag);
                }
                let released = response.drag_stopped() || !response.is_pointer_button_down_on();
                if released {
                    let drag = app.lock_drag.take().expect("checked above");
                    // A press without a drag is a *fixed frame*: it used to be
                    // created here and then thrown away again on release, so a
                    // click looked like it did nothing and the frame could only
                    // be placed by dragging a one-frame bar exactly.
                    app.locks.normalise(frames);
                    app.push_lock_edit(track, drag.before);
                }
            }

            // Draw the track.
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(0x2a, 0x2a, 0x2d));
            // The detector's own runs, dimmed: an automatic track then shows
            // *what* it follows instead of only the word "automatic", and a
            // hand-drawn track shows what the drawing is replacing. Without
            // this there is nothing to estimate a lock against.
            let auto: Vec<[usize; 2]> = app
                .flow
                .get(app.selected)
                .map(|clip| clip.contacts[track].clone())
                .unwrap_or_default();
            for [start, end] in &auto {
                let left = x_of(rect, *start);
                let right = x_of(rect, (*end).min(frames.saturating_sub(1)).max(*start));
                painter.rect_filled(
                    egui::Rect::from_min_max(
                        egui::pos2(left, rect.top() + 4.5),
                        egui::pos2(
                            (right.max(left + 1.0)).min(rect.right()),
                            rect.bottom() - 4.5,
                        ),
                    ),
                    1.0,
                    colour.gamma_multiply(0.30),
                );
            }
            match app.locks.track(track) {
                Some(intervals) => {
                    for (index, [start, end]) in intervals.iter().enumerate() {
                        let left = x_of(rect, *start);
                        let right = x_of(rect, (*end).min(frames.saturating_sub(1)));
                        let bar = egui::Rect::from_min_max(
                            egui::pos2(left, rect.top() + 2.0),
                            egui::pos2((right.max(left + 1.0)).min(rect.right()), rect.bottom() - 2.0),
                        );
                        let selected = app.selected_lock == Some((track, index));
                        painter.rect_filled(
                            bar,
                            1.0,
                            colour.gamma_multiply(if selected { 0.95 } else { 0.6 }),
                        );
                        painter.rect_stroke(
                            bar,
                            1.0,
                            egui::Stroke::new(
                                if selected { 1.6_f32 } else { 1.0 },
                                if selected {
                                    egui::Color32::from_rgb(0xd9, 0xe8, 0xfa)
                                } else {
                                    colour
                                },
                            ),
                            egui::StrokeKind::Inside,
                        );
                        if bar.width() > 42.0 {
                            painter.text(
                                bar.center(),
                                egui::Align2::CENTER_CENTER,
                                if selected {
                                    format!("{}–{}", start + 1, end)
                                } else {
                                    i18n::t("Lock").to_string()
                                },
                                egui::FontId::proportional(9.5),
                                egui::Color32::from_rgb(0x0d, 0x0d, 0x0d),
                            );
                        }
                    }
                }
                None => {
                    if auto.is_empty() {
                        painter.text(
                            rect.left_center() + egui::vec2(6.0, 0.0),
                            egui::Align2::LEFT_CENTER,
                            i18n::t("automatic"),
                            egui::FontId::proportional(9.5),
                            MUTED,
                        );
                    }
                }
            }
            let x = rect.left()
                + rect.width()
                    * (app.playhead.min(frames - 1) as f32 / (frames.max(2) - 1) as f32);
            painter.line_segment(
                [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                egui::Stroke::new(1.0_f32, ACCENT),
            );

            // The reference's context menu, item for item.
            response.clone().context_menu(|ui| {
                ui.label(i18n::t(velomocap_motion::TRACK_LABELS[track]));
                // Start from what the detector found instead of drawing the
                // whole track by hand: the runs are what the user is trying to
                // match, so they are the natural starting point.
                if !auto.is_empty()
                    && ui
                        .add(egui::Button::new(i18n::t("Copy the automatic runs")))
                        .clicked()
                {
                    let before = app.locks.track(track).map(<[_]>::to_vec);
                    app.locks.set_track(track, auto.clone());
                    app.locks.normalise(frames);
                    app.push_lock_edit(track, before);
                    ui.close();
                }
                if ui.button(i18n::t("Add lock here")).clicked() {
                    let frame = ui
                        .ctx()
                        .pointer_latest_pos()
                        .map(|position| frame_of(rect, position.x))
                        .unwrap_or(app.playhead);
                    let before = app.locks.track(track).map(<[_]>::to_vec);
                    let mut list = before.clone().unwrap_or_default();
                    list.push([frame, (frame + 1).min(frames)]);
                    app.locks.set_track(track, list);
                    app.locks.normalise(frames);
                    app.push_lock_edit(track, before);
                    ui.close();
                }
                let selected = app.selected_lock.map(|(index, _)| index) == Some(track);
                if ui
                    .add_enabled(selected, egui::Button::new(i18n::t("Delete lock")))
                    .clicked()
                {
                    app.delete_selected_lock();
                    ui.close();
                }
                ui.separator();
                if ui
                    .add_enabled(!app.lock_undo.is_empty(), egui::Button::new(i18n::t("Undo")))
                    .clicked()
                {
                    app.undo_lock();
                    ui.close();
                }
                if ui
                    .add_enabled(!app.lock_redo.is_empty(), egui::Button::new(i18n::t("Redo")))
                    .clicked()
                {
                    app.redo_lock();
                    ui.close();
                }
                ui.separator();
                if ui
                    .add_enabled(
                        app.locks.track(track).is_some(),
                        egui::Button::new(i18n::t("Reset this track to automatic")),
                    )
                    .clicked()
                {
                    app.reset_locks(Some(track));
                    ui.close();
                }
                if ui
                    .add_enabled(
                        !app.locks.is_automatic(),
                        egui::Button::new(i18n::t("Reset all locks to automatic")),
                    )
                    .clicked()
                {
                    app.reset_locks(None);
                    ui.close();
                }
            });
            if response.clicked() {
                if let Some(position) = response.interact_pointer_pos() {
                    app.seek_to(frame_of(rect, position.x));
                    app.playing = false;
                }
            }
        });
    }
    if app.locks_dirty {
        ui.label(
            egui::RichText::new(i18n::t("Locks edited — apply to update the motion"))
                .size(9.5)
                .color(MUTED),
        );
    }
}

fn sidebar(app: &mut VeloMocap, ctx: &egui::Context) {
    egui::SidePanel::right("sidebar")
        .resizable(true)
        .default_width(SIDEBAR_WIDTH)
        .min_width(240.0)
        .max_width(470.0)
        .frame(egui::Frame::new().fill(WINDOW_BG).inner_margin(egui::Margin::same(6)))
        .show(ctx, |ui| {
            panel(ui, |ui| {
                header(ui, i18n::t("AI Model"), |ui| {});
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        character_group(app, ui);
                        people_group(app, ui);
                        solver_group(app, ui);
                        correction_group(app, ui);
                        resources_group(app, ui);
                    });
            });
        });
}

fn group<R>(ui: &mut egui::Ui, title: &str, body: impl FnOnce(&mut egui::Ui) -> R) -> R {
    let mut frame = egui::Frame::new()
        .fill(PANEL_BG)
        .stroke(egui::Stroke::new(1.0_f32, BORDER))
        .inner_margin(egui::Margin::symmetric(7, 6));
    frame.outer_margin = egui::Margin::symmetric(0, 5);
    frame.show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(
            egui::RichText::new(title)
                .size(11.5)
                .strong()
                .color(TEXT),
        );
        ui.add_space(3.0);
        body(ui)
    })
    .inner
}

/// The details of the selected character card: what the file is, and the one
/// action that matters — put it on the person the workspace has selected.
fn character_group(app: &mut VeloMocap, ui: &mut egui::Ui) {    let Some(index) = app.character_panel else {
        return;
    };
    if index >= app.characters.len() {
        app.character_panel = None;
        return;
    }
    let entry = app.characters[index].clone();
    let name = entry
        .path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| entry.path.display().to_string());
    let format = entry
        .path
        .extension()
        .map(|extension| extension.to_string_lossy().to_uppercase())
        .unwrap_or_default();
    let matched = entry.character.matched();
    let bones = entry.character.bones.len();
    let vertices = entry.character.vertex_count();
    let faces = entry.character.faces.len();
    let unmatched: Vec<&str> = entry.character.unmatched().into_iter().take(8).collect();
    let person = app.people.get(app.selected).map(|person| person.name.clone());
    let wearing = app
        .people
        .get(app.selected)
        .and_then(|person| person.character.as_ref())
        .map(|assigned| assigned.path == entry.path)
        .unwrap_or(false);
    group(ui, i18n::t("Characters"), |ui| {
        ui.label(egui::RichText::new(&name).size(11.5).color(TEXT));
        ui.add(
            egui::Label::new(
                egui::RichText::new(format!(
                    "{format} · {matched}/{bones} {} · {vertices} {} · {faces} {}",
                    i18n::t("Bones matched"),
                    i18n::t("Vertices"),
                    i18n::t("Faces"),
                ))
                .size(10.0)
                .color(MUTED),
            )
            .truncate(),
        );
        ui.add(
            egui::Label::new(
                egui::RichText::new(entry.path.display().to_string())
                    .size(9.5)
                    .color(MUTED),
            )
            .truncate(),
        );
        if !unmatched.is_empty() {
            ui.label(
                egui::RichText::new(format!("{}: {}", i18n::t("Unmatched"), unmatched.join(", ")))
                    .size(9.5)
                    .color(MUTED),
            );
        }
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let label = match &person {
                Some(person) => i18n::zh_en(
                    &format!("用于「{person}」"),
                    &format!("Use on {person}"),
                ),
                None => i18n::t("Use on this person").to_string(),
            };
            if ui
                .add_enabled(
                    person.is_some() && !wearing,
                    egui::Button::new(egui::RichText::new(label).size(11.0)),
                )
                .on_hover_text(i18n::t(
                    "Replace the selected person's body with this model",
                ))
                .clicked()
            {
                app.use_character(app.selected, index);
            }
            if ui
                .add(egui::Button::new(
                    egui::RichText::new(i18n::t("Remove from list")).size(11.0),
                ))
                .clicked()
            {
                app.remove_character(index);
            }
        });
        if wearing {
            ui.label(
                egui::RichText::new(i18n::zh_en(
                    &format!("「{}」正在使用这个模型", person.unwrap_or_default()),
                    "This model is already on the selected person",
                ))
                .size(10.0)
                .color(OK),
            );
        }
    });
}

/// One row per person, like the reference's people list: clicking a row opens
/// that person's panel, where the built-in body can be replaced by an imported
/// character (and the choice is saved beside the clip).
fn people_group(app: &mut VeloMocap, ui: &mut egui::Ui) {    if app.people.is_empty() {
        return;
    }
    // What the dropdown can put on a person: the built-in body, or any model in
    // the shared library.
    let library: Vec<(usize, String)> = app
        .characters
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            (
                index,
                entry
                    .path
                    .file_stem()
                    .map(|stem| stem.to_string_lossy().into_owned())
                    .unwrap_or_else(|| entry.path.display().to_string()),
            )
        })
        .collect();
    group(ui, i18n::t("People"), |ui| {
        for index in 0..app.people.len() {
            let name = app.people[index].name.clone();
            let colour = app.people[index].colour;
            let assigned = app.people[index]
                .character
                .as_ref()
                .and_then(|assigned| assigned.path.file_stem())
                .map(|stem| stem.to_string_lossy().into_owned());
            let label = match &assigned {
                Some(stem) => format!("{name}  ·  {stem}"),
                None => name.clone(),
            };
            let selected = app.selected == index;
            let response = ui
                .add(
                    egui::Button::new(
                        egui::RichText::new(label)
                            .size(11.0)
                            .color(if selected { TEXT } else { colour }),
                    )
                    .frame(selected)
                    .min_size(egui::vec2(ui.available_width(), 18.0)),
                )
                .on_hover_text(i18n::t("Click to open this person's model settings"));
            if response.clicked() {
                app.selected = index;
                app.person_panel = Some(index);
                app.orbit.framed = false;
            }
            // The model this person wears, changeable right here.
            let chosen = assigned.clone().unwrap_or_else(|| i18n::t("Built-in").to_string());
            let mut pick: Option<Option<usize>> = None;
            ui.horizontal(|ui| {
                ui.add_space(6.0);
                egui::ComboBox::from_id_salt(("person-model", index))
                    .selected_text(egui::RichText::new(chosen).size(10.5))
                    .width(ui.available_width() - 4.0)
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_label(assigned.is_none(), i18n::t("Built-in"))
                            .clicked()
                        {
                            pick = Some(None);
                        }
                        for (slot, label) in &library {
                            let here = app.people[index]
                                .character
                                .as_ref()
                                .map(|assigned| assigned.path == app.characters[*slot].path)
                                .unwrap_or(false);
                            if ui.selectable_label(here, label).clicked() {
                                pick = Some(Some(*slot));
                            }
                        }
                    });
            });
            match pick {
                Some(Some(slot)) => {
                    app.selected = index;
                    app.use_character(index, slot);
                }
                Some(None) => app.clear_character(index),
                None => {}
            }
        }
    });
}

/// The panel a person row opens: which model is on this person, and how to
/// replace it.
fn person_window(app: &mut VeloMocap, ctx: &egui::Context) {
    let Some(index) = app.person_panel else {
        return;
    };
    if index >= app.people.len() {
        app.person_panel = None;
        return;
    }
    let name = app.people[index].name.clone();
    let current = app.people[index]
        .character
        .as_ref()
        .map(|assigned| assigned.path.display().to_string());
    let mut open = true;
    egui::Window::new(i18n::zh_en(
        &format!("人物：{name}"),
        &format!("Person: {name}"),
    ))
    .open(&mut open)
    .default_width(420.0)
    .resizable(false)
    .show(ctx, |ui| {
        ui.label(
            egui::RichText::new(match &current {
                Some(path) => path.clone(),
                None => i18n::t("Built-in body").to_string(),
            })
            .size(11.0)
            .color(MUTED),
        );
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if ui
                .add(egui::Button::new(i18n::t("Replace model…")))
                .on_hover_text(i18n::t("Import a VRM / glTF humanoid for this person"))
                .clicked()
            {
                app.import_character_dialog(Some(index));
            }
            if ui
                .add_enabled(
                    current.is_some(),
                    egui::Button::new(i18n::t("Use built-in body")),
                )
                .clicked()
            {
                app.clear_character(index);
            }
        });
    });
    if !open {
        app.person_panel = None;
    }
}

fn solver_group(app: &mut VeloMocap, ui: &mut egui::Ui) {
    group(ui, i18n::t("Solver Settings"), |ui| {
        ui.label(egui::RichText::new(i18n::t("Camera")).size(11.0).color(MUTED));
        egui::ComboBox::from_id_salt("camera-mode")
            .selected_text(if app.estimate_camera {
                i18n::t("Estimated camera (VGGT)")
            } else {
                i18n::t("Static Camera")
            })
            .width(ui.available_width())
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut app.estimate_camera,
                    false,
                    i18n::t("Static Camera"),
                );
                ui.selectable_value(
                    &mut app.estimate_camera,
                    true,
                    i18n::t("Estimated camera (VGGT)"),
                );
            });
        ui.add_space(3.0);
        ui.add_enabled(
            app.estimate_camera,
            egui::Checkbox::new(
                &mut app.refine_camera,
                egui::RichText::new(i18n::t("Refine Camera Rotation")).size(11.0),
            ),
        )
        .on_disabled_hover_text(i18n::t("Only used when the camera is estimated"));
        ui.checkbox(
            &mut app.layers.bones,
            egui::RichText::new(i18n::t("Bone overlay")).size(11.0),
        )
        .on_hover_text(i18n::t("Draw the 70-keypoint skeleton over the selected body"));
        ui.label(
            egui::RichText::new(i18n::people_line(
                app.people.len(),
                app.frame_count(),
                app.current_fps(),
            ))
            .size(10.5)
            .color(MUTED),
        );
    });
}

fn correction_group(app: &mut VeloMocap, ui: &mut egui::Ui) {
    use velomocap_motion::correction::{CorrectionMode, LockPreset};
    group(ui, i18n::t("Motion Correction"), |ui| {
        ui.checkbox(
            &mut app.correction.enabled,
            egui::RichText::new(i18n::t("Enable Correction")).size(11.0),
        )
        .on_hover_text(i18n::t(
            "Pin planted feet and resting hands after the motion is generated",
        ));
        ui.add_enabled_ui(app.correction.enabled, |ui| {
            ui.label(egui::RichText::new(i18n::t("Correction Mode")).size(11.0).color(MUTED));
            egui::ComboBox::from_id_salt("correction-mode")
                .selected_text(match app.correction.mode {
                    CorrectionMode::Flat => i18n::t("Flat Ground"),
                    CorrectionMode::MultiLevel => i18n::t("Multi-level"),
                })
                .width(ui.available_width())
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut app.correction.mode,
                        CorrectionMode::Flat,
                        i18n::t("Flat Ground"),
                    );
                    ui.selectable_value(
                        &mut app.correction.mode,
                        CorrectionMode::MultiLevel,
                        i18n::t("Multi-level"),
                    );
                });
            ui.horizontal(|ui| {
                ui.checkbox(
                    &mut app.correction.foot,
                    egui::RichText::new(i18n::t("Foot")).size(11.0),
                );
                ui.checkbox(
                    &mut app.correction.hand,
                    egui::RichText::new(i18n::t("Hand")).size(11.0),
                );
            });
            ui.label(egui::RichText::new(i18n::t("Lock Preset")).size(11.0).color(MUTED));
            egui::ComboBox::from_id_salt("lock-preset")
                .selected_text(match app.correction.preset {
                    LockPreset::Tight => i18n::t("Tight"),
                    LockPreset::Normal => i18n::t("Normal"),
                    LockPreset::Loose => i18n::t("Loose"),
                })
                .width(ui.available_width())
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut app.correction.preset,
                        LockPreset::Tight,
                        i18n::t("Tight"),
                    );
                    ui.selectable_value(
                        &mut app.correction.preset,
                        LockPreset::Normal,
                        i18n::t("Normal"),
                    );
                    ui.selectable_value(
                        &mut app.correction.preset,
                        LockPreset::Loose,
                        i18n::t("Loose"),
                    );
                });
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(i18n::t("Floor Height")).size(11.0));
                if ui
                    .add(
                        egui::DragValue::new(&mut app.correction.floor)
                            .speed(0.005)
                            .range(-2.0..=2.0)
                            .suffix(" m"),
                    )
                    .changed()
                {
                    // The grid follows the correction floor so both agree.
                    app.floor = app.correction.floor;
                }
                if ui
                    .add(egui::Button::new(i18n::t("Reset")).min_size(egui::vec2(44.0, 16.0)))
                    .clicked()
                {
                    app.correction.floor = 0.0;
                    app.floor = 0.0;
                }
            });
            ui.horizontal(|ui| {
                // The reference calls this "Pelvis Height Offset (from Auto)": it
                // may only give back the reach drop the root pass applied, so the
                // useful range is the same ±0.25 m.
                ui.label(egui::RichText::new(i18n::t("Pelvis Offset")).size(11.0))
                    .on_hover_text(i18n::t(
                        "Offset from the automatic pelvis height; positive gives back the reach drop, negative lowers the body",
                    ));
                ui.add(
                    egui::DragValue::new(&mut app.correction.pelvis_offset)
                        .speed(0.005)
                        .range(-0.25..=0.25)
                        .suffix(" m"),
                );
            });
            ui.add_space(4.0);
            // `Apply Correction`: re-solve the contact correction on the takes
            // already generated, with these settings and the timeline's locks —
            // no inference. The reference puts the same button here.
            let enabled = app.can_apply_correction();
            let label = egui::RichText::new(i18n::t("Apply Correction")).size(11.0);
            if ui
                .add_enabled(enabled, egui::Button::new(label).min_size(egui::vec2(0.0, 22.0)))
                .on_hover_text(i18n::t(
                    "Re-solve the contacts with these settings and the timeline's locks, without running the model again",
                ))
                .on_disabled_hover_text(i18n::t(
                    "Generate a motion first: a take loaded from a project has no saved solver input",
                ))
                .clicked()
            {
                app.apply_correction();
            }
            if !app.recorrect_note.is_empty() {
                ui.label(
                    egui::RichText::new(&app.recorrect_note)
                        .size(9.5)
                        .color(MUTED),
                );
            } else if app.locks_dirty {
                ui.label(
                    egui::RichText::new(i18n::t("Locks edited — apply to update the motion"))
                        .size(9.5)
                        .color(MUTED),
                );
            }
        });
    });
}

/// Label for the residency combo, in the user's language.
fn memory_mode_label(mode: crate::capture::MemoryMode) -> String {
    match mode {
        crate::capture::MemoryMode::Unlimited => i18n::zh_en("无限制（常驻显存）", "Unlimited (resident)"),
        crate::capture::MemoryMode::Scheduled => i18n::zh_en("计划（逐阶段装卸）", "Scheduled (per stage)"),
        crate::capture::MemoryMode::Hybrid => i18n::zh_en("混合（相机走 CPU）", "Hybrid (camera on CPU)"),
        crate::capture::MemoryMode::Cpu => i18n::zh_en("仅 CPU", "CPU only"),
    }
}

/// Which cards each mode is meant for, shown on hover.
fn memory_mode_hint(mode: crate::capture::MemoryMode) -> String {
    match mode {
        crate::capture::MemoryMode::Unlimited => i18n::zh_en(
            "适合 24 GB 及以上（3090/4090）：全程不卸载，响应最快",
            "24 GB and up (3090/4090): never unload, fastest response",
        ),
        crate::capture::MemoryMode::Scheduled => i18n::zh_en(
            "适合 16 GB：逐阶段实时装载与卸载，避免权重落内存",
            "16 GB: load and unload per stage, nothing spills to host memory",
        ),
        crate::capture::MemoryMode::Hybrid => i18n::zh_en(
            "适合 8 GB 左右：相机模型走 CPU，把显存留给 SAM 与动作模型",
            "Around 8 GB: the camera model runs on the CPU, the device goes to SAM and motion",
        ),
        crate::capture::MemoryMode::Cpu => i18n::zh_en(
            "完全使用内存，不占用任何显存（最慢）",
            "Everything in host memory, no device memory at all (slowest)",
        ),
    }
}
/// How a `.vp` store writes its geometry. This is the one place a project's size
/// is decided, so it lives with the other settings rather than in a dialog.
fn storage_group(app: &mut VeloMocap, ui: &mut egui::Ui) {
    group(ui, i18n::t("Project storage"), |ui| {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(i18n::t("Precision")).size(11.0));
            egui::ComboBox::from_id_salt("storage-precision")
                .selected_text(i18n::t(app.storage_precision.label()))
                .width((ui.available_width() - 8.0).max(80.0))
                .show_ui(ui, |ui| {
                    for precision in crate::project::Precision::ALL {
                        ui.selectable_value(
                            &mut app.storage_precision,
                            precision,
                            i18n::t(precision.label()),
                        );
                    }
                });
        });
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(i18n::t("Compression")).size(11.0));
            egui::ComboBox::from_id_salt("storage-compression")
                .selected_text(i18n::t(app.storage_compression.label()))
                .width((ui.available_width() - 8.0).max(80.0))
                .show_ui(ui, |ui| {
                    for compression in crate::project::Compression::ALL {
                        ui.selectable_value(
                            &mut app.storage_compression,
                            compression,
                            i18n::t(compression.label()),
                        );
                    }
                });
        });
        ui.label(
            egui::RichText::new(i18n::t("Storage precision note"))
                .size(9.5)
                .color(MUTED),
        );
    });
}

fn resources_group(app: &mut VeloMocap, ui: &mut egui::Ui) {
    group(ui, i18n::t("Processing Resources"), |ui| {
        ui.horizontal(|ui| {
            ui.checkbox(&mut app.gpu, egui::RichText::new("GPU (CUDA)").size(11.0));
            let (label, colour) = if app.cuda_ok {
                (i18n::t("provider found"), OK)
            } else {
                (i18n::t("CPU only"), egui::Color32::from_rgb(0xe8, 0xa8, 0x60))
            };
            ui.label(egui::RichText::new(label).size(10.0).color(colour));
        });
        // Residency strategy: auto-selected from the card's size at start-up,
        // overridable here. The four modes are what the pipeline supports.
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(i18n::t("Memory mode")).size(11.0));
            egui::ComboBox::from_id_salt("memory-mode")
                .selected_text(memory_mode_label(app.memory_mode))
                .width(ui.available_width() - 6.0)
                .show_ui(ui, |ui| {
                    for mode in crate::capture::MemoryMode::ALL {
                        let selected = app.memory_mode == mode;
                        let response = ui.selectable_label(selected, memory_mode_label(mode));
                        if response.clicked() && !selected {
                            app.memory_mode = mode;
                            app.gpu = mode.uses_gpu();
                            app.log(format!("Memory mode: {}", mode.label()));
                        }
                        response.on_hover_text(memory_mode_hint(mode));
                    }
                });
        })
        .response
        .on_hover_text(i18n::t(
            "Detected from the card at start-up; pick another mode to override it",
        ));
        if !app.vram.is_empty() {
            ui.label(egui::RichText::new(format!("VRAM {}", app.vram)).size(10.0).color(MUTED));
        }
        ui.add(
            egui::Slider::new(&mut app.scale_width, 320..=1920)
                .step_by(32.0)
                .text(i18n::t("Work width")),
        );
        ui.add(
            egui::Slider::new(&mut app.confidence, 0.05..=0.95)
                .text(i18n::t("Confidence"))
                .fixed_decimals(2),
        );
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(i18n::t("Max frames")).size(11.0));
            ui.add(
                egui::DragValue::new(&mut app.max_frames)
                    .speed(1.0)
                    .range(0..=100_000),
            )
            .on_hover_text(i18n::t("0 reconstructs the whole clip"));
            ui.label(egui::RichText::new(i18n::t("(0 = all)")).size(10.0).color(MUTED));
        });
        ui.add(
            egui::Slider::new(&mut app.play_speed, 0.1..=4.0)
                .text(i18n::t("Playback speed"))
                .fixed_decimals(2),
        );
        ui.checkbox(
            &mut app.use_flow,
            egui::RichText::new(i18n::t("FlowHMR motion")).size(11.0),
        )
        .on_hover_text(i18n::t(
            "After the reconstruction, generate one coherent SMPL-H clip per person with the FlowHMR ONNX model and preview that instead of the per-frame bodies",
        ));
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let (mark, colour) = if app.ort_path.is_some() {
                ("ONNX Runtime", OK)
            } else {
                ("ONNX Runtime", ERR)
            };
            ui.label(egui::RichText::new(mark).size(10.5).color(colour));
            let (mark, colour) = if app.tools_ok {
                ("ffmpeg", OK)
            } else {
                ("ffmpeg", ERR)
            };
            ui.label(egui::RichText::new(mark).size(10.5).color(colour));
        });
    });
}

/// The bottom strip: the clip on the left with the state of the last action, and
/// the numbers a glance needs on the right. It used to be a second title bar
/// above the panels, which repeated the window title and pushed everything down.
fn status_bar(app: &mut VeloMocap, ctx: &egui::Context) {
    egui::TopBottomPanel::bottom("status")
        .exact_height(STATUS_HEIGHT)
        .frame(
            egui::Frame::new()
                .fill(HEADER_BG)
                .inner_margin(egui::Margin::symmetric(10, 3)),
        )
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                let (mark, colour) = match (&app.video, &app.error) {
                    (_, Some(_)) => ("●", ERR),
                    (Some(_), None) => ("●", OK),
                    (None, None) => ("●", MUTED),
                };
                ui.label(egui::RichText::new(mark).size(9.0).color(colour));
                if let Some(error) = &app.error {
                    ui.add(
                        egui::Label::new(egui::RichText::new(error).size(11.0).color(ERR))
                            .truncate(),
                    );
                } else {
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(&app.status).size(11.0).color(colour),
                        )
                        .truncate(),
                    );
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add(egui::Button::new(i18n::t("Console")).min_size(egui::vec2(72.0, 18.0)))
                        .clicked()
                    {
                        app.show_console = !app.show_console;
                    }
                    // The portable package ships without the body model, so a
                    // fresh install cannot reconstruct anything. Say it here, once,
                    // with the way to fix it — rather than failing inside a run.
                    if app.key_model_missing() {
                        if ui
                            .add(
                                egui::Button::new(
                                    egui::RichText::new(i18n::zh_en("获取模型", "Get the model"))
                                        .size(10.5)
                                        .color(ACCENT),
                                )
                                .min_size(egui::vec2(78.0, 18.0)),
                            )
                            .on_hover_text(i18n::zh_en(
                                "打开 SMPL-H 官方下载页（download.is.tue.mpg.de，需注册）",
                                "Open the official SMPL-H download page (download.is.tue.mpg.de, registration required)",
                            ))
                            .clicked()
                        {
                            ui.ctx().open_url(egui::OpenUrl::new_tab(MODEL_DOWNLOAD_URL));
                        }
                        ui.label(
                            egui::RichText::new(i18n::zh_en(
                                "关键模型（SMPL-H）未下载",
                                "Key model (SMPL-H) not downloaded",
                            ))
                            .size(10.5)
                            .color(ERR),
                        );
                    }
                    let mut chips: Vec<String> = Vec::new();
                    chips.push(match &app.project_path {
                        Some(path) => path
                            .file_name()
                            .map(|name| name.to_string_lossy().to_string())
                            .unwrap_or_else(|| i18n::t("Project").into()),
                        None => i18n::zh_en("未保存", "unsaved").to_string(),
                    });
                    if app.video.is_some() {
                        chips.push(i18n::zh_en(
                            &format!("{} 人", app.people.len()),
                            &format!("{} people", app.people.len()),
                        ));
                        if app.video_size != (0, 0) {
                            chips.push(format!("{}×{}", app.video_size.0, app.video_size.1));
                        }
                        chips.push(format!("{:.2} fps", app.current_fps()));
                    }
                    for chip in &chips {
                        ui.label(egui::RichText::new(chip.as_str()).size(10.0).color(MUTED));
                        ui.label(egui::RichText::new("·").size(10.0).color(BORDER));
                    }
                    let busy = app.capture.as_ref().map(|j| j.is_running()).unwrap_or(false);
                    if busy {
                        let dot = egui::RichText::new("● working").size(10.5).color(ACCENT);
                        ui.label(dot);
                    }
                    // Device memory after the last stage: the number that shows
                    // whether the previous model really left the GPU.
                    if !app.vram.is_empty() {
                        ui.label(
                            egui::RichText::new(format!("VRAM {}", app.vram))
                                .size(10.5)
                                .color(MUTED),
                        )
                        .on_hover_text(i18n::t(
                            "Device memory after the last stage finished",
                        ));
                    }
                });
            });
        });
}

fn console_window(app: &mut VeloMocap, ctx: &egui::Context) {
    if !app.show_console {
        return;
    }
    let mut open = app.show_console;
    egui::Window::new(i18n::t("Console"))
        .open(&mut open)
        .default_size([620.0, 300.0])
        .resizable(true)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for line in &app.console {
                        ui.label(egui::RichText::new(line).monospace().size(11.0));
                    }
                });
        });
    app.show_console = open;
}

/// The About window: who makes it, where the source is, how to support it.
fn about_app_window(app: &mut VeloMocap, ctx: &egui::Context) {
    if !app.show_about {
        return;
    }
    let mut open = app.show_about;
    egui::Window::new(i18n::zh_en("关于 VeloMocap", "About VeloMocap"))
        .open(&mut open)
        .default_width(460.0)
        .min_width(380.0)
        .resizable(true)
        .collapsible(false)
        .show(ctx, |ui| {
            ui.label(egui::RichText::new("VeloMocap").size(18.0).strong().color(TEXT));
            ui.label(
                egui::RichText::new(format!(
                    "{} {} · {}",
                    i18n::t("Version"),
                    env!("CARGO_PKG_VERSION"),
                    i18n::zh_en("原生 Rust + C++，无 Python 运行时", "native Rust + C++, no Python at runtime"),
                ))
                .size(10.5)
                .color(MUTED),
            );
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new(i18n::zh_en(
                    "由 github/YuanBaoSMadLab 开源。",
                    "Open source by github/YuanBaoSMadLab.",
                ))
                .size(11.5)
                .color(TEXT),
            );
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(i18n::zh_en("仓库", "Repository")).size(10.5).color(MUTED));
                ui.hyperlink_to(
                    egui::RichText::new(REPOSITORY_URL).size(10.5),
                    REPOSITORY_URL,
                );
            });
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(i18n::zh_en("赞助", "Sponsor")).size(10.5).color(MUTED));
                ui.hyperlink_to(egui::RichText::new(SPONSOR_URL).size(10.5), SPONSOR_URL);
            });
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new(i18n::zh_en(
                    "如果你喜欢这个软件，就去点个小星星 ⭐ ——这对项目真的有用。",
                    "If you like it, leave a star ⭐ — it genuinely helps the project.",
                ))
                .size(11.0)
                .color(TEXT),
            );
            ui.add_space(8.0);
            ui.separator();
            ui.horizontal(|ui| {
                if ui
                    .button(i18n::zh_en("打开仓库", "Open repository"))
                    .clicked()
                {
                    ui.ctx().open_url(egui::OpenUrl::new_tab(REPOSITORY_URL));
                }
                if ui
                    .button(i18n::zh_en("去点星", "Star it"))
                    .clicked()
                {
                    ui.ctx().open_url(egui::OpenUrl::new_tab(REPOSITORY_URL));
                }
                if ui
                    .button(i18n::zh_en("赞助", "Sponsor"))
                    .clicked()
                {
                    ui.ctx().open_url(egui::OpenUrl::new_tab(SPONSOR_URL));
                }
                if ui.button(i18n::t("Close")).clicked() {
                    app.show_about = false;
                }
            });
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new(format!(
                    "core {} · kernels {} · {}",
                    velomocap_core::VERSION,
                    velomocap_core::backend(),
                    app.ort_path
                        .as_ref()
                        .map(|path| path.display().to_string())
                        .unwrap_or_else(|| "ONNX Runtime missing".into())
                ))
                .size(9.5)
                .color(MUTED),
            );
        });
    app.show_about = open && app.show_about;
}

fn about_window(app: &mut VeloMocap, ctx: &egui::Context) {
    if !app.show_help {
        return;
    }
    let mut open = app.show_help;
    egui::Window::new(i18n::t("Keyboard shortcuts"))
        .open(&mut open)
        .resizable(false)
        .collapsible(false)
        .show(ctx, |ui| {
            let shortcuts: [(&str, &str); 7] = if i18n::lang() == i18n::Lang::Zh {
                [
                    ("空格", "播放 / 暂停"),
                    ("← / →", "单帧前后"),
                    ("Shift + ← / →", "前后一秒"),
                    ("Home / End", "第一帧 / 最后一帧"),
                    ("左键拖动", "旋转 3D 视图"),
                    ("滚轮", "缩放 3D 视图"),
                    ("双击", "全部取景"),
                ]
            } else {
                [
                    ("Space", "Play / pause"),
                    ("Left / Right", "Step one frame"),
                    ("Shift + Left / Right", "Step one second"),
                    ("Home / End", "First / last frame"),
                    ("Left-drag", "Orbit the 3D view"),
                    ("Wheel", "Zoom the 3D view"),
                    ("Double-click", "Frame all"),
                ]
            };
            for (keys, action) in shortcuts {
                ui.horizontal(|ui| {
                    ui.add_sized(
                        [150.0, 16.0],
                        egui::Label::new(egui::RichText::new(keys).size(11.0)),
                    );
                    ui.label(egui::RichText::new(action).size(11.0).color(MUTED));
                });
            }
            ui.add_space(6.0);
            ui.label(
                egui::RichText::new(format!(
                    "core {}  ·  kernels {}  ·  {}",
                    velomocap_core::VERSION,
                    velomocap_core::backend(),
                    app.ort_path
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "ONNX Runtime missing".into())
                ))
                .size(10.0)
                .color(MUTED),
            );
        });
    app.show_help = open;
}

/// A muted caption that titles a block inside a window.
fn section_label(ui: &mut egui::Ui, text: impl Into<String>) {
    let text: String = text.into();
    ui.add_space(3.0);
    ui.label(
        egui::RichText::new(text)
            .size(10.0)
            .strong()
            .color(MUTED),
    );
}

/// The one button in a window that does the thing, filled with the accent so it
/// reads as the primary action rather than as one of equals.
fn primary_button(ui: &mut egui::Ui, text: impl Into<String>, enabled: bool, width: f32) -> bool {
    let text: String = text.into();
    let fill = if enabled { ACCENT } else { HEADER_BG };
    let stroke = if enabled { ACCENT } else { BORDER };
    let ink = if enabled { egui::Color32::WHITE } else { MUTED };
    ui.add_enabled(
        enabled,
        egui::Button::new(egui::RichText::new(text).size(12.0).strong().color(ink))
            .fill(fill)
            .stroke(egui::Stroke::new(1.0_f32, stroke))
            .corner_radius(5.0)
            .min_size(egui::vec2(width, 27.0)),
    )
    .clicked()
}

/// The export window: what to write, where, and under which name.
///
/// Shaped like a DCC's export panel on purpose — a format block, the parts to
/// include, one row per character with its own file name, the destination, then
/// the action. Everything it writes is a single `.glb` per character.
fn export_window(app: &mut VeloMocap, ctx: &egui::Context) {
    if !app.show_export {
        return;
    }
    let mut open = app.show_export;
    egui::Window::new(i18n::t("Export character assets"))
        .open(&mut open)
        .default_width(800.0)
        .default_height(560.0)
        .min_width(640.0)
        .min_height(420.0)
        .resizable(true)
        .show(ctx, |ui| {
            section_label(ui, i18n::zh_en("格式", "FORMAT"));
            ui.horizontal(|ui| {
                egui::Frame::new()
                    .fill(HEADER_BG)
                    .stroke(egui::Stroke::new(1.0_f32, ACCENT_DIM))
                    .corner_radius(4.0)
                    .inner_margin(egui::Margin::symmetric(8, 3))
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new("glTF Binary  ·  .glb")
                                .size(11.0)
                                .strong()
                                .color(TEXT),
                        );
                    });
                ui.label(
                    egui::RichText::new(i18n::zh_en(
                        "网格、骨骼、材质和动画都在这一个文件里",
                        "Mesh, skeleton, materials and animation in one file",
                    ))
                    .size(10.0)
                    .color(MUTED),
                );
            });

            section_label(ui, i18n::zh_en("内容", "INCLUDE"));
            ui.horizontal(|ui| {
                ui.checkbox(
                    &mut app.asset_skin,
                    i18n::zh_en("网格与蒙皮", "Mesh and skin"),
                );
                ui.checkbox(
                    &mut app.asset_materials,
                    i18n::zh_en("材质与贴图", "Materials"),
                );
                ui.checkbox(&mut app.asset_animation, i18n::zh_en("动画", "Animation"));
            });

            let rows: Vec<(usize, String, String)> = app
                .people
                .iter()
                .enumerate()
                .filter_map(|(index, person)| {
                    let assigned = person.character.as_ref()?;
                    let stem = assigned
                        .path
                        .file_stem()
                        .map(|stem| stem.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    Some((index, person.name.clone(), stem))
                })
                .collect();
            ui.horizontal(|ui| {
                section_label(ui, i18n::zh_en("角色", "CHARACTERS"));
                if rows.len() > 1 {
                    let every = rows.iter().all(|(index, _, _)| {
                        app.asset_selected.get(*index).copied().unwrap_or(true)
                    });
                    let mut all = every;
                    if ui
                        .checkbox(&mut all, i18n::zh_en("全选", "All"))
                        .changed()
                    {
                        for (index, _, _) in &rows {
                            if app.asset_selected.len() <= *index {
                                app.asset_selected.resize(index + 1, true);
                            }
                            app.asset_selected[*index] = all;
                        }
                    }
                }
            });
            for (index, person, stem) in &rows {
                if app.asset_selected.len() <= *index {
                    app.asset_selected.resize(index + 1, true);
                }
                if app.asset_names.len() <= *index {
                    app.asset_names.resize(index + 1, String::new());
                }
                ui.horizontal(|ui| {
                    let mut chosen = app.asset_selected[*index];
                    if ui
                        .checkbox(&mut chosen, egui::RichText::new(person).size(11.0))
                        .changed()
                    {
                        app.asset_selected[*index] = chosen;
                    }
                    // The file name is editable here, like a DCC export dialog:
                    // the model keeps its own name until the user types another.
                    ui.add(
                        egui::TextEdit::singleline(&mut app.asset_names[*index])
                            .hint_text(stem)
                            .desired_width(220.0)
                            .font(egui::TextStyle::Small),
                    );
                    ui.label(egui::RichText::new(".glb").size(10.0).color(MUTED));
                });
            }
            if rows.is_empty() {
                ui.label(
                    egui::RichText::new(i18n::zh_en(
                        "还没有人物使用导入的模型",
                        "No person is wearing an imported model yet",
                    ))
                    .size(10.0)
                    .color(MUTED),
                );
            }

            section_label(ui, i18n::zh_en("输出目录", "DESTINATION"));
            ui.horizontal(|ui| {
                let width = (ui.available_width() - 88.0).max(140.0);
                ui.add(
                    egui::TextEdit::singleline(&mut app.asset_folder)
                        .hint_text(i18n::zh_en("留空＝工程所在目录", "empty = beside the project"))
                        .desired_width(width),
                );
                if ui
                    .button(i18n::zh_en("浏览…", "Browse…"))
                    .on_hover_text(i18n::zh_en("选择输出文件夹", "Pick the output folder"))
                    .clicked()
                {
                    app.browse_asset_folder();
                }
            });
            if !app.asset_status.is_empty() {
                ui.label(egui::RichText::new(&app.asset_status).size(10.0).color(OK));
            }
            if let Some(note) = &app.export_note {
                ui.label(egui::RichText::new(note).size(10.0).color(MUTED));
            }
            // While the worker writes, show which file is being written and how
            // far along the run is: exporting two characters used to be two
            // silent seconds with nothing on screen.
            if let Some(job) = &app.asset_job {
                let state = job.progress.lock().unwrap();
                let fraction = if state.total == 0 {
                    0.0
                } else {
                    (state.done as f32 / state.total as f32).clamp(0.0, 1.0)
                };
                ui.add_space(6.0);
                let label = if state.cancelled {
                    i18n::zh_en("正在停止…", "Stopping…").to_string()
                } else {
                    format!(
                        "{}/{}  {}",
                        (state.done + 1).min(state.total),
                        state.total,
                        state.current
                    )
                };
                ui.add(
                    egui::ProgressBar::new(fraction)
                        .desired_height(16.0)
                        .show_percentage()
                        .text(egui::RichText::new(label).size(10.0)),
                );
                ui.label(
                    egui::RichText::new(i18n::zh_en(
                        "写文件时窗口不会卡住，可以继续用",
                        "The window stays live while the files are written",
                    ))
                    .size(9.5)
                    .color(MUTED),
                );
            }
            ui.add_space(8.0);
            ui.separator();
            ui.horizontal(|ui| {
                let running = app.asset_job.is_some();
                let can = !rows.is_empty() && app.project_path.is_some() && !running;
                if primary_button(ui, i18n::zh_en("导出", "Export"), can, 104.0) {
                    app.export_asset_models();
                }
                if running {
                    if ui
                        .button(i18n::zh_en("取消", "Cancel"))
                        .on_hover_text(i18n::zh_en(
                            "写完当前这个文件后停止",
                            "Stop after the file being written now",
                        ))
                        .clicked()
                    {
                        app.cancel_asset_export();
                    }
                }
                if ui.button(i18n::t("Close")).clicked() {
                    // Closing does not stop a running export: it keeps going in
                    // the background and the status line reports the result.
                    app.show_export = false;
                }
                if rows.is_empty() {
                    ui.label(
                        egui::RichText::new(i18n::zh_en(
                            "先在「人物」里给一行指定模型",
                            "Assign a model to a person first",
                        ))
                        .size(10.0)
                        .color(MUTED),
                    );
                } else if app.project_path.is_none() {
                    ui.label(
                        egui::RichText::new(i18n::zh_en("先保存工程", "Save the project first"))
                            .size(10.0)
                            .color(MUTED),
                    );
                }
            });
        });
    // A button in the window may have set `show_export` to false; without the
    // second half that assignment is undone here and Close does nothing.
    app.show_export = open && app.show_export;
}

/// Settings that belong to the application rather than to a clip or a panel.
///
/// The storage precision and compression used to sit at the bottom of the
/// sidebar, where they read as a property of the renderer and pushed the
/// per-clip settings off the screen.
fn settings_window(app: &mut VeloMocap, ctx: &egui::Context) {
    if !app.show_settings {
        return;
    }
    let mut open = app.show_settings;
    egui::Window::new(i18n::t("Settings"))
        .open(&mut open)
        .default_width(430.0)
        .min_width(360.0)
        .resizable(false)
        .collapsible(false)
        .show(ctx, |ui| {
            section_label(ui, i18n::zh_en("工程存储", "PROJECT STORAGE"));
            storage_group(app, ui);
            ui.add_space(8.0);
            ui.separator();
            ui.horizontal(|ui| {
                if primary_button(ui, i18n::zh_en("完成", "Done"), true, 92.0) {
                    app.show_settings = false;
                }
                ui.label(
                    egui::RichText::new(i18n::zh_en(
                        "设置存在本机，对所有工程生效",
                        "Stored on this machine, applied to every project",
                    ))
                    .size(10.0)
                    .color(MUTED),
                );
            });
        });
    app.show_settings = open && app.show_settings;
}

fn guide_window(app: &mut VeloMocap, ctx: &egui::Context) {    if !app.show_guide {
        return;
    }
    let mut open = app.show_guide;
    egui::Window::new(i18n::t("Workspace guide"))
        .open(&mut open)
        .default_width(520.0)
        .resizable(true)
        .show(ctx, |ui| {
            let chinese = i18n::lang() == i18n::Lang::Zh;
            ui.label(egui::RichText::new(if chinese { "流程" } else { "Workflow" }).strong());
            let workflow: &[&str] = if chinese {
                &[
                    "1. 文件 → 导入视频…，把片段放到工作台。",
                    "2. 文件 → 重建所有人物：YOLOv8 找出每个人，再对每人运行 SAM 3D Body。",
                    "3. 拖动时间轴、或按空格回放；下面四条锁定轨可以手绘接触帧。",
                    "4. 右侧「人物」列表点一行 → 替换模型…：导入 VRM / GLB 虚拟人物，",
                    "   用它替换内置身体播放这段动作（选择会存到片段旁边）。",
                    "5. 文件 → 导出角色资产…：勾网格/材质/动画、给每个角色起名，然后一键写出",
                ]
            } else {
                &[
                    "1. File → Import video… to open a clip.",
                    "2. File → Reconstruct all people detects everyone with YOLOv8 and runs",
                    "   the SAM 3D Body chain on each of them.",
                    "3. Scrub the timeline or press Space; the four lock rows below can be",
                    "   drawn by hand to fix contact timing.",
                    "4. Click a row in the People list on the right → Replace model… to",
                    "   import a VRM / GLB avatar and play this motion on it (the choice",
                    "   is saved next to the clip).",
                    "5. File → Export character assets… ticks mesh/material/animation and",
                    "   names each character, then writes one .glb per character.",
                ]
            };
            for line in workflow {
                ui.label(egui::RichText::new(*line).size(11.0).color(MUTED));
            }
            ui.add_space(8.0);
            ui.label(egui::RichText::new(if chinese { "面板" } else { "Panels" }).strong());
            let panels: &[&str] = if chinese {
                &[
                    "视频片段：源画面以及其中找到的人物。",
                    "3D 视口：旋转、缩放、取景；网格、地面、骨骼、轨迹。",
                    "时间轴：帧标尺、播放控制、倍速、每个人一行轨道。",
                    "侧栏：求解器、校正与处理资源设置。",
                ]
            } else {
                &[
                    "Movie Clip: the source frame and the people found in it.",
                    "3D Viewport: orbit, zoom, frame; meshes, grid, bones, path.",
                    "Timeline: frame ruler, transport, playback speed, per-person rows.",
                    "Sidebar: solver, correction and processing settings.",
                ]
            };
            for line in panels {
                ui.label(egui::RichText::new(*line).size(11.0).color(MUTED));
            }
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new(if chinese {
                    "这里的一切都是原生的：ffmpeg 解码，ONNX Runtime 跑模型，\
                     Rust 负责检测、跟踪、SAM 链路和渲染。没有 Python。"
                } else {
                    "Everything here runs natively: ffmpeg decodes, ONNX Runtime runs the \
                     models, and Rust does detection, tracking, the SAM chain and the renderer. \
                     No Python."
                })
                .size(11.0)
                .color(MUTED),
            );
        });
    app.show_guide = open;
}

/// Window title in the current language.
pub fn window_title() -> String {
    format!("VeloMocap | {}", i18n::t("Motion Workspace").trim())
}

fn person_palette() -> [egui::Color32; 5] {
    [
        egui::Color32::from_rgb(96, 178, 255),
        egui::Color32::from_rgb(255, 170, 96),
        egui::Color32::from_rgb(140, 220, 160),
        egui::Color32::from_rgb(230, 140, 220),
        egui::Color32::from_rgb(240, 210, 120),
    ]
}

/// The part of a motion path a trail shows: the frames that have already played,
/// and at most `span` of them. Frame 0 is one point, so nothing is drawn until
/// the take has actually moved.
fn trail_window(path: &[[f32; 3]], playhead: usize, span: usize) -> &[[f32; 3]] {
    let head = (playhead + 1).min(path.len());
    let start = head.saturating_sub(span.max(1));
    &path[start..head]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The trail grows with the playhead and forgets what is older than its
    /// span, so the viewport shows the last few metres walked rather than the
    /// whole take at once.
    #[test]
    fn the_trail_grows_with_the_playhead_and_fades_out_behind() {
        let path: Vec<[f32; 3]> = (0..10).map(|step| [step as f32, 0.0, 0.0]).collect();
        assert_eq!(trail_window(&path, 0, 4).len(), 1);
        assert_eq!(trail_window(&path, 3, 4).len(), 4);
        assert_eq!(trail_window(&path, 9, 4), &path[6..10]);
        assert_eq!(trail_window(&path, 9, 90), &path[..]);
        // A playhead past the end of the path cannot produce an empty slice out
        // of range.
        assert_eq!(trail_window(&path, 999, 4), &path[6..10]);
    }

    /// The lock drag arithmetic: growing from the press, dragging an edge
    /// without crossing the other one, and sliding a lock inside the clip.
    #[test]
    fn dragging_a_lock_edits_the_interval_it_grabbed() {
        use super::{lock_frame_edit, LockGrab};
        let frames = 100;
        // A new lock grows from the press in either direction.
        assert_eq!(lock_frame_edit(LockGrab::New, 10, [10, 11], 15, frames), [10, 16]);
        assert_eq!(lock_frame_edit(LockGrab::New, 10, [10, 11], 4, frames), [4, 11]);
        // Edges move, but never past each other.
        assert_eq!(lock_frame_edit(LockGrab::Start, 10, [10, 20], 5, frames), [5, 20]);
        assert_eq!(lock_frame_edit(LockGrab::Start, 10, [10, 20], 30, frames), [19, 20]);
        assert_eq!(lock_frame_edit(LockGrab::End, 10, [10, 20], 30, frames), [10, 31]);
        assert_eq!(lock_frame_edit(LockGrab::End, 10, [10, 20], 0, frames), [10, 11]);
        // The body slides with the pointer and stays inside the clip.
        assert_eq!(lock_frame_edit(LockGrab::Move, 15, [10, 20], 25, frames), [20, 30]);
        assert_eq!(lock_frame_edit(LockGrab::Move, 0, [0, 10], 500, frames), [90, 100]);
        assert_eq!(lock_frame_edit(LockGrab::Move, 50, [45, 55], 0, frames), [0, 10]);
    }

    /// An edit can only write frames the clip has.
    #[test]
    fn a_lock_never_leaves_the_clip() {
        use super::{lock_frame_edit, LockGrab};
        for grab in [LockGrab::New, LockGrab::Move, LockGrab::Start, LockGrab::End] {
            let result = lock_frame_edit(grab, 3, [3, 6], 9_999, 10);
            assert!(result[1] <= 10, "{grab:?} wrote past the end: {result:?}");
            let result = lock_frame_edit(grab, 3, [3, 6], 0, 10);
            assert!(result[1] > result[0], "{grab:?} emptied the lock: {result:?}");
        }
    }
    use super::track_frame;

    /// Clicking a track row must land on the frame under the pointer, and must
    /// clamp at both ends instead of running off the clip.
    #[test]
    fn clicking_a_track_maps_to_its_frame() {
        let rect = egui::Rect::from_min_size(egui::pos2(100.0, 0.0), egui::vec2(200.0, 17.0));
        assert_eq!(track_frame(100.0, rect, 10), 0, "left edge is frame 0");
        assert_eq!(track_frame(200.0, rect, 10), 5, "middle is the middle frame");
        assert_eq!(track_frame(300.0, rect, 10), 9, "right edge is the last frame");
        assert_eq!(track_frame(20.0, rect, 10), 0, "left of the row clamps");
        assert_eq!(track_frame(900.0, rect, 10), 9, "right of the row clamps");
        assert_eq!(track_frame(100.0, rect, 1), 0, "a single-frame clip stays at 0");
    }
}



