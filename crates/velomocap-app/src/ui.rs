//! Widgets and layout for the VeloMocap window.
//!
//! This mirrors the original RGB MoCap Qt workspace: the same panel order, the
//! same headers and footer strips, the same names and the same dark palette
//! (`mocap/theme.py`), so the two applications read as one product.

use eframe::egui;

use crate::app::{ExportFormat, LockDrag, LockGrab, VeloMocap};
use crate::i18n;
use crate::capture::CaptureStage;
use crate::viewport::{self, Scene, ViewMode};

// Palette from the Qt stylesheet.
const WINDOW_BG: egui::Color32 = egui::Color32::from_rgb(0x25, 0x25, 0x25);
const PANEL_BG: egui::Color32 = egui::Color32::from_rgb(0x30, 0x30, 0x30);
const HEADER_BG: egui::Color32 = egui::Color32::from_rgb(0x29, 0x29, 0x29);
const BORDER: egui::Color32 = egui::Color32::from_rgb(0x19, 0x19, 0x19);
const MENU_BG: egui::Color32 = egui::Color32::from_rgb(0x22, 0x22, 0x22);
const TEXT: egui::Color32 = egui::Color32::from_rgb(0xd5, 0xd5, 0xd5);
const MUTED: egui::Color32 = egui::Color32::from_rgb(0xa0, 0xa0, 0xa0);
const ACCENT: egui::Color32 = egui::Color32::from_rgb(0x47, 0x7b, 0xb7);
const ACCENT_DIM: egui::Color32 = egui::Color32::from_rgb(0x37, 0x68, 0x9d);
const ERR: egui::Color32 = egui::Color32::from_rgb(0xff, 0x8a, 0x80);
const OK: egui::Color32 = egui::Color32::from_rgb(0x8f, 0xd0, 0x9b);

const TITLE_HEIGHT: f32 = 26.0;
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
    visuals.extreme_bg_color = egui::Color32::from_rgb(0x24, 0x24, 0x24);
    visuals.faint_bg_color = egui::Color32::from_rgb(0x33, 0x33, 0x33);
    visuals.override_text_color = Some(TEXT);
    visuals.selection.bg_fill = ACCENT;
    visuals.selection.stroke = egui::Stroke::new(1.0_f32, egui::Color32::WHITE);
    visuals.window_stroke = egui::Stroke::new(1.0_f32, BORDER);
    visuals.widgets.noninteractive.bg_fill = PANEL_BG;
    visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0_f32, BORDER);
    visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(0x45, 0x45, 0x45);
    visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(0x24, 0x24, 0x24));
    visuals.widgets.inactive.corner_radius = 3.into();
    visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(0x55, 0x55, 0x55);
    visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(0x73, 0x73, 0x73));
    visuals.widgets.hovered.corner_radius = 3.into();
    visuals.widgets.active.bg_fill = egui::Color32::from_rgb(0x33, 0x33, 0x33);
    visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0_f32, ACCENT_DIM);
    visuals.widgets.active.corner_radius = 3.into();
    visuals.widgets.open.bg_fill = egui::Color32::from_rgb(0x45, 0x45, 0x45);
    visuals.window_corner_radius = 2.into();
    ctx.set_visuals(visuals);

    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(6.0, 5.0);
    style.spacing.button_padding = egui::vec2(9.0, 3.0);
    style.spacing.interact_size.y = 20.0;
    style.spacing.slider_width = 110.0;
    style.spacing.combo_width = 110.0;
    style.spacing.indent = 12.0;
    use egui::{FontFamily, FontId, TextStyle};
    style.text_styles = [
        (TextStyle::Small, FontId::new(10.5, FontFamily::Proportional)),
        (TextStyle::Body, FontId::new(12.0, FontFamily::Proportional)),
        (TextStyle::Button, FontId::new(12.0, FontFamily::Proportional)),
        (TextStyle::Heading, FontId::new(13.0, FontFamily::Proportional)),
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
    title_strip(app, ctx);
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
    export_window(app, ctx);
    clear_scene_prompt(app, ctx);
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
                    // Everything a written file needs — its format, the bone
                    // vocabulary, the rest correction and the name — lives in
                    // one export window. As a menu submenu the naming read like
                    // a property of the document rather than a choice made when
                    // a file is written.
                    let can_export = !app.flow.is_empty() || !app.people.is_empty();
                    if ui
                        .add_enabled(can_export, egui::Button::new(i18n::t("Export motion…")))
                        .on_hover_text(i18n::t(
                            "Format, skeleton naming, rest correction and file name",
                        ))
                        .clicked()
                    {
                        ui.close();
                        app.show_export = true;
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
                });
            });
        });
}

fn title_strip(app: &mut VeloMocap, ctx: &egui::Context) {
    egui::TopBottomPanel::top("title")
        .exact_height(TITLE_HEIGHT)
        .frame(
            egui::Frame::new()
                .fill(WINDOW_BG)
                .inner_margin(egui::Margin::symmetric(8, 3)),
        )
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                ui.label(
                    egui::RichText::new("VeloMocap")
                        .size(13.0)
                        .strong()
                        .color(egui::Color32::from_rgb(0xed, 0xed, 0xed)),
                );
                ui.label(egui::RichText::new(i18n::t("Motion Workspace   |")).color(MUTED));
                let name = app
                    .video
                    .as_ref()
                    .and_then(|p| p.file_name())
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| i18n::t("Open a video or drop it into the workspace").into());
                ui.add(egui::Label::new(egui::RichText::new(name).color(MUTED)).truncate());
            });
        });
}

/// A panel with the Qt `editor` look: 1 px border, flat body.
fn panel(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::new()
        .fill(PANEL_BG)
        .stroke(egui::Stroke::new(1.0_f32, BORDER))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.spacing_mut().item_spacing = egui::vec2(6.0, 4.0);
            add(ui);
        });
}

/// The `editorHeader` strip that titles every panel.
fn header(ui: &mut egui::Ui, title: &str, right: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::new()
        .fill(HEADER_BG)
        .inner_margin(egui::Margin::symmetric(9, 3))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            // `horizontal_centered` would stretch the strip to the panel height.
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(title).color(TEXT));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), right);
            });
        });
}

/// The matching footer strip (viewport layers, action bars).
fn footer(ui: &mut egui::Ui, left: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::new()
        .fill(HEADER_BG)
        .inner_margin(egui::Margin::symmetric(9, 3))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(left);
        });
}

fn movie_clip(app: &mut VeloMocap, ctx: &egui::Context) {
    egui::SidePanel::left("movie-clip")
        .exact_width(CLIP_PANEL_WIDTH)
        .resizable(false)
        .frame(egui::Frame::new().fill(WINDOW_BG).inner_margin(egui::Margin::same(4)))
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
                        let split = body.left() + 162.0;
                        let people = egui::Rect::from_min_max(
                            body.min,
                            egui::pos2(split, body.bottom()),
                        );
                        let video = egui::Rect::from_min_max(
                            egui::pos2(split + 4.0, body.min.y),
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
    // People | Stream tabs.
    ui.horizontal(|ui| {
        for (index, label) in [i18n::t("People"), i18n::t("Stream")].iter().enumerate() {
            let selected = index == 0;
            let text = egui::RichText::new(*label)
                .size(11.5)
                .color(if selected { egui::Color32::WHITE } else { TEXT });
            let button = egui::Button::new(text)
                .fill(if selected { ACCENT } else { HEADER_BG })
                .stroke(egui::Stroke::new(1.0_f32, if selected { ACCENT_DIM } else { BORDER }))
                .min_size(egui::vec2(58.0, 20.0));
            let _ = ui.add_enabled(index == 0, button);
        }
    });
    ui.add_space(4.0);
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

    egui::Frame::new()
        .fill(egui::Color32::from_rgb(0x24, 0x24, 0x24))
        .stroke(egui::Stroke::new(1.0_f32, BORDER))
        .inner_margin(egui::Margin::same(3))
        .show(ui, |ui| {
            ui.set_min_height((ui.available_height() - 28.0).max(60.0));
            ui.set_width(ui.available_width());
            egui::ScrollArea::vertical()
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
    painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(0x24, 0x24, 0x24));

    let Some(texture) = &app.preview else {
        painter.text(
            rect.center() - egui::vec2(0.0, 10.0),
            egui::Align2::CENTER_CENTER,
            i18n::t("Open a video to begin"),
            egui::FontId::proportional(13.0),
            egui::Color32::from_gray(190),
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
        .frame(egui::Frame::new().fill(WINDOW_BG).inner_margin(egui::Margin::same(4)))
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
                let draws: Vec<viewport::PersonDraw<'_>> = people
                    .iter()
                    .enumerate()
                    .filter(|(_, person)| person.visible)
                    .filter_map(|(slot, person)| {
                        // Prefer the FlowHMR body: it is the temporally coherent
                        // motion the export uses, and the one the original
                        // previews. Fall back to the per-frame MHR mesh while no
                        // clip has been generated for this person yet.
                        if let (Some(clip), Some(faces)) = (flow.get(slot), smplh_faces) {
                            if let Some(verts) = clip.frame_clamped(app.playhead) {
                                return Some(viewport::PersonDraw {
                                    name: &person.name,
                                    colour: person.colour,
                                    verts,
                                    keypoints: &[],
                                    faces,
                                    selected: slot == selected,
                                });
                            }
                        }
                        person
                            .frame(app.playhead)
                            .map(|frame| viewport::PersonDraw {
                                name: &person.name,
                                colour: person.colour,
                                verts: &frame.verts,
                                keypoints: &frame.keypoints,
                                faces: mhr_faces,
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
                let trail: Vec<[f32; 3]> = people                    .get(selected)
                    .map(|person| {
                        person
                            .frames
                            .iter()
                            .flatten()
                            .filter(|frame| frame.joints.len() >= 3)
                            .map(|frame| [frame.joints[0], frame.joints[1], frame.joints[2]])
                            .collect()
                    })
                    .unwrap_or_default();
                let scene = Scene {
                    people: &draws,
                    trail: &trail,
                    layers,
                    view: ViewMode::from_index(app.view_mode),
                    floor,
                    empty_hint: (
                        i18n::t("No motion loaded"),
                        i18n::t("Reconstruct a clip to see every person as a body here."),
                    ),
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
    let mut correction_floor = false;
    ui.add_enabled(
        false,
        egui::Checkbox::new(&mut correction_floor, egui::RichText::new(i18n::t("Correction Floor")).size(11.0)),
    )
    .on_disabled_hover_text(i18n::t("Motion correction is not part of this build"));
    ui.checkbox(
        &mut app.layers.motion_path,
        egui::RichText::new(i18n::t("Motion Path")).size(11.0),
    );
    let mut locks = false;
    ui.add_enabled(
        false,
        egui::Checkbox::new(&mut locks, egui::RichText::new(i18n::t("Lock")).size(11.0)),
    )
    .on_disabled_hover_text(i18n::t("Contact locks are not part of this build"));
    ui.checkbox(&mut app.layers.names, egui::RichText::new(i18n::t("Name")).size(11.0));
    let mut cameras = false;
    ui.add_enabled(
        false,
        egui::Checkbox::new(&mut cameras, egui::RichText::new(i18n::t("Camera")).size(11.0)),
    )
    .on_disabled_hover_text(i18n::t("Camera frustums are not part of this build"));
    ui.checkbox(
        &mut app.layers.follow,
        egui::RichText::new(i18n::t("Follow Actor")).size(11.0),
    );
}

fn timeline(app: &mut VeloMocap, ctx: &egui::Context) {
    egui::TopBottomPanel::bottom("timeline")
        .exact_height(TIMELINE_HEIGHT)
        .frame(egui::Frame::new().fill(WINDOW_BG).inner_margin(egui::Margin::same(4)))
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
                tracks(app, ui);
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
                    ui.add_enabled(
                        false,
                        egui::Button::new(i18n::t("Reset Trim")).min_size(egui::vec2(72.0, 18.0)),
                    );
                    if ui
                        .add(egui::Button::new(i18n::t("Frame All")).min_size(egui::vec2(64.0, 18.0)))
                        .clicked()
                    {
                        app.orbit.framed = false;
                    }
                    ui.add_enabled(
                        false,
                        egui::Button::new(i18n::t("Reset Locks")).min_size(egui::vec2(76.0, 18.0)),
                    )
                    .on_disabled_hover_text(i18n::t("Contact locks are not part of this build"));
                });
            });
        });
}

fn frame_ruler(app: &mut VeloMocap, ui: &mut egui::Ui) {
    let frames = app.frame_count();
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
            painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(0x24, 0x24, 0x24));
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
            painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(0x1c, 0x1c, 0x1c));
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
                egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(0xe8, 0xe8, 0xe8)),
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
            egui::RichText::new(i18n::t("Locks edited — re-run to apply them"))
                .size(9.5)
                .color(MUTED),
        );
    }
}

fn sidebar(app: &mut VeloMocap, ctx: &egui::Context) {
    egui::SidePanel::right("sidebar")
        .exact_width(SIDEBAR_WIDTH)
        .resizable(false)
        .frame(egui::Frame::new().fill(WINDOW_BG).inner_margin(egui::Margin::same(4)))
        .show(ctx, |ui| {
            panel(ui, |ui| {
                header(ui, i18n::t("AI Model"), |ui| {
                    if ui
                        .add(egui::Button::new(i18n::t("? Model Guide")).min_size(egui::vec2(96.0, 18.0)))
                        .clicked()
                    {
                        app.show_guide = true;
                    }
                });
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        model_group(app, ui);
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
        .stroke(egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(0x20, 0x20, 0x20)))
        .inner_margin(egui::Margin::symmetric(7, 6));
    frame.outer_margin = egui::Margin::symmetric(0, 5);
    frame.show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(
            egui::RichText::new(title)
                .size(11.5)
                .strong()
                .color(egui::Color32::from_rgb(0xc5, 0xc5, 0xc5)),
        );
        ui.add_space(3.0);
        body(ui)
    })
    .inner
}

fn model_group(app: &mut VeloMocap, ui: &mut egui::Ui) {
    group(ui, i18n::t("Solver"), |ui| {
        // The reference lets the user choose a motion backend (ExMR / GVHMR /
        // PEAR / …). This build has exactly one — FlowHMR with SAM 3D Body — so
        // the row is a label, not a combo box with a single dead entry.
        ui.label(egui::RichText::new(i18n::t("SAM 3D Body · MHR")).size(11.0))
            .on_hover_text(i18n::t(
                "The only reconstruction backend in this build; ExMR, GVHMR, PEAR, MAMMA, PromptHMR and GEM-X are not ported",
            ));
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(i18n::t("Graphs")).size(11.0));
            if ui
                .add(egui::Button::new("…").min_size(egui::vec2(22.0, 18.0)))
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new().pick_folder() {
                    app.sam3d_dir = path;
                }
            }
        });
        ui.add(
            egui::Label::new(
                egui::RichText::new(
                    app.sam3d_dir
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| i18n::t("(none)").into()),
                )
                .size(10.5)
                .color(MUTED),
            )
            .truncate(),
        );
        ui.label(
            egui::RichText::new(match &app.mesh {
                Some(mesh) => {
                    i18n::mesh_line(mesh.vertices, mesh.triangles.len(), mesh.keypoints)
                }
                None => i18n::t("Mesh not loaded yet").to_string(),
            })
            .size(10.5)
            .color(MUTED),
        );
        if let Some(faces) = &app.smplh_faces {
            // What the viewport actually draws once the motion stage has run.
            let vertices = app
                .flow
                .first()
                .and_then(|clip| clip.frames.first())
                .map(|verts| verts.len() / 3)
                .unwrap_or(0);
            ui.label(
                egui::RichText::new(i18n::smplh_line(vertices, faces.len()))
                    .size(10.5)
                    .color(MUTED),
            );
        }
    });
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
        if ui
            .add(egui::Button::new(i18n::t("Model Manager")).min_size(egui::vec2(ui.available_width(), 20.0)))
            .clicked()
        {
            app.show_models = true;
            app.refresh_models(false);
        }
        if !app.model_core_ready {
            ui.label(
                egui::RichText::new(i18n::t("Some required model files are missing."))
                    .size(10.0)
                    .color(egui::Color32::from_rgb(0xe8, 0xa8, 0x60)),
            );
        }
    });
}

fn status_bar(app: &mut VeloMocap, ctx: &egui::Context) {
    egui::TopBottomPanel::bottom("status")
        .exact_height(STATUS_HEIGHT)
        .frame(
            egui::Frame::new()
                .fill(HEADER_BG)
                .inner_margin(egui::Margin::symmetric(8, 3)),
        )
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                let colour = match app.capture.as_ref().map(|j| {
                    j.shared.lock().unwrap().progress.stage
                }) {
                    Some(CaptureStage::Failed) => ERR,
                    Some(CaptureStage::Done) => OK,
                    _ => egui::Color32::from_rgb(0xb8, 0xb8, 0xb8),
                };
                if let Some(error) = &app.error {
                    ui.label(egui::RichText::new(error).size(11.0).color(ERR));
                } else {
                    ui.label(egui::RichText::new(&app.status).size(11.0).color(colour));
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add(egui::Button::new(i18n::t("Console")).min_size(egui::vec2(72.0, 18.0)))
                        .clicked()
                    {
                        app.show_console = !app.show_console;
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

/// The export window: format, skeleton naming, rest correction, file name.
///
/// The naming and the rest-correction toggle used to be in the File menu, which
/// made them look like properties of the open document. They are choices about
/// the file being written, so they live with the write: one window that shows
/// exactly which files a click will produce.
fn export_window(app: &mut VeloMocap, ctx: &egui::Context) {
    if !app.show_export {
        return;
    }
    let mut open = app.show_export;
    egui::Window::new(i18n::t("Export motion"))
        .open(&mut open)
        .default_width(440.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(i18n::t("Format")).size(11.0).color(MUTED));
                for format in [ExportFormat::Bvh, ExportFormat::Fbx, ExportFormat::Both] {
                    ui.selectable_value(&mut app.export_format, format, format.label());
                }
            });
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(i18n::t("Skeleton naming")).size(11.0).color(MUTED));
                egui::ComboBox::from_id_salt("export-rig")
                    .selected_text(i18n::t(app.export_rig.label()))
                    .width(206.0)
                    .show_ui(ui, |ui| {
                        for rig in velomocap_motion::retarget::Rig::ALL {
                            if ui
                                .selectable_label(app.export_rig == rig, i18n::t(rig.label()))
                                .clicked()
                            {
                                app.export_rig = rig;
                                // Rigs that bind in the canonical T-pose want
                                // the correction; the ones that do not (Unreal's
                                // mannequin is an A-pose) do not.
                                app.export_rest_correct =
                                    rig != velomocap_motion::retarget::Rig::Biped
                                        && rig.binds_in_t_pose();
                            }
                        }
                    });
            })
            .response
            .on_hover_text(i18n::t("Write the bones with this rig's names"));
            ui.add_enabled(
                app.export_format.bvh(),
                egui::Checkbox::new(
                    &mut app.export_rest_correct,
                    egui::RichText::new(i18n::t("Rest correction")).size(11.0),
                ),
            )
            .on_disabled_hover_text(i18n::t(
                "An FBX carries its own bind pose, so the correction applies to BVH exports",
            ))
            .on_hover_text(i18n::t(
                "Turn the clip into the canonical T-pose bind, so a Mixamo-class rig plays it exactly",
            ));
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(i18n::t("File name")).size(11.0).color(MUTED));
                let hint = app
                    .video
                    .as_ref()
                    .and_then(|video| video.file_stem())
                    .map(|stem| stem.to_string_lossy().into_owned())
                    .unwrap_or_default();
                ui.add(
                    egui::TextEdit::singleline(&mut app.export_name)
                        .hint_text(hint)
                        .desired_width(240.0),
                );
            });
            let count = if app.flow.is_empty() {
                app.people.len()
            } else {
                app.flow.len()
            };
            if count > 1 {
                ui.checkbox(
                    &mut app.export_everyone,
                    egui::RichText::new(i18n::zh_en(
                        &format!("全部 {count} 人各写一份"),
                        &format!("One file per person ({count})"),
                    ))
                    .size(11.0),
                );
            }
            ui.add_space(4.0);
            // Show the exact paths, so a multi-person export cannot silently
            // overwrite the previous person's file.
            let indices: Vec<usize> = if app.export_everyone && count > 1 {
                (0..count).collect()
            } else {
                vec![app.selected.min(count.saturating_sub(1))]
            };
            for index in indices {
                for (enabled, ext) in [
                    (app.export_format.bvh(), "bvh"),
                    (app.export_format.fbx(), "fbx"),
                ] {
                    if !enabled {
                        continue;
                    }
                    let path = app
                        .export_path(ext, index)
                        .map(|path| path.display().to_string())
                        .unwrap_or_else(|| i18n::t("Open a video first").to_string());
                    ui.label(egui::RichText::new(path).size(10.0).color(MUTED));
                }
            }
            if app.export_format.fbx() && !app.export_rig.ascii_names() {
                ui.label(
                    egui::RichText::new(i18n::t(
                        "This rig's bone names are not ASCII: the FBX writer cannot hold them, write BVH instead",
                    ))
                    .size(10.0)
                    .color(ERR),
                );
            }
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let can = !app.flow.is_empty() || !app.people.is_empty();
                if ui
                    .add_enabled(can, egui::Button::new(i18n::t("Export")))
                    .clicked()
                {
                    app.export_now();
                }
                if ui.button(i18n::t("Cancel")).clicked() {
                    app.show_export = false;
                }
            });
        });
    // `export_now` closes the window itself, so keep it closed even though the
    // local `open` is still true.
    app.show_export = open && app.show_export;
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
                    "1. 文件 → 打开视频，或在命令行里直接给出片段。",
                    "2. 文件 → 重建所有人物：YOLOv8 找出每个人，再对每人运行",
                    "   SAM 3D Body 的 ONNX 链路。",
                    "3. 重建过程中人体会实时出现在 3D 视口；每个人都是完整的",
                    "   MHR 网格，而不只是骨骼。",
                    "4. 拖动时间轴或按空格回放结果。",
                ]
            } else {
                &[
                    "1. File → Open video, or pass a clip on the command line.",
                    "2. File → Reconstruct all people detects every person with YOLOv8",
                    "   and runs the SAM 3D Body ONNX chain on each of them.",
                    "3. Bodies stream into the viewport while the run is going; each",
                    "   person gets an MHR mesh, not just a skeleton.",
                    "4. Scrub the timeline or press Space to play the result back.",
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
    format!("VeloMocap | {}", i18n::t("Motion Workspace   |").trim())
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

#[cfg(test)]
mod tests {
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
