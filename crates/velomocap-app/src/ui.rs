//! Widgets and layout for the VeloMocap window.
//!
//! This mirrors the original RGB MoCap Qt workspace: the same panel order, the
//! same headers and footer strips, the same names and the same dark palette
//! (`mocap/theme.py`), so the two applications read as one product.

use eframe::egui;

use crate::app::VeloMocap;
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
    visuals.selection.stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);
    visuals.window_stroke = egui::Stroke::new(1.0, BORDER);
    visuals.widgets.noninteractive.bg_fill = PANEL_BG;
    visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, BORDER);
    visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(0x45, 0x45, 0x45);
    visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(0x24, 0x24, 0x24));
    visuals.widgets.inactive.corner_radius = 3.into();
    visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(0x55, 0x55, 0x55);
    visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(0x73, 0x73, 0x73));
    visuals.widgets.hovered.corner_radius = 3.into();
    visuals.widgets.active.bg_fill = egui::Color32::from_rgb(0x33, 0x33, 0x33);
    visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, ACCENT_DIM);
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
                    // The FlowHMR clip is what the viewport shows, so the menu
                    // exports that when it exists and the per-frame MHR bodies
                    // otherwise.
                    let can_export = !app.flow.is_empty() || !app.people.is_empty();
                    if ui
                        .add_enabled(can_export, egui::Button::new(i18n::t("Export BVH…")))
                        .on_hover_text(i18n::t(
                            "Write the selected person's motion next to the video",
                        ))
                        .clicked()
                    {
                        ui.close();
                        app.export_bvh();
                    }
                    if ui
                        .add_enabled(
                            !app.flow.is_empty(),
                            egui::Button::new(i18n::t("Export FBX…")),
                        )
                        .on_hover_text(i18n::t(
                            "Write the selected person's motion as binary FBX 7.4",
                        ))
                        .clicked()
                    {
                        ui.close();
                        app.export_fbx();
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
        .stroke(egui::Stroke::new(1.0, BORDER))
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
                .stroke(egui::Stroke::new(1.0, if selected { ACCENT_DIM } else { BORDER }))
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
        .stroke(egui::Stroke::new(1.0, BORDER))
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
        egui::ComboBox::from_id_salt("person-take")
            .selected_text(i18n::t("Default"))
            .width(84.0)
            .show_ui(ui, |ui| {
                // One take for now; the list exists so the layout matches.
                let mut take = 0usize;
                ui.selectable_value(&mut take, 0, i18n::t("Default"));
            });
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
                    egui::Stroke::new(if selected { 2.6 } else { 1.4 }, person.colour),
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
                    egui::Stroke::new(if selected { 2.6 } else { 1.4 }, colour),
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
                .stroke(egui::Stroke::new(1.0, ACCENT_DIM))
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
                        egui::Stroke::new(1.4, egui::Color32::WHITE),
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
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(0x20, 0x20, 0x20)))
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
        egui::ComboBox::from_id_salt("model")
            .selected_text(i18n::t("SAM 3D Body · MHR"))
            .width(ui.available_width())
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut 0, 0, i18n::t("SAM 3D Body · MHR"));
            });
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
                ui.label(egui::RichText::new(i18n::t("Pelvis Offset")).size(11.0));
                ui.add(
                    egui::DragValue::new(&mut app.correction.pelvis_offset)
                        .speed(0.005)
                        .range(-1.0..=1.0)
                        .suffix(" m"),
                );
            });
        });
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

fn guide_window(app: &mut VeloMocap, ctx: &egui::Context) {
    if !app.show_guide {
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
