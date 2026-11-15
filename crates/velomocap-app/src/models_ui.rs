//! Model Manager: shows what is installed and how to obtain what is missing.
//!
//! Gated weights cannot be shipped, so this panel is the user-facing contract:
//! it names the file, the exact destination path, and the provider page to open.
//! Nothing is downloaded silently.

use eframe::egui;

use velomocap_models::{human_bytes, GroupStatus, ModelLibrary, State};

use crate::app::VeloMocap;

const OK: egui::Color32 = egui::Color32::from_rgb(110, 200, 140);
const WARN: egui::Color32 = egui::Color32::from_rgb(232, 168, 96);
const ERR: egui::Color32 = egui::Color32::from_rgb(235, 118, 118);
const DIM: egui::Color32 = egui::Color32::from_gray(140);

pub fn window(app: &mut VeloMocap, ctx: &egui::Context) {
    let mut open = app.show_models;
    egui::Window::new("Model Manager")
        .open(&mut open)
        .default_size([760.0, 560.0])
        .resizable(true)
        .show(ctx, |ui| {
            header(app, ui);
            ui.separator();
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let groups = app.model_status.clone();
                    for group in &groups {
                        group_section(app, ui, group);
                    }
                });
        });
    app.show_models = open;
}

fn header(app: &mut VeloMocap, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        let ready = app.model_core_ready;
        let (mark, text, colour) = if ready {
            ("Ready", "All required models installed", OK)
        } else {
            ("Incomplete", "Some required models are missing", ERR)
        };
        let (rect, _) = ui.allocate_exact_size(egui::vec2(9.0, 9.0), egui::Sense::hover());
        ui.painter().circle_filled(rect.center(), 3.5, colour);
        ui.label(egui::RichText::new(mark).strong().color(colour));
        ui.label(egui::RichText::new(text).color(DIM));

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("Re-check").clicked() {
                app.refresh_models(false);
            }
            if ui
                .button("Verify hashes")
                .on_hover_text("Reads every file and compares SHA-256; slow for large models")
                .clicked()
            {
                app.refresh_models(true);
            }
        });
    });
    if let Some(root) = &app.model_root {
        ui.label(
            egui::RichText::new(format!("Model folder: {}", root.display()))
                .size(10.0)
                .color(DIM),
        );
    }
}

fn group_section(_app: &mut VeloMocap, ui: &mut egui::Ui, group: &GroupStatus) {
    ui.add_space(8.0);
    ui.horizontal(|ui| {
        let colour = if group.ready() { OK } else if group.group.required { ERR } else { WARN };
        let (rect, _) = ui.allocate_exact_size(egui::vec2(9.0, 9.0), egui::Sense::hover());
        ui.painter().circle_filled(rect.center(), 3.5, colour);
        ui.label(egui::RichText::new(&group.group.title).strong().size(13.0));
        ui.label(
            egui::RichText::new(if group.group.required { "required" } else { "optional" })
                .size(10.0)
                .color(DIM),
        );
        ui.label(
            egui::RichText::new(format!(
                "{}/{}",
                group.installed_count(),
                group.entries.len()
            ))
            .size(10.0)
            .color(DIM),
        );
    });
    if !group.group.description.is_empty() {
        ui.label(egui::RichText::new(&group.group.description).size(11.0).color(DIM));
    }

    for entry in &group.entries {
        let colour = match entry.state {
            State::Ok => OK,
            State::Missing => {
                if entry.spec.optional {
                    WARN
                } else {
                    ERR
                }
            }
            _ => ERR,
        };
        ui.horizontal(|ui| {
            let (rect, _) = ui.allocate_exact_size(egui::vec2(7.0, 7.0), egui::Sense::hover());
            ui.painter().circle_filled(rect.center(), 2.8, colour);
            ui.label(egui::RichText::new(&entry.spec.title).size(12.0));

            if entry.spec.gated {
                ui.label(
                    egui::RichText::new("account")
                        .size(9.0)
                        .color(WARN)
                        .background_color(WARN.gamma_multiply(0.12)),
                );
            }
            if entry.spec.optional {
                ui.label(egui::RichText::new("optional").size(9.0).color(DIM));
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let size = if entry.actual_bytes > 0 {
                    human_bytes(entry.actual_bytes)
                } else if entry.spec.bytes > 0 {
                    human_bytes(entry.spec.bytes)
                } else {
                    String::new()
                };
                ui.label(egui::RichText::new(size).size(10.0).color(DIM));
                ui.label(
                    egui::RichText::new(entry.state.label())
                        .size(10.0)
                        .color(colour),
                );
            });
        });

        // Missing: show where it goes and where to get it.
        if !entry.state.is_ok() {
            ui.indent(entry.spec.key.as_str(), |ui| {
                ui.label(
                    egui::RichText::new(format!("place at: {}", entry.path.display()))
                        .size(10.0)
                        .color(DIM),
                );
                if let Some(account) = &entry.spec.account {
                    ui.label(egui::RichText::new(account).size(10.0).color(WARN));
                }
                ui.horizontal(|ui| {
                    if !entry.spec.url.is_empty() && ui.button("Copy download URL").clicked() {
                        ui.ctx().copy_text(entry.spec.url.clone());
                    }
                    if !entry.spec.url.is_empty() && ui.button("Open page").clicked() {
                        ui.ctx().open_url(egui::OpenUrl::new_tab(entry.spec.url.clone()));
                    }
                    if let Some(export) = &entry.spec.export {
                        if ui.button("Copy export command").clicked() {
                            ui.ctx().copy_text(export.clone());
                        }
                    }
                });
                if !entry.spec.url.is_empty() {
                    ui.label(
                        egui::RichText::new(&entry.spec.url)
                            .size(9.0)
                            .color(egui::Color32::from_gray(110)),
                    );
                }
            });
        }
    }
    ui.add_space(4.0);
}

/// Load the library and refresh cached status on the app.
pub fn refresh(app: &mut VeloMocap, verify_hash: bool) {
    match ModelLibrary::discover() {
        Ok(library) => {
            let status = library.status(verify_hash);
            app.model_core_ready = library.core_ready();
            app.model_root = Some(library.root.clone());
            app.model_status = status;
            app.model_error = None;
        }
        Err(error) => {
            app.model_status.clear();
            app.model_core_ready = false;
            app.model_error = Some(format!("{error:#}"));
        }
    }
}
