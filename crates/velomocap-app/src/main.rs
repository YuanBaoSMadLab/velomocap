//! VeloMocap native application.
//!
//! A zero-Python desktop pipeline: ffmpeg decodes the video, ONNX Runtime runs
//! the person detector and eleven SAM 3D Body graphs on the GPU, and the UI
//! draws every reconstructed MHR body in its own software-rasterised viewport.
//! No Qt, no PyTorch, no Python runtime.

// No console window next to the GUI. When the binary is started from a
// terminal the console is inherited anyway, so the `eprintln!` diagnostics
// still land there; a double-click gets just the window.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod capture;
mod i18n;
mod models_ui;
mod probe;
mod render;
mod stream;
mod ui;
mod viewport;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 820.0])
            .with_min_inner_size([1080.0, 660.0])
            .with_title(ui::window_title()),
        ..Default::default()
    };
    eframe::run_native(
        &ui::window_title(),
        options,
        Box::new(|cc| Ok(Box::new(app::VeloMocap::new(cc)))),
    )
}
