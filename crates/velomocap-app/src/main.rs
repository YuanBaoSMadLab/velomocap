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
mod asset;
mod capture;
mod character;
mod i18n;
mod probe;
mod project;
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
    let result = eframe::run_native(
        &ui::window_title(),
        options,
        Box::new(|cc| Ok(Box::new(app::VeloMocap::new(cc)))),
    );
    // Closing the window used to leave the process behind. The capture, camera
    // and export workers own CUDA and ONNX Runtime contexts, and tearing one of
    // those down can block a thread until the driver gives up; `run_native` then
    // returned while the process stayed alive - invisible, holding the GPU and
    // the executable's file lock, and looking like a crash to the next launch.
    // Nothing is written after the window closes, so leave here instead of
    // waiting for destructors that may never finish.
    //
    // The session marker goes first: it is what tells the next launch that this
    // one exited properly, so there is nothing to recover.
    project::end_session();
    match result {
        Ok(()) => std::process::exit(0),
        Err(error) => {
            eprintln!("VeloMocap: {error}");
            std::process::exit(1);
        }
    }
}
