//! Windows DLL loading for ONNX Runtime.
//!
//! ONNX Runtime's CUDA build loads `onnxruntime_providers_cuda.dll` and the
//! `cudnn64_9.dll` family *by base name* at session-creation time. Those files
//! live next to `onnxruntime.dll`, which is often not on `PATH` when the app is
//! started from Explorer or a shortcut — the result is a silent CPU fallback or
//! a "cuDNN is unavailable" error.
//!
//! Two rules learned the hard way:
//!
//! 1. **Do not call `SetDefaultDllDirectories`.** It replaces the default
//!    search order and removes `PATH`, which makes ONNX Runtime fail to find
//!    its own provider DLLs (it then runs on CPU without reporting an error).
//! 2. **Preload by absolute path instead.** Mapping each dependency into the
//!    process means later `LoadLibrary("cudnn64_9.dll")` calls resolve to the
//!    already-loaded module, whatever the search order is.

use std::path::Path;

/// Prepare `folder` (the directory holding `onnxruntime.dll`) for loading.
pub fn prepare_dll_directory(library: &Path) {
    if let Some(folder) = library.parent() {
        prepare_dll_directory_path(folder);
    }
}

/// Preload ONNX Runtime's sibling DLLs and make `folder` resolvable.
pub fn prepare_dll_directory_path(folder: &Path) {
    #[cfg(windows)]
    {
        windows_impl::preload(folder);

        // Keep PATH in sync for child processes and for code that uses the
        // plain loader; `AddDllDirectory` is deliberately not used because it
        // changes global resolution semantics.
        if let Some(path) = std::env::var_os("PATH") {
            let mut paths: Vec<std::path::PathBuf> = vec![folder.to_path_buf()];
            paths.extend(std::env::split_paths(&path));
            if let Ok(joined) = std::env::join_paths(paths) {
                std::env::set_var("PATH", joined);
            }
        }
    }

    #[cfg(not(windows))]
    {
        let _ = folder;
    }
}

#[cfg(windows)]
mod windows_impl {
    use std::path::Path;

    /// DLLs ONNX Runtime's CUDA provider needs, in dependency order.
    const DEPENDENCIES: [&str; 12] = [
        "cudnn_engines_tensor_ir64_9.dll",
        "cudnn_ops64_9.dll",
        "cudnn_graph64_9.dll",
        "cudnn_heuristic64_9.dll",
        "cudnn_engines_precompiled64_9.dll",
        "cudnn_engines_runtime_compiled64_9.dll",
        "cudnn_cnn64_9.dll",
        "cudnn_adv64_9.dll",
        "cudnn_ext64_9.dll",
        "cudnn64_9.dll",
        "onnxruntime_providers_shared.dll",
        "onnxruntime_providers_cuda.dll",
    ];

    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryExW(
            name: *const u16,
            file: *mut core::ffi::c_void,
            flags: u32,
        ) -> *mut core::ffi::c_void;
    }

    /// `LOAD_WITH_ALTERED_SEARCH_PATH`: resolve the DLL's own dependencies from
    /// its folder. This is what makes a self-contained CUDA runtime work
    /// without touching `PATH`.
    const LOAD_WITH_ALTERED_SEARCH_PATH: u32 = 0x0000_0008;

    fn wide(path: &Path) -> Vec<u16> {
        use std::os::windows::ffi::OsStrExt;
        path.as_os_str().encode_wide().chain(std::iter::once(0)).collect()
    }

    pub fn preload(folder: &Path) {
        for name in DEPENDENCIES {
            let candidate = folder.join(name);
            if !candidate.is_file() {
                continue;
            }
            let path = wide(&candidate);
            unsafe {
                // A failure here is not fatal: ORT may still load the DLL
                // through its own search, and the CPU provider always works.
                LoadLibraryExW(path.as_ptr(), std::ptr::null_mut(), LOAD_WITH_ALTERED_SEARCH_PATH);
            }
        }
    }
}
