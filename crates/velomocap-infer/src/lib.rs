//! ONNX Runtime session management for VeloMocap.
//!
//! The native pipeline loads ONNX Runtime dynamically (`ort`'s `load-dynamic`
//! feature), so the application binary has no link-time dependency on it and
//! the same build works with any CUDA 12 or CUDA 13 runtime package. The
//! library is located at runtime in this order:
//!
//! 1. `$VELOMOCAP_ORT_DYLIB`
//! 2. `runtime/onnxruntime/**/lib/onnxruntime.dll` next to the executable
//! 3. a system-wide `onnxruntime.dll` on `PATH`
//!
//! Execution providers are selected per session: CUDA first when the provider
//! library is present, otherwise CPU. Falling back is always allowed — a CPU
//! run is slower, never wrong.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result};

pub mod camera;
pub mod detector;
pub mod mesh;
pub mod motion;
pub mod platform;
pub mod pose;
pub mod sam3d;
pub mod sam3d_params;
pub mod tracker;

pub use camera::{quat_to_mat, rope_table, CameraTrack, VggtCamera, VggtConfig};
pub use detector::{Detection, Detector, DetectorConfig};
pub use mesh::{KeypointRegressor, MeshTopology, JOINTS as MESH_JOINTS, VERTICES as MESH_VERTICES};
pub use motion::{
    matrix_to_axis_angle, rot6d_to_matrix, flow_rot6d_to_matrix, GeneratedMotion, MotionConfig,
    MotionGenerator, MotionStats, Narrowband,
};
pub use pose::{Keypoint, PoseConfig, PoseDetection, PoseEstimator, KEYPOINT_NAMES};
pub use sam3d::{default_intrinsics, Sam3dBody, Sam3dFrame, Sam3dInput, IMAGE_SIZE};
pub use sam3d_params::{DecodedParams, ParamsTables};
pub use tracker::{Track, Tracker};

/// Which execution provider a session actually received.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cuda,
    Cpu,
}

impl Backend {
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::Cuda => "cuda",
            Backend::Cpu => "cpu",
        }
    }
}

#[derive(Clone, Debug)]
pub struct SessionConfig {
    /// Prefer the CUDA execution provider.
    pub gpu: bool,
    /// CUDA device index used when `gpu` is set.
    pub device_id: i32,
    /// Intra-op threads for CPU execution (0 = ORT default).
    pub intra_threads: usize,
    /// ORT graph optimisation level name (`"all"`, `"extended"`, `"basic"`, `"disabled"`).
    pub optimization: String,
    /// Fail instead of falling back to CPU when CUDA is unavailable.
    pub require_gpu: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            gpu: true,
            device_id: 0,
            intra_threads: 0,
            optimization: "all".to_string(),
            require_gpu: false,
        }
    }
}

const LIBRARY_NAMES: [&str; 3] = ["onnxruntime.dll", "libonnxruntime.so", "libonnxruntime.dylib"];

/// Locate the ONNX Runtime shared library.
pub fn find_runtime_library() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("VELOMOCAP_ORT_DYLIB") {
        let path = PathBuf::from(explicit);
        if path.is_file() {
            return Some(path);
        }
    }

    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(folder) = exe.parent() {
            roots.push(folder.to_path_buf());
            roots.push(folder.join("runtime/onnxruntime"));
        }
    }
    roots.push(PathBuf::from("runtime/onnxruntime"));
    roots.push(PathBuf::from("../runtime/onnxruntime"));

    for root in roots {
        for name in LIBRARY_NAMES {
            if let Some(found) = search(&root, name, 4) {
                return Some(found);
            }
        }
    }

    for name in LIBRARY_NAMES {
        if let Some(found) = which(name) {
            return Some(found);
        }
    }
    None
}

fn search(root: &Path, name: &str, depth: usize) -> Option<PathBuf> {
    let direct = root.join(name);
    if direct.is_file() {
        return Some(direct);
    }
    let lib = root.join("lib").join(name);
    if lib.is_file() {
        return Some(lib);
    }
    if depth == 0 {
        return None;
    }
    for entry in std::fs::read_dir(root).ok()?.flatten() {
        if entry.file_type().ok()?.is_dir() {
            if let Some(found) = search(&entry.path(), name, depth - 1) {
                return Some(found);
            }
        }
    }
    None
}

fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

static RUNTIME: OnceLock<Result<PathBuf, String>> = OnceLock::new();

/// Load ONNX Runtime once per process and report the library used.
pub fn init_runtime() -> Result<PathBuf> {
    RUNTIME
        .get_or_init(|| {
            let library = find_runtime_library().ok_or_else(|| {
                "onnxruntime library not found; set VELOMOCAP_ORT_DYLIB".to_string()
            })?;
            std::env::set_var("ORT_DYLIB_PATH", &library);
            // Make cuDNN and the provider DLLs resolvable before ORT loads.
            platform::prepare_dll_directory(&library);
            Ok(library)
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

/// Whether the CUDA provider library ships next to ONNX Runtime.
pub fn cuda_provider_available() -> bool {
    let Some(library) = find_runtime_library() else {
        return false;
    };
    let Some(lib_dir) = library.parent() else {
        return false;
    };
    lib_dir.join("onnxruntime_providers_cuda.dll").is_file()
        || lib_dir.join("libonnxruntime_providers_cuda.so").is_file()
}

/// Build the ONNX Runtime environment (CUDA EP included when available).
///
/// `ort` allows exactly one environment per process, so this is called once
/// and the result cached.
static ENVIRONMENT: OnceLock<Result<&'static ort::environment::Environment, String>> =
    OnceLock::new();

fn environment(gpu: bool, device_id: i32) -> Result<&'static ort::environment::Environment> {
    ENVIRONMENT
        .get_or_init(|| {
            let library = find_runtime_library().ok_or_else(|| {
                "onnxruntime library not found; set VELOMOCAP_ORT_DYLIB".to_string()
            })?;
            std::env::set_var("ORT_DYLIB_PATH", &library);
            platform::prepare_dll_directory(&library);
            let mut builder = ort::init_from(&library)
                .map_err(|error| format!("loading {}: {error}", library.display()))?
                .with_name("velomocap");
            let use_cuda = gpu && cuda_provider_available();
            if std::env::var("VELOMOCAP_DEBUG_TIMING").is_ok() {
                eprintln!(
                    "environment: gpu={gpu} provider_available={} use_cuda={use_cuda}",
                    cuda_provider_available()
                );
            }
            if use_cuda {
                // TensorFloat-32 keeps the kernels roughly twice as fast and
                // costs about 1e-3 on the final token. It is on by default;
                // `VELOMOCAP_TF32=0` trades that speed back for full fp32, which
                // is what the reference parity numbers are measured with.
                let tf32 = std::env::var("VELOMOCAP_TF32")
                    .map(|value| value != "0")
                    .unwrap_or(true);
                builder = builder.with_execution_providers([ort::ep::CUDA::default()
                    .with_device_id(device_id)
                    .with_tf32(tf32)
                    .build()]);
            }
            let env = builder
                .build()
                .map_err(|error| format!("creating ORT environment: {error}"))?;
            Ok(Box::leak(Box::new(env)) as &'static ort::environment::Environment)
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

/// A loaded ONNX model plus the backend it was actually given.
pub struct Model {
    session: ort::session::Session,
    backend: Backend,
    path: PathBuf,
}

impl Model {
    pub fn load(path: impl AsRef<Path>, config: &SessionConfig) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if !path.is_file() {
            anyhow::bail!("model not found: {}", path.display());
        }

        let wants_cuda = config.gpu && cuda_provider_available();
        if config.require_gpu && !wants_cuda {
            anyhow::bail!("CUDA execution provider unavailable and require_gpu was set");
        }
        let backend = if wants_cuda { Backend::Cuda } else { Backend::Cpu };

        let env = environment(wants_cuda, config.device_id)?;
        let mut builder = ort::session::Session::builder(env)
            .map_err(|error| anyhow::anyhow!("creating ONNX session builder: {error}"))?;
        builder = builder
            .with_optimization_level(optimization_level(&config.optimization)?)
            .map_err(|error| anyhow::anyhow!("setting optimisation level: {error}"))?;
        if config.intra_threads > 0 {
            builder = builder
                .with_intra_threads(config.intra_threads)
                .map_err(|error| anyhow::anyhow!("setting intra-op threads: {error}"))?;
        }

        let session = builder
            .commit_from_file(&path)
            .with_context(|| format!("loading ONNX model {}", path.display()))?;

        Ok(Self {
            session,
            backend,
            path,
        })
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn session(&self) -> &ort::session::Session {
        &self.session
    }

    pub fn session_mut(&mut self) -> &mut ort::session::Session {
        &mut self.session
    }

    /// Human-readable input/output names, for diagnostics.
    pub fn describe(&self) -> String {
        let name = self
            .path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let mut out = format!("{name} [{}]", self.backend.as_str());

        for input in self.session.inputs() {
            out.push_str(&format!(" in:{}", input.name()));
        }
        for output in self.session.outputs() {
            out.push_str(&format!(" out:{}", output.name()));
        }
        out
    }
}

fn optimization_level(name: &str) -> Result<ort::session::builder::GraphOptimizationLevel> {
    use ort::session::builder::GraphOptimizationLevel;
    Ok(match name.to_ascii_lowercase().as_str() {
        "all" => GraphOptimizationLevel::Level3,
        "extended" => GraphOptimizationLevel::Level2,
        "basic" => GraphOptimizationLevel::Level1,
        "disabled" | "none" => GraphOptimizationLevel::Disable,
        other => anyhow::bail!("unknown optimisation level {other:?}"),
    })
}


