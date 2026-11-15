//! VeloMocap core: the native replacement for the Python motion pipeline's
//! hot paths.
//!
//! Layers:
//! - [`cpp`] re-exports the C++ kernel crate (SO(3), FK, skinning, filters).
//! - [`math`] and [`filters`] wrap those kernels with scipy/numpy semantics.
//! - [`rig`], [`bvh`], [`fbx`] and [`contact`] implement the motion pipeline
//!   stages that used to run as NumPy/SciPy Python code.

pub use velomocap_cpp as cpp;

pub mod bvh;
pub mod contact;
pub mod error;
pub mod fbx;
pub mod fbx_read;
pub mod filters;
pub mod math;
pub mod rig;

pub use error::{Error, Result};

/// Version reported by the native core and the Python bridge.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Acceleration backend reported by the compiled C++ kernels.
pub fn backend() -> &'static str {
    cpp::build_features()
}
