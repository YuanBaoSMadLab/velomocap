//! C ABI surface consumed by the Python bridge (`python/velomocap_bridge.py`).
//!
//! Design: small JSON descriptors carry names/parents/settings, while every
//! bulk array travels as a borrowed NumPy buffer.  No structs cross the
//! boundary, so the ABI stays stable while the Rust internals evolve.
//!
//! Ownership: functions returning `*mut c_char` hand back a heap string that
//! the caller must release with [`vm_free_string`].  The `*const c_char`
//! returned by `vm_version`, `vm_backend` and `vm_last_error` are borrowed and
//! must never be freed.

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::path::PathBuf;

use serde::Deserialize;
use serde_json::json;

use velomocap_core::bvh;
use velomocap_core::contact::{self, Preset};
use velomocap_core::fbx::{self, FbxInput, IDENTITY4};
use velomocap_core::rig::Binding;
use velomocap_core::{Error, Result};

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

fn set_error(message: impl Into<String>) -> *mut c_char {
    let message = message.into();
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = CString::new(message.replace('\0', " ")).unwrap_or_default();
    });
    std::ptr::null_mut()
}

fn to_c_string(value: &str) -> *mut c_char {
    match CString::new(value.replace('\0', " ")) {
        Ok(text) => text.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

unsafe fn spec<'a, T: Deserialize<'a>>(ptr: *const c_char) -> Result<T> {
    if ptr.is_null() {
        return Err(Error::new("Missing request descriptor"));
    }
    let text = CStr::from_ptr(ptr)
        .to_str()
        .map_err(|_| Error::new("Request descriptor is not valid UTF-8"))?;
    serde_json::from_str(text).map_err(|e| Error::new(format!("Invalid request descriptor: {e}")))
}

unsafe fn slice<'a>(ptr: *const f64, len: usize) -> Result<&'a [f64]> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(Error::new("Missing array buffer"));
    }
    Ok(std::slice::from_raw_parts(ptr, len))
}

#[derive(Deserialize)]
struct BvhSpec {
    path: String,
    fps: f64,
    #[serde(default)]
    version: i32,
    names: Vec<String>,
    parents: Vec<i32>,
    rest: Vec<f64>,
    ends: Vec<f64>,
    basis: Vec<f64>,
    /// Optional per-bone channel orders that pin the Euler encoding.
    #[serde(default)]
    orders: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct FbxSpec {
    path: String,
    names: Vec<String>,
    parents: Vec<i32>,
    rest: Vec<f64>,
    rest_frames: Vec<f64>,
    display: Vec<f64>,
    tip_local: Vec<f64>,
    fps: f64,
    #[serde(default)]
    transform: Option<Vec<f64>>,
    #[serde(default)]
    prepend_tpose: bool,
    #[serde(default = "default_bones")]
    expected_bones: usize,
    #[serde(default = "default_rig_name")]
    rig_name: String,
    #[serde(default = "default_creator")]
    creator: String,
}

fn default_bones() -> usize {
    52
}

fn default_rig_name() -> String {
    "SourceAnimation".to_string()
}

fn default_creator() -> String {
    "VeloMocap".to_string()
}

#[derive(Deserialize)]
struct ContactSpec {
    fps: f64,
    #[serde(default)]
    floor: Option<f64>,
    #[serde(default = "default_preset")]
    preset: String,
    #[serde(default)]
    manual: Option<[Vec<[usize; 2]>; 2]>,
}

fn default_preset() -> String {
    "normal".to_string()
}

/// `"0.1.0"` — the native core version.
#[no_mangle]
pub extern "C" fn vm_version() -> *const c_char {
    static VERSION: std::sync::OnceLock<CString> = std::sync::OnceLock::new();
    VERSION
        .get_or_init(|| CString::new(velomocap_core::VERSION).unwrap())
        .as_ptr()
}

/// `"avx2"` or `"scalar"` — how the C++ kernels were compiled.
#[no_mangle]
pub extern "C" fn vm_backend() -> *const c_char {
    static BACKEND: std::sync::OnceLock<CString> = std::sync::OnceLock::new();
    BACKEND
        .get_or_init(|| CString::new(velomocap_core::backend()).unwrap())
        .as_ptr()
}

/// Message for the most recent failed call on this thread (empty on success).
#[no_mangle]
pub extern "C" fn vm_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}

/// Release a string returned by any `vm_*` function.
///
/// # Safety
/// `ptr` must be a pointer previously returned by this library, or null.
#[no_mangle]
pub unsafe extern "C" fn vm_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}

/// Write a BVH take. Returns the report JSON, or null on failure.
///
/// # Safety
/// `roots` must hold `frames * 3` and `local` `frames * bones * 9` doubles.
#[no_mangle]
pub unsafe extern "C" fn vm_write_bvh(
    descriptor: *const c_char,
    roots: *const f64,
    local: *const f64,
    frames: usize,
    bones: usize,
) -> *mut c_char {
    let result = (|| -> Result<String> {
        let spec: BvhSpec = spec(descriptor)?;
        let binding = Binding {
            names: spec.names,
            parents: spec.parents,
            rest: spec.rest,
            ends: spec.ends,
            basis: spec.basis,
            origin: [0.0; 3],
            version: spec.version,
        };
        if binding.bones() != bones {
            return Err(Error::new(format!(
                "Bone count mismatch: descriptor {} vs buffers {}",
                binding.bones(),
                bones
            )));
        }
        let roots = slice(roots, frames * 3)?;
        let local = slice(local, frames * bones * 9)?;
        let report = bvh::write_bvh_with_orders(
            &PathBuf::from(&spec.path),
            &binding,
            roots,
            local,
            spec.fps,
            spec.version,
            spec.orders.as_deref(),
        )?;
        Ok(json!({
            "orders": report.orders,
            "minimum_singularity_margin": report.minimum_singularity_margin,
            "frames": frames,
            "bones": bones,
        })
        .to_string())
    })();
    match result {
        Ok(text) => to_c_string(&text),
        Err(error) => set_error(error.to_string()),
    }
}

/// Write a binary FBX take. Returns the report JSON, or null on failure.
///
/// # Safety
/// `frames_matrices` must hold `frames * bones * 16` doubles.
#[no_mangle]
pub unsafe extern "C" fn vm_write_fbx(
    descriptor: *const c_char,
    frames_matrices: *const f64,
    frames: usize,
    bones: usize,
) -> *mut c_char {
    let result = (|| -> Result<String> {
        let spec: FbxSpec = spec(descriptor)?;
        if spec.expected_bones != bones {
            return Err(Error::new(format!(
                "Bone count mismatch: descriptor {} vs buffers {}",
                spec.expected_bones, bones
            )));
        }
        let matrices = slice(frames_matrices, frames * bones * 16)?;
        let transform = match spec.transform {
            Some(values) => {
                if values.len() != 16 {
                    return Err(Error::new("transform must have 16 values"));
                }
                let mut out = [0.0f64; 16];
                out.copy_from_slice(&values);
                out
            }
            None => IDENTITY4,
        };
        let input = FbxInput {
            names: &spec.names,
            parents: &spec.parents,
            rest: &spec.rest,
            rest_frames: &spec.rest_frames,
            display: &spec.display,
            tip_local: &spec.tip_local,
            frames: matrices,
            fps: spec.fps,
            transform,
            prepend_tpose: spec.prepend_tpose,
            expected_bones: spec.expected_bones,
            rig_name: &spec.rig_name,
            creator: &spec.creator,
        };
        let report = fbx::write(&PathBuf::from(&spec.path), &input)?;
        Ok(json!({
            "key_count": report.key_count,
            "stop_tick": report.stop_tick,
            "bones": report.bones,
            "bytes": std::fs::metadata(&spec.path).map(|m| m.len()).unwrap_or(0),
        })
        .to_string())
    })();
    match result {
        Ok(text) => to_c_string(&text),
        Err(error) => set_error(error.to_string()),
    }
}

/// Plan foot contacts and grounding.
///
/// Optional output buffers (pass null to skip): `active_out` (`frames*2` u8),
/// `anchors_out` (`frames*2*3` f64), `offsets_out` (`frames*2*3` f64) and
/// `shift_out` (`frames*3` f64). Returns the plan report JSON, or null.
///
/// # Safety
/// `probes` must hold `frames*4*3` doubles and `probabilities` `frames*4`.
#[no_mangle]
pub unsafe extern "C" fn vm_prepare_contacts(
    descriptor: *const c_char,
    probes: *const f64,
    probabilities: *const f64,
    frames: usize,
    active_out: *mut u8,
    anchors_out: *mut f64,
    offsets_out: *mut f64,
    shift_out: *mut f64,
) -> *mut c_char {
    let result = (|| -> Result<String> {
        let spec: ContactSpec = spec(descriptor)?;
        let preset = Preset::named(&spec.preset)?;
        let probes = slice(probes, frames * 4 * 3)?;
        let probabilities = slice(probabilities, frames * 4)?;
        let manual = match spec.manual {
            Some(ranges) => {
                let owned: [Vec<(usize, usize)>; 2] = [
                    ranges[0].iter().map(|r| (r[0], r[1])).collect(),
                    ranges[1].iter().map(|r| (r[0], r[1])).collect(),
                ];
                Some(contact::normalize_ranges(&owned, frames)?)
            }
            None => None,
        };
        let plan = contact::prepare_contacts(
            probes,
            probabilities,
            frames,
            spec.fps,
            spec.floor,
            &preset,
            manual.as_ref(),
        )?;
        if !active_out.is_null() {
            for (i, value) in plan.active.iter().enumerate() {
                *active_out.add(i) = u8::from(*value);
            }
        }
        if !anchors_out.is_null() {
            std::ptr::copy_nonoverlapping(plan.anchors.as_ptr(), anchors_out, plan.anchors.len());
        }
        if !offsets_out.is_null() {
            std::ptr::copy_nonoverlapping(plan.offsets.as_ptr(), offsets_out, plan.offsets.len());
        }
        if !shift_out.is_null() {
            std::ptr::copy_nonoverlapping(plan.shift.as_ptr(), shift_out, plan.shift.len());
        }
        let segments: Vec<serde_json::Value> = plan
            .segments
            .iter()
            .map(|s| {
                json!({
                    "foot": s.foot,
                    "start": s.start,
                    "end": s.end,
                    "anchor": s.anchor,
                    "anchor_end": s.anchor_end,
                    "handoff": s.handoff,
                })
            })
            .collect();
        Ok(json!({
            "floor": plan.floor,
            "frames": plan.frames,
            "active_count": plan.active.iter().filter(|v| **v).count(),
            "segments": segments,
            "max_vertical_shift": plan.shift.chunks_exact(3).map(|c| c[1].abs()).fold(0.0f64, f64::max),
        })
        .to_string())
    })();
    match result {
        Ok(text) => to_c_string(&text),
        Err(error) => set_error(error.to_string()),
    }
}

/// Transition offsets that carry a stance correction into neighbouring swing.
///
/// `offsets_out` must hold `frames*2*3` and `transition_out` `frames*2` bytes.
///
/// # Safety
/// All pointers must be valid for the lengths documented above.
#[no_mangle]
pub unsafe extern "C" fn vm_contact_offsets(
    descriptor: *const c_char,
    probes: *const f64,
    active: *const u8,
    anchors: *const f64,
    support: *const f64,
    shift: *const f64,
    strength: *const f64,
    flight: *const u8,
    frames: usize,
    offsets_out: *mut f64,
    transition_out: *mut u8,
) -> *mut c_char {
    let result = (|| -> Result<String> {
        let spec: ContactSpec = spec(descriptor)?;
        let probes = slice(probes, frames * 4 * 3)?;
        let anchors = slice(anchors, frames * 2 * 3)?;
        let support = slice(support, frames * 2 * 3)?;
        let shift = slice(shift, frames * 3)?;
        let strength = slice(strength, frames * 2)?;
        if active.is_null() || flight.is_null() || offsets_out.is_null() || transition_out.is_null()
        {
            return Err(Error::new("Missing contact offset buffers"));
        }
        let active_flags: Vec<bool> = (0..frames * 2).map(|i| *active.add(i) != 0).collect();
        let flight_flags: Vec<bool> = (0..frames).map(|i| *flight.add(i) != 0).collect();
        let (offsets, transition) = contact::contact_offsets(
            probes,
            &active_flags,
            anchors,
            support,
            shift,
            frames,
            spec.fps,
            strength,
            &flight_flags,
        );
        std::ptr::copy_nonoverlapping(offsets.as_ptr(), offsets_out, offsets.len());
        for (i, value) in transition.iter().enumerate() {
            *transition_out.add(i) = u8::from(*value);
        }
        Ok(json!({
            "transition_frames": transition.iter().filter(|v| **v).count(),
            "frames": frames,
        })
        .to_string())
    })();
    match result {
        Ok(text) => to_c_string(&text),
        Err(error) => set_error(error.to_string()),
    }
}
