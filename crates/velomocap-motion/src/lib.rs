//! Motion: multi-person SAM 3D Body reconstruction, plus the legacy SMPL-X
//! fitting path and the BVH exporter.

pub mod anatomical;
pub mod arm_plan;
pub mod bvh_people;
pub mod cleanup;
pub mod contacts;
pub mod correction;
pub mod export;
pub mod fit;
pub mod floor;
pub mod flow;
pub mod ik;
pub mod locks;
pub mod palm;
pub mod pipeline;
pub mod placement;
pub mod pose_plan;
pub mod root_height;
pub mod retarget;
pub mod sam;
pub mod sole;
pub mod solver;
pub mod temporal;
pub mod trajectory;

pub use bvh_people::write_person_bvh;
pub use cleanup::{stabilise, CleanupConfig, CleanupReport};
pub use export::{binding_from_rest, write_bvh, BvhExport};
pub use flow::{
    feature_rows, generate as generate_flow, prepare_bvh, prepare_bvh_for, summarise as summarise_flow,
    write_clip_bvh, write_prepared, write_prepared_fbx, FlowClip, FlowConfig, PreparedBvh, ANCHOR_FRAMES, FEATURE_DIM,
    INPUT_DIM, NARROWBAND, VTXT_DIM,
};
pub use pipeline::{
    capture, capture_with_hooks, capture_with_progress, capture_with_sink, CaptureConfig,
    CaptureResult, FitSink, FrameSink, Progress, Timings,
};
pub use fit::{fit_sequence, summarise, FitConfig, FittedFrame, Fitter};
pub use locks::{ContactTracks, Interval, LOCK_VERSION, TRACK_COUNT, TRACK_LABELS};
pub use sam::{
    capture_people, capture_people_with, PeopleCapture, PersonFrame, PersonTrack, SamConfig,
    SamHooks,
};

/// Device-memory housekeeping, re-exported so callers do not have to depend on
/// the C++ crate just to trim the CUDA cache between stages.
pub mod vram {
    pub use velomocap_cpp::sam3d::{cuda_memory, cuda_memory_line, empty_cuda_cache};
}
