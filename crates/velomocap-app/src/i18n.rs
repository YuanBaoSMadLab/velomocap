//! Bilingual interface strings, Chinese by default.
//!
//! The original workspace is English-only. This build ships both languages and
//! starts in Chinese, as the reference workflow expects; `Help → 语言 / Language`
//! switches at runtime. Every visible string goes through [`t`], so a missing
//! translation falls back to the English source text instead of a blank label.

use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lang {
    Zh,
    En,
}

impl Lang {
    pub fn code(self) -> &'static str {
        match self {
            Lang::Zh => "zh",
            Lang::En => "en",
        }
    }

    /// Label used by the switcher, always showing the *other* language.
    pub fn other(self) -> Lang {
        match self {
            Lang::Zh => Lang::En,
            Lang::En => Lang::Zh,
        }
    }
}

static LANG: AtomicU8 = AtomicU8::new(0);

pub fn lang() -> Lang {
    match LANG.load(Ordering::Relaxed) {
        1 => Lang::En,
        _ => Lang::Zh,
    }
}

pub fn set_lang(value: Lang) {
    LANG.store(
        match value {
            Lang::Zh => 0,
            Lang::En => 1,
        },
        Ordering::Relaxed,
    );
}

/// Translate an interface string. English is the source language.
pub fn t(english: &'static str) -> &'static str {
    if lang() == Lang::En {
        return english;
    }
    match english {
        // Menus and window chrome.
        "File" => "文件",
        "Help" => "帮助",
        "Open video…" => "打开视频…",
        "Import video…" => "导入视频…",
        "Import character…" => "导入角色…",
        "Bone Overlay" => "骨骼叠加",
        "Bone overlay" => "骨骼叠加",
        "Characters" => "角色",
        // PMX (MMD) avatars load and animate, but cloth and hair that MMD moves
        // with rigid-body physics are not simulated yet, so the format is marked
        // experimental rather than hidden.
        "Experimental" => "实验性",
        "Drop to import" => "松开即导入",
        "as the motion clip" => "作为动捕视频",
        "as a character" => "作为角色模型",
        "as the whole project" => "作为工程",
        "Import character" => "导入角色",
        "No characters imported yet" => "还没有导入角色",
        "Import a VRM / glTF / FBX / PMX humanoid and it joins this list" => {
            "导入 VRM / glTF / FBX / PMX 人物，它会进到这个列表里"
        }
        "Use on this person" => "用于当前人物",
        "Remove from list" => "从列表移除",
        "Model" => "模型",
        "Bones matched" => "骨骼匹配",
        "Vertices" => "顶点数",
        "Person" => "人物",
        "Project storage" => "工程存储",
        "Precision" => "存储精度",
        "Compression" => "压缩算法",
        "Settings" => "设置",
        "Project storage" => "工程存储",
        "Storage precision note" => "精度越高越还原，越低越省空间；下次保存工程时生效",
        "Ultra (lossless)" => "最高（无损）",
        "High (default)" => "高（默认）",
        "Medium" => "中",
        "Low" => "低",
        "Fast" => "快",
        "Balanced" => "均衡",
        "Smallest" => "最小",
        "Storage note" => "存储说明",
        "Faces" => "面数",
        "Unmatched" => "未匹配",
        "Replace the selected person's body with this model" => {
            "用这个模型替换选中人物的身体"
        }
        "Resolution" => "分辨率",
        "Frame rate" => "帧率",
        "Frames" => "帧数",
        "Position" => "位置",
        "Stream" => "视频流",
        "Ultra (lossless)" => "超高（无损）",
        "High (default)" => "高（默认）",
        "Medium" => "中",
        "Low" => "低",
        "None" => "不压缩",
        "Fast" => "快速",
        "Balanced" => "均衡",
        "Smallest" => "最小",
        "Storage precision note" => {
            "几何按这个精度存盘；骨架（导出用）始终原样保存，导出结果不受影响"
        }
        "Open project…" => "打开工程…",
        "Save project…" => "保存工程…",
        "Open project" => "打开工程",
        "Save project" => "保存工程",
        "VeloMocap project" => "VeloMocap 工程",
        "Load a .vp project: the video, the people, the motion, the avatars and the locks" => {
            "读取 .vp 工程：视频、人物、动作、角色模型和锁定一起回来"
        }
        "Write the whole workspace to a .vp file, so reopening it does not run the reconstruction again" => {
            "把整个工作区写进一个 .vp 文件，重新打开时不用再跑一遍重建"
        }
        "Import a VRM / glTF humanoid and play the motion on it" => {
            "导入 VRM / glTF 人物，用它播放这段动作"
        }
        "Import video" => "导入视频",
        "Cancel" => "取消",
        "Detecting people…" => "正在检测人物…",
        "Person" => "人物",
        "Built-in body" => "内置身体（SMPL-H）",
        "Replace model…" => "替换模型…",
        "Use built-in body" => "恢复内置身体",
        "Click to open this person's model settings" => "点击打开这个人物的人物设置",
        "Put all four lock tracks back on automatic" => "四条锁定轨全部恢复自动",
        "No video open" => "还没有打开视频",
        "File → Import video… to start; the viewport fills once a clip is open." => {
            "文件 → 导入视频… 开始；打开片段后这里就会出现画面"
        }
        "No people yet" => "还没有人物",
        "File → Reconstruct all people runs the detector and the body model." => {
            "文件 → 重建所有人物：先检测人物，再跑身体模型"
        }
        "No motion yet" => "还没有动作",
        "The motion stage is still running, or it produced no clip." => {
            "动作阶段还在运行，或者这段视频没有生成动作"
        }
        "Import a VRM / glTF humanoid for this person" => "给这个人物导入 VRM / glTF / FBX / PMX 人物模型",
        "Reconstruct all people" => "重建所有人物",
        "Detect every person, then run SAM 3D Body on each" => "先检测出每个人，再逐人运行 SAM 3D Body",
        "Cancel reconstruction" => "取消重建",
        "Export character assets…" => "导出角色资产…",
        "Export character assets" => "导出角色资产",
        "Mesh, skeleton, materials and animation as .glb or .fbx" => {
            "网格、骨骼、材质和动画，写成 .glb 或 .fbx"
        }
        "Export" => "导出",
        "Close" => "关闭",
        "Format" => "格式",
        "Open a video first" => "请先打开一个视频",
        "VeloMocap biped" => "VeloMocap 双足",
        "DAZ Genesis" => "DAZ Genesis",
        // Contact locks, under the timeline.
        "L foot" => "左脚",
        "R foot" => "右脚",
        "L hand" => "左手",
        "R hand" => "右手",
        "Lock" => "锁定",
        "automatic" => "自动",
        "Add lock here" => "在此添加锁定",
        "Copy the automatic runs" => "把自动区间复制到本轨",
        "Delete lock" => "删除锁定",
        "Undo" => "撤销",
        "Redo" => "重做",
        "Reset this track to automatic" => "该轨恢复自动",
        "Reset all locks to automatic" => "全部恢复自动",
        "Locks edited — re-run to apply them" => "锁定已修改 — 重新生成后生效",
        "Reset" => "重置",
        "Stop" => "停止",
        "Estimated camera (VGGT)" => "估算相机（VGGT）",
        // Reconstruction stages, shown one at a time in the timeline.
        "loading models" => "加载模型",
        "reconstructing people" => "重建人物",
        "estimating camera" => "估算相机",
        "generating motion" => "生成动作",
        "skinning" => "蒙皮身体",
        "done" => "完成",
        "Overall progress" => "总进度",
        "Device memory after the last stage finished" => "上一阶段结束后的显存占用",
        "working…" => "处理中…",
        "Refine Camera Rotation" => "优化相机旋转",
        "Only used when the camera is estimated" => "仅在估算相机时生效",
        "Multi-level" => "多平面",
        "Tight" => "严格",
        "Loose" => "宽松",
        "Pelvis Offset" => "骨盆高度偏移（相对自动）",
        "Causal Foot Lock" => "因果脚锁",
        "Hold each planted foot with the reference's live lock: 75 ms of evidence to lock, no lookahead, and a smooth hand-over when the foot lifts" => "用参考实现的实时锁把着地脚逐帧钉住：需要 75 毫秒的着地证据才上锁、不看未来帧、抬脚时平滑过渡",
        "The reference's live-preview stabiliser, run on top of the exact contact solve: 75 ms of evidence to lock, no lookahead, and a smooth hand-over when the foot lifts. Off by default — the reference does not run it in batch, and measured on three takes it made every sole metric worse" => "参考实现的“实时预览”稳定器，叠在精确接触求解之上：需要 75 毫秒证据才上锁、不看未来帧、抬脚时平滑交接。默认关闭 —— 参考的批处理并不跑它，三台片子实测它让脚掌指标全面变差",
        "Apply Correction" => "应用校正",
        "Apply locks" => "应用锁定",
        "Applying…" => "正在应用…",
        "Re-solve the contacts with the timeline's locks, without running the model again" => "用时间轴上的锁定重新求解接触，不重跑模型",
        "Nothing to apply: no lock is drawn, or this take has no solver input (a project saved before the locks could be re-solved)" => "没有可应用的东西：没画锁定，或者这条 take 没有求解输入（旧版工程保存的，锁定还无法重解）",
        "Applying correction…" => "正在应用校正…",
        "Correction applied in" => "校正完成，用时",
        "Correction failed" => "校正失败",
        "Re-solve the contacts with these settings and the timeline's locks, without running the model again" => "用当前设置和时间轴上的锁定重新求解接触，不重跑模型",
        "Generate a motion first: a take loaded from a project has no saved solver input" => "先生成动作：从工程打开的 take 没有保存求解器输入",
        "Generate a motion first: a take loaded from a project saved before this build has no solver input" => "先生成动作：这个工程是旧版本保存的，没有求解器输入（新保存的工程会带上）",
        "Locks edited — apply to update the motion" => "锁定已改 —— 停手后自动重解（也可点“应用锁定”）",
        "Settings changed — apply to update the motion" => "设置已改 —— 点“应用校正”生效",
        "Offset from the automatic pelvis height; positive gives back the reach drop, negative lowers the body" => "相对自动骨盆高度的偏移；正值只把自动压低的部分还回来，负值压低身体",
        "Pin planted feet and resting hands after the motion is generated" => "动作生成后锁定着地的脚和静止的手",
        "Reconstruct this person when the run starts" => "开始重建时包含此人",
        "Select all" => "全选",
        "Select none" => "全不选",
        "Select this person: the box over the clip is drawn in this colour" => "选中此人：视频上的框会用这个颜色标出",
        "One take per video in this build; the reference's multi-take scene is not implemented" => "此版本每个视频一个 take；参考里的多 take 场景尚未实现",
        "The only reconstruction backend in this build; ExMR, GVHMR, PEAR, MAMMA, PromptHMR and GEM-X are not ported" => "此版本只有一个重建后端；ExMR、GVHMR、PEAR、MAMMA、PromptHMR、GEM-X 均未移植",
        "Max frames" => "最大帧数",
        "(0 = all)" => "（0 = 全部）",
        "Playback speed" => "播放速度",
        "0 reconstructs the whole clip" => "0 表示重建整段视频",
        "Put the ground back where the clip put it" => "把地面恢复到片段自动测出的高度",
        "Automatic contact correction is not part of this build" => "此版本不含自动接触校正",
        "Export FBX…" => "导出 FBX…",
        "Write the selected person's motion as binary FBX 7.4" => "把选中人物的动作写成二进制 FBX 7.4",
        "BVH export is being rebuilt for the SAM 3D Body skeleton" => "BVH 导出正在针对 SAM 3D Body 骨架重建",
        "Quit" => "退出",
        "Workspace guide" => "工作区指南",
        "Keyboard shortcuts" => "键盘快捷键",
        "Console" => "控制台",
        "Language / 语言" => "语言 / Language",
        "Motion Workspace" => "动作工作台",
        "Open a video or drop it into the workspace" => "打开视频，或把视频拖进工作区",
        "Ready. Open a video or a project." => "就绪。打开视频或工程开始。",

        // Movie clip.
        "Movie Clip" => "视频片段",
        "No clip loaded" => "未载入视频",
        "Overlays" => "叠加显示",
        "Show tracked people over the source frame" => "在源画面上显示已跟踪的人物",
        "People" => "人物",
        "Stream" => "视频流",
        "How to choose, rename and display people" => "如何选择、重命名和显示人物",
        "No people yet.\nReconstruct a clip to find them." => "还没有人物。\n重建一段视频即可找到。",
        "Show this person in the 3D viewport" => "在 3D 视口中显示此人物",
        "Find person" => "查找人物",
        "Reconstruct every person in the clip" => "重建视频中的每一个人",
        "Default" => "默认",
        "Open a video to begin" => "打开视频开始",
        "Perspective" => "透视",
        "Front" => "正面",
        "Back" => "背面",
        "Right" => "右侧",
        "Left" => "左侧",
        "Top" => "顶部",
        "Bottom" => "底部",
        "User Perspective" => "用户透视",

        // Viewport.
        "3D Viewport" => "3D 视口",
        "Frame All" => "全部取景",
        "Fit all visible people in the 3D view" => "把 3D 视图中所有人物纳入取景",
        "Bone Overlay" => "骨骼叠加",
        "Show the skeleton used for FBX export over the motion" => "在动作上叠加显示用于 FBX 导出的骨架",
        "Draw the 70-keypoint skeleton over the selected body" => "在选中人体上绘制 70 关键点骨架",
        "No motion loaded" => "未载入动作",
        "Reconstruct a clip to see every person as a body here." => "重建视频后，每个人都会以人体模型显示在这里。",
        "Floor Grid" => "地面网格",
        "Correction Floor" => "校正地面",
        "Motion Path" => "运动轨迹",
        "Name" => "名称",
        "Camera" => "相机",
        "Follow Actor" => "跟随人物",
        "Motion correction is not part of this build" => "此版本不含动作校正",
        "Contact locks are not part of this build" => "此版本不含接触锁定",
        "Camera frustums are not part of this build" => "此版本不含相机视锥",
        "LMB: Orbit | MMB/RMB: Pan | Wheel: Zoom | Double-click: Frame" => "左键 旋转 | 中/右键 平移 | 滚轮 缩放 | 双击 取景",

        // Timeline.
        "Timeline" => "时间轴",
        "How to seek, trim and edit contacts" => "如何定位、裁剪和编辑接触点",
        "Play / pause (Space)" => "播放 / 暂停（空格）",
        "Clip: no video" => "片段：无视频",
        "Reset Trim" => "重置裁剪",
        "Reset Locks" => "重置锁定",
        "Frame" => "帧",
        "Video" => "视频",
        "No data" => "无数据",
        "Hidden" => "已隐藏",

        // Sidebar.
        "AI Model" => "AI 模型",
        "? Model Guide" => "? 模型指南",
        "Solver" => "求解器",
        "SAM 3D Body · MHR" => "SAM 3D Body · MHR",
        "Graphs" => "模型图",
        "(none)" => "(无)",
        "Mesh not loaded yet" => "网格尚未载入",
        "Solver Settings" => "求解器设置",
        "Static Camera" => "静态相机",
        "Center At Origin" => "人物居中到原点",
        "Put the reconstructed people on the world origin (one shared offset, so their distances to each other stay). Turn it off to keep the metric position the camera saw. Used by the next reconstruction" => {
            "把重建后的人物放到世界原点（统一平移，人和人之间的距离不变）。关掉就保留相机看到的真实位置。下次重建生效"
        }
        "Camera refinement is not part of this build" => "此版本不含相机优化",
        "Motion Correction" => "动作校正",
        "Enable Correction" => "启用校正",
        "Correction Mode" => "校正模式",
        "Flat Ground" => "平地",
        "Foot" => "脚",
        "Hand" => "手",
        "Lock Preset" => "锁定预设",
    "Smoothing" => "时间平滑",
    "Off" => "关",
    "Light" => "轻",
    "Medium" => "中",
    "Smooth the generated motion before the contacts are detected. Helps a noisy take, but a fast one can be refused by the correction gate, which leaves it uncorrected" => "在接触检测之前先平滑网络生成的动作。对噪声大的片子有收益；但动作快的片子可能被闸门整段拒绝，那就等于没有校正",
        "Normal" => "正常",
        "Auto Foot Height" => "自动脚高",
        "Floor Height" => "地面高度",
        "Apply Correction" => "应用校正",
        "Foot/hand contact correction is not in this build." => "此版本不含手脚接触校正。",
        "Processing Resources" => "处理资源",
        "provider found" => "已找到执行后端",
        "CPU only" => "仅 CPU",
        "Work width" => "处理宽度",
        "Confidence" => "置信度",
        "FlowHMR motion" => "FlowHMR 动作",
        "After the reconstruction, generate one coherent SMPL-H clip per person with the FlowHMR ONNX model and preview that instead of the per-frame bodies" =>
            "重建后为每个人生成一段时序连贯的 SMPL-H 动作（FlowHMR ONNX），并用它代替逐帧人体来预览",
        "Model Manager" => "模型管理器",
        "Some required model files are missing." => "缺少部分必需的模型文件。",

        // Capture stages.
        "idle" => "空闲",
        "decoding" => "正在解码",
        "detecting people" => "正在检测人物",
        "reconstructing" => "正在重建",
        "detecting" => "正在检测人物",
        "interpolating hands" => "手部特征插值",
        "stabilising" => "正在稳定化",
        "failed" => "失败",
        "Reconstructing every person…" => "正在重建所有人物…",
        "Cancelling reconstruction…" => "正在取消重建…",
        "VeloMocap native core ready." => "VeloMocap 原生内核就绪。",
        "ONNX Runtime not found: runtime/onnxruntime is missing." => "未找到 ONNX Runtime：缺少 runtime/onnxruntime。",
        "ffmpeg not found: runtime/ffmpeg is missing." => "未找到 ffmpeg：缺少 runtime/ffmpeg。",

        // Windows.
        "Workspace guide body" => "Workspace guide body",
        other => other,
    }
}

/// `Person 1  ·  42 f` in the People list.
pub fn person_row(name: &str, frames: usize) -> String {
    match lang() {
        Lang::Zh => format!("{name}  ·  {frames} 帧"),
        Lang::En => format!("{name}  ·  {frames} f"),
    }
}

/// `Clip: <path>` above the timeline.
pub fn clip_line(path: &str) -> String {
    match lang() {
        Lang::Zh => format!("片段：{path}"),
        Lang::En => format!("Clip: {path}"),
    }
}

/// People/frames summary in the Solver group.
pub fn people_line(people: usize, frames: usize, fps: f64) -> String {
    match lang() {
        Lang::Zh => format!("{people} 人 · {frames} 帧 @ {fps:.2} fps"),
        Lang::En => format!("{people} people · {frames} frames @ {fps:.2} fps"),
    }
}

/// Status bar sentence while a run is going.
pub fn run_status(
    stage: &str,
    people: usize,
    frame: usize,
    total: usize,
    rate: f64,
    backend: &str,
) -> String {
    match lang() {
        Lang::Zh => format!(
            "{stage}  ·  {people} 人  ·  第 {frame}/{total} 帧  ·  {rate:.1} fps  ·  {backend}"
        ),
        Lang::En => format!(
            "{stage}  ·  {people} people  ·  frame {frame}/{total}  ·  {rate:.1} fps  ·  {backend}"
        ),
    }
}

/// Pick between a Chinese and an English sentence for the active language.
pub fn zh_en(chinese: &str, english: &str) -> String {
    match lang() {
        Lang::Zh => chinese.to_string(),
        Lang::En => english.to_string(),
    }
}

/// `Detected N people — pick who to reconstruct`.
pub fn candidate_line(count: usize) -> String {
    match lang() {
        Lang::Zh => format!("检测到 {count} 人 · 请勾选要重建的人物"),
        Lang::En => format!("Detected {count} people — pick who to reconstruct"),
    }
}

/// `Selected <path>` after opening a clip.
pub fn selected_line(path: &str) -> String {
    match lang() {
        Lang::Zh => format!("已选择 {path}"),
        Lang::En => format!("Selected {path}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chinese_is_the_default_and_every_key_translates() {
        set_lang(Lang::Zh);
        assert_eq!(lang(), Lang::Zh);
        assert_eq!(t("Movie Clip"), "视频片段");
        assert_eq!(t("3D Viewport"), "3D 视口");
        set_lang(Lang::En);
        assert_eq!(t("Movie Clip"), "Movie Clip");
        // Unknown strings must never blank a label.
        assert_eq!(t("Something we forgot"), "Something we forgot");
        set_lang(Lang::Zh);
    }

    /// The Settings window and its storage labels were falling back to English in
    /// Chinese mode; a translation that is missing is invisible here because `t`
    /// returns the key, so the keys are pinned explicitly.
    #[test]
    fn the_settings_window_is_translated() {
        set_lang(Lang::Zh);
        for key in [
            "Settings",
            "Project storage",
            "Precision",
            "Compression",
            "Storage precision note",
            "Ultra (lossless)",
            "High (default)",
            "Medium",
            "Low",
            "None",
            "Fast",
            "Balanced",
            "Smallest",
        ] {
            assert_ne!(t(key), key, "{key} has no Chinese translation");
        }
        assert_eq!(t("Settings"), "设置");
        set_lang(Lang::En);
    }
}
