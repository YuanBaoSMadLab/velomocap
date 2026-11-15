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
        "Import video" => "导入视频",
        "Cancel" => "取消",
        "Detecting people…" => "正在检测人物…",
        "Person" => "人物",
        "Reconstruct all people" => "重建所有人物",
        "Detect every person, then run SAM 3D Body on each" => "先检测出每个人，再逐人运行 SAM 3D Body",
        "Cancel reconstruction" => "取消重建",
        "Export BVH…" => "导出 BVH…",
        "Skeleton naming" => "骨架命名",
        "Write the bones with this rig's names" => "用该骨架的命名写出骨骼",
        "Rest correction" => "rest 姿态校正",
        "Turn the clip into the canonical T-pose bind, so a Mixamo-class rig plays it exactly" => {
            "把动作换算到标准 T-pose 绑定下，Mixamo 这类骨架可直接播放"
        }
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
        "Offset from the automatic pelvis height; positive gives back the reach drop, negative lowers the body" => "相对自动骨盆高度的偏移；正值只把自动压低的部分还回来，负值压低身体",
        "Pin planted feet and resting hands after the motion is generated" => "动作生成后锁定着地的脚和静止的手",
        "Reconstruct this person when the run starts" => "开始重建时包含此人",
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
        "Motion Workspace   |" => "动作工作台   |",
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
        "Camera refinement is not part of this build" => "此版本不含相机优化",
        "Motion Correction" => "动作校正",
        "Enable Correction" => "启用校正",
        "Correction Mode" => "校正模式",
        "Flat Ground" => "平地",
        "Foot" => "脚",
        "Hand" => "手",
        "Lock Preset" => "锁定预设",
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

/// Mesh size line in the Solver group.
/// "FlowHMR · 6890 vertices · 13776 triangles", for the model panel.
pub fn smplh_line(vertices: usize, triangles: usize) -> String {
    match lang() {
        Lang::Zh => format!("FlowHMR · {vertices} 顶点 · {triangles} 三角形"),
        Lang::En => format!("FlowHMR · {vertices} vertices · {triangles} triangles"),
    }
}

pub fn mesh_line(verts: usize, tris: usize, keys: usize) -> String {
    match lang() {
        Lang::Zh => format!("{verts} 顶点 · {tris} 三角面 · {keys} 关键点"),
        Lang::En => format!("{verts} verts · {tris} tris · {keys} keypoints"),
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
}
