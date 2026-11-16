# 下一阶段：彻底解决脚锁定 / 跳跃 / 抖动

> 2026-11-16 22:57 固化。停电前保存的现场记忆，明天从这里继续。
> 代码状态：`master` = `a25fbbf`，工作区干净，已推送 origin。

---

## 0. 明天要做的三件事（用户原话）

1. **脚锁定完全解决**——"原版脚根本都不会动，老稳了"，要追平参考实现的稳定性。
2. **跳跃不对**——跳起时人物不能被摁在地上，根骨骼要跟着人走。
3. **抖动**——"部分存在一点颤抖"，要消掉。

## 1. 已定决策（别再回头讨论）

- **不跑参考的 Python**，不再做 `foot_lock.exe` 那种外挂 helper。PyInstaller 虽已装进
  bundled python，但用户明确说"不跑他们的Python了，到时候还是我们自己研究"。
  → 脚锁定逻辑**用 Rust 原生复刻**，读懂公式后自己实现。
- 参考项目是规格，不许发明算法："Do not invent algorithms"。
- 我们是它的**优化版**（性能、质量更好），不是换一套做法。
- 用户赶时间：**不要跑全量测试**，要测就只测一个片段。
- 安装新 exe 时**绝不杀用户正在跑的窗口**（改名 `velomocap-app.running-<pid>.exe` 再覆盖）。

## 2. 当前校正管线（现状）

`crates/velomocap-motion/src/correction.rs` 的 `apply_with_locks` 顺序：

```
smoothing → detect → grounding(step 0) → foot_lock(X/Z drift)
          → root_height(drop/lift/offset/clearance) → pose_plan(时序根)
          → leg_ik(3D 接触目标) → hand lock
```

- `grounding_pass`：step 0 落地。**坑已修**（`b2364fb`）：以前用 `low.is_finite()` 判断，
  而 `f32::MAX` 也是 finite，于是所有腾空帧都被当成"踩在 3.4e38 m"，`desired` 被夹到
  −0.75 m，把人整体拖进地面（实测 `ground 0.750 m`、`sink 236 mm`）。改成显式 `any` 标志后
  正常：`ground 0.035 m`、`sink 0.0`、根抖动 30.2→21.8 mm。
- `leg_ik`：**只**保留参考的 `smooth_correction`（对应用后的髋/膝角度做 50 ms 一阶低通，
  `PinFilter`）。**不要**再想着去平滑 IK 位置目标——试过了，越平越糟
  （slide 0.47→0.26 cm/frame，float→1.1 mm，sink→8 mm），`leg_ik` 里有注释记录原因。
- `root_height.rs`：`grounding()` + 五对角 Cholesky `banded_grounding()`，`GROUND_LIMIT=0.75`。
- `pose_plan.rs`：时序根规划（掩码、reach 上下界用绝对高度、pose/vel/acc 项
  σ=.175/.025/.004、temporal weight .5、根 σ .01、根 acc .0004、±0.02 m）。
- `anatomical.rs`：`KneePlane` + `knee_plane()`、`rotation_angle()`。

### 已证明正确的锚点（改动后必须仍然成立）

| 素材 | 结果 |
|---|---|
| `1min.mp4` 走路段 | slide 0.00/0.00 cm/frame，float 0.0/0.0 mm，sink ≤0.2 mm，foot jitter 0.0/0.0 mm |
| `tennis.mp4` walk 段 | 同上，全部 0.00 / 0.0 |
| `tennis.mp4` serve 段（90–179 帧） | ground 0.035，root 0.84..1.06 m（跟着跳），root jitter 21.7 mm（原 30.1） |

`tennis.mp4`：10.44 s @ 29.83 fps ≈ 311 帧，**跳跃在约第 131 帧**。

## 3. 参考的脚锁定逻辑（明天要 1:1 复刻进 Rust 的部分）

`E:\RGBMocap\toyxyz_RGBMocap\mocap\pear\foot_lock.py`（130 行，已全部读过，要点如下）。

**`Contact` 状态机（因果、实时）**

| 参数 | 值 | 含义 |
|---|---|---|
| 候选窗口 | `-0.06 < y < 0.045` 且 speed < 0.18 | 累计 dt 进入候选 |
| 锁定 | 候选累计 0.075 s | 进入 locked |
| 解锁 `y` | `y > 0.065` | 抬脚 |
| 解锁 speed | `speed > 0.30` | 甩脚 |
| 解锁 drift | drift > 0.12 | 飘走 |

**应用（`LiveFootLock.apply`）**

- 锁定时不硬钉，给 **0.16 s Hermite 偏移过渡**（offset handover）。
- 强度用**临界阻尼**逼近 `omega = 24`；骨盆位移 `omega = 12`，夹在 `[-0.04, 0]`。
- `observable` 门限：接近伸直时才允许校正（防 IK 病态）。
- 膝盖 pole 用**上一帧方向延续**（`bone_facing` 兜底）。
- 脚趾跟随踝关节的位移增量（`toe follows ankle delta`）。
- `smooth_correction`：对实际下发的旋转校正做 **50 ms 一阶低通**（已复刻）。
- `TIPS` 映射（SMPL-X 脚趾尖端）——SMPL-H 52 关节里没有，明天决定用 toe(10/11) 末梢代替或忽略。

**还没搬的三块**（≈20 分钟工作量，明天主攻）：
① `Contact` 因果状态机；② 实时两骨锁定（observable 门限 + pole 延续）；③ 脚趾跟随。

## 4. 明天开工顺序建议

1. 先把 `Contact` 状态机 + 两骨锁定 + 脚趾跟随搬进 `correction.rs`（或新模块
   `crates/velomocap-motion/src/live_lock.rs`），保留 `apply_with_locks` 作为离线路径。
2. 只测 `tennis.mp4` 的 **serve 段（90–179）** 一个片段，看三件事：
   `slide / float / sink / foot jitter` 不变差，`root jitter` 继续降，跳起时根不贴地。
3. 抖动若仍在：优先查 `pose_plan` 的 temporal 项与 `root_height` 的 clearance 之间是否互相拉扯
   （两处都在动根，容易形成低频振荡）。考虑把根的最终结果只由一处决定。
4. 通过后再改 UI / 打包，不要提前动 exe。

## 5. 环境与命令（照抄）

```powershell
$env:VELOMOCAP_LIBTORCH="1"
$env:PATH="C:\Users\38820\miniconda3\Lib\site-packages\torch\lib;"+$env:PATH
cargo build --release -p velomocap-app -p velomocap-cli
```

- 安装目录：`E:\VeloMocap-Release\VeloMocap\velomocap-app.exe`
- 当前已安装版本：由 `a25fbbf` 构建，SHA256 开头 `D32A1AF2...`（用户尚未试）
- 补丁分发：用户**只替换主 exe**，不得新增 DLL/模型，`.vp` 格式不能变；
  打包前删掉残留的 `velomocap-app.running-*.exe`。
- 参考项目：`E:\RGBMocap\toyxyz_RGBMocap`（Qt 原版，`mocap/correction.py`、
  `mocap/pear/foot_lock.py` 是规格来源）。

## 6. 其他未完项（低优先级，别在这周碰）

- 布局不跨重启保存；时间轴滚动条不常显。
- `.vp` 未持久化 solver 输入 → 项目里加载的 take 目前不能点 Apply Correction。
- FootMR / PEAR 实时预览 / GEMX / retarget 稳定器未做。
- `assets/smplh.tar.xz` 曾出现又消失，从未提交（模型不进 git）。
