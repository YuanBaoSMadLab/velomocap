# 脚锁定 / 跳跃 / 抖动 —— 进展与后续

> 2026-11-17 12:15。代码状态：`master` 见 `git log`；已安装 exe 的 SHA256 开头 `8ADB9F70`。
> 上一版记忆（2026-11-16）已归档在 git 历史里，结论继续有效。

---

## 0. 用户的三个目标（原话）

1. **脚锁定完全解决**——"原版脚根本都不会动，老稳了"，要追平参考实现的稳定性。
2. **跳跃不对**——跳起时人物不能被摁在地上，根骨骼要跟着人走。
3. **抖动**——"部分存在一点颤抖"，要消掉。

## 1. 今天做完的（都有数据）

### 1.1 参考实现的实时脚锁已原生复刻

`correction.rs::live_foot_lock` + `LiveContact`，是 `apply_with_locks` 的**最后一遍**：

- 因果状态机：着地证据要 75 ms（`-60mm < y < +45mm` 且速度 < 0.18 m/s）才上锁；
  解锁条件 `y > 65mm` / `speed > 0.30` / 漂移 > 0.12 m；0.16 s Hermite 交接。
- 两骨求解用参考自己的公式：接近伸直时的 `observable` 门限、膝盖 pole 跨帧延续、
  对**实际下发的旋转**做 50 ms 一阶低通、脚趾随踝位移。
- 骨盆补偿 `move_root` **关掉**：上面的根部链条已经把接触点做成可达，这一遍再压只会
  让身体下沉 14 mm、把已钉住的脚压进地面 14 mm。参考同一份代码在 `move_root=False`
  时就是我们现在这样。

### 1.2 三个真正的 bug（这才是"脚不锁/抖"的根源）

1. **"do no harm" 闸门用全部 52 个关节算抖动**。发球动作的手臂单独就把 p99 抬到
   144 mm，于是"把浮空 17 mm 的脚修到 1 mm"的校正被判成"更抖"并整体回滚——
   用户看到的正是"脚根本没被锁"。现在只量这条管线动的关节
   （根、髋、膝、踝、趾，`CORRECTION_JOINTS`）。
2. **腿 IK 的 50 ms 低通原来滤的是"绝对关节角"**，跨站姿时会把上一站姿的整条腿角度
   混进来：新站姿第一帧脚会飞出去 ~35 cm 再花几帧飘回来（发球窗口实测 467 mm 抖动、
   slide 0.47→1.59 cm/frame）。改成滤**相对源姿势的增量**后：抖动 46.5 mm、
   slide 0.47→0.22 cm/frame。
3. **实时脚锁原来跟踪的是"滚动支撑点"**（脚跟→脚尖），滚脚被读成脚在动，
   于是站姿中途解锁、3 帧后在新位置重锁 = 打滑。改成跟踪踝坐标系里的**固定材料点**
   （参考的 `rig.sole_offsets`），取整段的中位数。

### 1.3 现在的实测（两个片段，真实管线）

发球窗口 `tennis.mp4` 90–179（全片推理，90 帧）：

| 指标 | 修正前 | 修正后 |
|---|---|---|
| slide (cm/frame) | 0.47 / 0.26 | **0.22 / 0.12** |
| float (mm) | 16.8 / 8.8 | **0.8 / 1.1** |
| sink (mm) | 2.2 / 0.0 | 3.4 / 3.6 |
| 脚抖动 (mm) | 0.9 / 0.5 | **0.4 / 0.3** |
| 根抖动 (mm) | 30.1 | **21.7** |

`短暂.mp4` 全片（84 帧，24 fps）：

| 指标 | 修正前 | 修正后 |
|---|---|---|
| slide | 0.30 / 0.30 | **0.16 / 0.08** |
| float (mm) | 13.2 / 4.0 | **0.6 / 0.2** |
| sink (mm) | 4.0 / 0.6 | 3.0 / 1.7 |
| 根抖动 (mm) | 4.2 | 4.1 |

根曲线仍跟着动作走（发球窗口 0.842–1.057 m，215 mm 行程），没有被按在地面。

## 2. 快速迭代工具（今天新加的，务必用）

```
# 一次推理，换永久秒级迭代
velomocap-cli flow <clip>.mp4 --start A --frames N --no-correction \
  --dump-joints src-joints.bin --dump-rotations src-rotations.bin \
  --dump-betas src-betas.bin --dump-contacts src-contacts.bin

# 之后每次改代码只要:
velomocap-cli replay src-joints.bin src-rotations.bin \
  --betas src-betas.bin --contacts src-contacts.bin --fps <fps> \
  [--no-live-lock | --no-leg-ik | --no-pose-plan] \
  --dump-joints out.bin --dump-soles out.txt
```

- 重建误差 0.27 mm（f32 旋转 + 16→10 betas 截断），A/B 足够。
- `VELOMOCAP_NO_GATE=1` 环境变量可以让被回滚的校正保留下来，用于看它自己的输出。
- **不要切片段再推理**：只给 90 帧时 VGGT 解不出平移，重建出来的人几乎站着不动
  （实测根位移 5 mm），结论会骗人。要测窗口就用原片 + `--start/--frames`。

## 3. 还没做 / 下一步

- **跳跃场景没有真正的素材验证过**：`tennis` 窗口里脚最高只离地 48 mm，`短暂.mp4`
  全程着地。要验证"腾空不被按下去"，需要一个真跳的片段（或用户那个 `.vp` 工程）。
  逻辑上已经确认：`grounding_pass` 只吃 `foot.support` 的帧，实时脚锁的
  速度/高度门限也不会在腾空时上锁。
- sink 3–4 mm 是"单点钉脚"的固有结果（参考也是单点）：脚被压得越斜，另一片鞋底越低。
  要再降需要多点接触约束，属于新功能，不是复刻。
- UI 里只加了一个“因果脚锁”开关；`leg_ik` / `pose_plan` 还没有开关。
- 布局不跨重启保存；`.vp` 仍不保存求解器输入（工程里的 take 不能 Apply Correction）。
- FootMR / PEAR 实时预览 / GEMX / retarget 稳定器未做。

## 4. 环境与规则

```powershell
$env:VELOMOCAP_LIBTORCH="1"
$env:PATH="C:\Users\38820\miniconda3\Lib\site-packages\torch\lib;"+$env:PATH
cargo build --release -p velomocap-app -p velomocap-cli
```

- 安装目录：`E:\VeloMocap-Release\VeloMocap\velomocap-app.exe`
- 补丁分发：用户只替换主 exe；不新增 DLL/模型，`.vp` 格式不变（`CorrectionSettings`
  现在有 `#[serde(default)]`，老工程文件仍能读）。
- 安装时**绝不杀用户正在跑的窗口**：先 `Get-Process velomocap-app`，若在运行就改名
  `velomocap-app.running-<pid>.exe` 再覆盖，最后删掉残留。
- 参考项目：`E:\RGBMocap\toyxyz_RGBMocap`（`mocap/pear/foot_lock.py` 已 1:1 复刻完）。

## 5. 2026-11-17 傍晚：Apply Correction 从来没到过视口

`CaptureShared` 在任务结束后一直保留结果，`poll_capture` 每帧都读它；只要 take 与快照
不同就整段替换。而"重新求解后的 take"必然与快照不同 —— 于是校正结果落地 16 ms 后被
换回校正前的烘焙。用户看到的"开不开锁脚一样"不是锁脚无效，是两次看到的是同一份。

- `poll_capture` 每轮 capture 只采纳一次（`flow_adopted`）；`start_capture` 里清空。
- 勾选"因果脚锁"立即触发 `apply_correction`；其余设置改动把 take 标记为过期并在面板提示。
- 因果脚锁的"do no harm"单独判定：变差只回滚它自己，不再拖累前面所有 pass。
- `rolling_anchor` 先对脚跟/脚尖高度差做 `max(0.5, fps*0.04)` 高斯（参考
  `support_weights` 就是这么做的），再进去抖动带。

### 三条路径的实测（`短暂.mp4`，105 帧，同源 dump，replay）

| pass | slide cm/帧 | float mm | sink mm | jitter mm |
|---|---|---|---|---|
| 精确 IK + 因果锁 | 0.16 / 0.08 | 0.6 / 0.1 | 3.0 / 1.9 | 21.9→35.5 |
| 只有精确 IK | 0.16 / 0.13 | 0.3 / 0.1 | 3.0 / 1.5 | 21.9→33.4 |
| 只有因果锁 | 0.25 / 0.16 | 2.0 / 1.2 | 17.8 / 6.5 | 21.9→21.1 |

因果锁单独跑明显不如精确 IK（sink 17.8 mm），叠在精确 IK 上才有用（右脚 slide 0.13→0.08）。
参考里 `LiveFootLock` 只用在 PEAR 实时预览（`pear/multi_worker.py:224`），批处理
`flow_correction.py` 完全没有它 —— 我们是把它当批处理的最后一级叠加，这是和参考的结构差异。

## 6. 脚抖的真凶：batch 里用了 live 路径的 50 ms 滤波器

`apply_leg_pin` 把求解结果又过了一遍一阶低通（抄自 `pear/foot_lock.py::smooth_correction`）。
那条滤波在 live 路径是对的（目标是因果猜测，需要阻尼），但 batch 里目标本来就是精确的
（stance anchor + 规划好的交接），滤波只是让"已解出的纠正"滞后于仍在移动的源。

| 版本 | slide cm/帧 | float mm | sink mm | 整体 jitter | ankle shake |
|---|---|---|---|---|---|
| 带滤波（旧） | 0.16 / 0.08 | 0.6 / 0.2 | 3.0 / 1.9 | 35.5 | 42.8 |
| 直接应用（新） | **0.02 / 0.01** | **0.4 / 0.1** | **0.0 / 0.7** | **28.3** | **34.6** |

tennis 90 帧窗口同样：slide `0.47/0.26 → 0.01/0.02`，float `16.8/8.8 → 0.3/0.5`，
root 0.842–1.057 m（跳还在），ankle shake 47.5→48.4（几乎不加）。

同时修正两处：
- `reach_falloff` 改回参考的 `1/(1+(gap/.03)^4)`（gap 以米计、只算"够不着"的部分）。
- 支撑点用 stance 的平滑脚跟权重（参考 `support_samples` 与 `rolling_anchor` 共用同一份）。

## 7. 显存 20 GB（未解决，先加日志）

16 GB 卡跑出 20 GB 说明分配器已经超订、Windows 在换页——之前的 onnxruntime CUDA 崩溃很可能就是它。
现在 Log 里每 10 帧、以及每个阶段边界都会写一行 `VRAM 1234/16311 MiB · <stage>`，
问题复现时把这几行贴出来就能定位是哪个阶段在涨。

## 8. 2026-11-17 晚：极端片子（2月27日.mp4, 562 帧）—— 四个真正的 bug

用户原话："这台极端片子脚抖、全身也抖，原版能真正钉死。" 逐条查下来，全部有个共同点：
**校正要么没生效，要么被自己人拆台。**

### 8.1 闸门把好结果扔了（"没生效"的元凶）

`2月27日.mp4` 第一版跑出来：`(rolled back by jitter: float 50.0->0.7 mm, slide 0.29->0.06)`。
它把 float 从 50 mm 干到 0.7 mm，然后因为**膝盖**的二阶差分从 22.6 涨到 56.9（>1.5 倍）整段丢弃。
用户看到的还是网络原始动作。修法：脚掌大幅改善时不允许因腿的加速度丢，加 3 倍上限。

### 8.2 释放帧把脚按到地面以下

`add_transitions` 把 stance 的竖直残差（这台片子约 7 cm）原样带进"脚已经在离地"的帧，
实测把鞋底压到 **-56 mm**，膝盖为够到它被拉直 —— 那就是看到的膝盖弹跳（膝 20 -> 100 mm）。
参考在非接触帧有 `floor_lift`；我们给交接帧加了同一条限制：可以降到鞋底触地，不能更低。
膝 100 -> 34 mm，整体 jitter 56.9 -> 28.6，踝抖动 48.5 -> 33.9。

### 8.3 因果脚锁是实时预览用的，批处理里只会拆台

参考的 `LiveFootLock` 只出现在 `pear/multi_worker.py:224`，批处理 `flow_correction.py` 没有。
在精确接触 IK 之后再加一道因果求解，等于对同一条腿投第二票，而且更不精确：

|  | slide cm/帧 | float mm | sink mm |
|---|---|---|---|
| 2月27日 开 | 0.05/0.06 | 0.7/0.4 | 3.2/0.0 |
| 2月27日 关 | **0.03/0.05** | **0.0/0.0** | 3.1/0.0 |
| 短暂 开/关 | 0.02/0.01 → **0.01/0.00** | 0.4/0.2 → **0.1/0.0** | 0.0/0.7 → 0.0/0.5 |
| tennis 开/关 | 0.01/0.02 → **0.00/0.00** | 0.3/0.5 → **0.0/0.0** | 0.2/3.5 → **0.0/0.1** |

默认改为关（UI 里仍可勾选，CLI 从 `--no-live-lock` 改成 `--live-lock`）。

### 8.4 多层级模式在平地上更糟

`--lock-mode multi` 三台片子全部更差（jitter 22.6->81.6 且被回滚），flat 保持默认。

### 最终数字（replay，同源 dump）

| 片子 | slide cm/帧 | float mm | sink mm | jitter | ankle shake |
|---|---|---|---|---|---|
| 2月27日 562f | 0.03 / 0.05 | 0.0 / 0.0 | 3.1 / 0.0 | 22.6→29.1 | 28.1→35.2 |
| 短暂 105f | 0.01 / 0.00 | 0.1 / 0.0 | 0.0 / 0.5 | 21.9→28.3 | 27.9→34.0 |
| tennis 90f 窗口 | 0.00 / 0.00 | 0.0 / 0.0 | 0.0 / 0.1 | 45.4→49.1 | 47.5→49.1 |

### 还没解决

- `2月27日.mp4` 的**输入本身**竖直不稳：同一个 stance 里鞋底在 -1.6 cm..+7.3 cm 之间漂，
  grounding 因此把根抬高 9 cm。校正只能把它压到一个平面上，代价是腿要伸直；这是网络/落地尺度的
  问题，不是锁脚算法的问题。
- CLI 全程采样：SAM 3D 阶段瞬时冲到 **15840 MiB**（16 GB 卡），20 秒后回落。app 里加上查看器
  的开销就会超订 —— 之前 20 GB 与 onnxruntime CUDA 崩溃大概率是同一个峰值。

## 9. 膝盖极向稳定器：试过，无效，已回退

"脚还有一丢丢不稳、腿有一丢丢晃" 之后试的方案：钉脚固定了踝、根固定了髋，腿只剩一个自由度 ——
膝盖绕"髋→踝"轴的方位角。近伸直时这个角是噪声（这台片子上 p99 二阶差分 5.2°→25.4°，
叶屈角只有 5–13°），所以按参考 `TwoBoneIk` 的思路做"极向延续"：每帧最多把膝绕钉脚轴转 7°，
朝上一帧的方向（先把上一帧的极向用轴对齐旋转搬到当前轴坐标系，避免把转身当成晃动）。

- 旋转本身是对的：实测"要求 -2.0°，实现 -2.0°，踝移动 0.00 mm"（踝在旋转轴上，脚不动）。
- 但膝盖二阶差分 p99 只从 34.5 → 34.2 mm（限制收到 2° 时 33.7），整体 jitter 29.1 → 29.4。
- 原因：腿的残余运动不是极向在晃，而是**钉脚本身**要求腿改变形状 —— 踝和髋都被固定后，
  膝盖位置由 |髋−踝| 唯一确定，只有极向可调，而极向对结果影响 ~0.3 mm。

结论：这段代码已回退（`git checkout`），只保留文档。剩下能改善"腿晃"的只有**输入**本身
（这台片子的鞋底在 −1.6 cm..+7.3 cm 之间漂，校正必须让腿去吸收这 8 cm），以及参考里那套
全局最小二乘的 `contact_temporal_root`（我们只有逐窗口的 `pose_plan`）。

另：`tight` / `normal` 预设在这三台片子上输出**完全相同**（没有短缺口可桥接，5 cm 单帧上限也从不触发），
所以预设不是这里的杠杆。

## 10. 原版 `contact_pose_plan` 的完整移植：做了什么，什么没换来收益

参考的稳定性来自 `mocap/contact_pose_plan.py::_optimise_window`（在 `correction.py:383` 调用）：
变量 = 根高度 **+ 每个摆动帧的目标高度（±2 cm）**，残差 = 预测膝盖角的**位置/速度/加速度**
（乘 `motion_weight = .25+.75/(1+(源膝速/20°)²)`）+ 根加速度 + 可达性 + 自由脚离地间隙。

我们的 `pose_plan.rs` 早就有膝盖角那三项（所以根抖动一直不差），缺的是：
每摆动帧目标高度变量、`follow` 权重（`free_follow_weights`）、以及把规划出的目标回写给 IK。

### 实测（三台片子，replay，同源 dump）

| 版本 | feb 整体 jitter | feb 踝抖动 | feb sink |
|---|---|---|---|
| 基线 | 29.1 | 35.2 | 3.1 mm |
| 完整移植（含交接帧正则） | **35.6** | 39.5 | 0.5 mm |
| 修正根位移 double-count 后 | 31.2 | 36.1 | 0.5 mm |
| **只保留离地间隙、正则只作用于 stance** | **28.6** | **34.7** | 3.1 mm |

- 第一版少减了一个「规划后根位移」：`offset` 必须写成 `goal_y − (ankle_y + Δroot)`，否则交接帧的
  脚会跟着根被抬/压（这台片子根动 9 cm），那正是最大的一截抖动来源。
- 即便如此，把交接帧纳入膝盖角正则仍然更抖：这台片子的交接帧本来就需要腿改变形状，
  正则只是在跟输入较劲。
- **留下的是**：规划自己拥有交接帧目标高度 + 参考的离地间隙项（自由脚被抬出地面而不是埋进去），
  以及 `angle − phase·extension` 的状态定义。变量与掩码都已就位，只是当前 `movable=false`。

## 11. 2026-11-17 晚：对照原项目审计 —— 这次是整段被闸门丢了

完整报告见 [foot-lock-audit.md](foot-lock-audit.md)。要点：

- `2月27日.vp`（17:57，Tight，`live_lock: true`）里保存的 take **就是原始动作**：
  summary 明写 `(rolled back by jitter: float 62.2->0.7 mm, slide 0.28->0.14,
  jitter 21.6->40.7 mm)`，且 feet before == after（float 62.2/57.8 mm）。
- 丢掉的原因：`big_sole_gain` 要求 slide 至少减半（`0.28*0.5 = 0.14`），
  而工程里带的因果脚锁把 slide 从 0.08 拖到 0.14，正好卡在边界 → 大 float
  收益被 jitter 1.88 倍否掉。参考批处理既没有这道闸门，也没有因果脚锁。
- 工程文件会把 `live_lock=true` 带回来（`project.rs:474` / `app.rs:1886`），
  即使代码默认已关。
- 另两处真实差异：每帧 IK 多了参考没有的 `reach_falloff` 权重和 5 cm 截断
  （`correction.rs:1495/1526/1594`）；导入角色（本工程是 `vrm.vrm`）没有参考
  `retarget/` 那一级的目标骨架脚重锚（foot_calibration / polish_feet / stabilize）。
- 锁住之后腿仍动是几何必然：脚底钉死、踝世界朝向保留源旋转，源脚踝 pitch
  二阶差分 p99 4.6°，踝/膝必须反向补偿（实测 ankle shake 26.9→42.9 mm）。
  `短暂.vp` 的中位数是 0.13 mm / 0.5°，尖峰只在 stance 进出的 1–2 帧。
- 已加两个只读工具：`probe/vp_inspect.py`（直接量 .vp 的脚/踝/膝）和
  `probe/vp_dump.py`（把 .vp 抽成 `replay` 的 dump，免推理 A/B）。

## 12. 2026-11-17 22:00：工程文件里的 take 是原始动作，闸门又把它丢了

直接读用户保存的 `C:\Users\38820\Videos\测试\2月27日.vp`（17:57，106 MB）：

```
correction settings: preset Tight, live_lock true, smoothing 0, 其余默认
take summary: ... jitter 21.6->40.7 mm; live lock re-solved 1124 leg(s),
  worst residual 45.9 mm (rolled back by jitter: float 62.2->0.7 mm,
  slide 0.28->0.14 cm/frame, jitter 21.6->40.7 mm)
feet: slide 0.25/0.28 -> 0.25/0.28, float 62.2/57.8 -> 62.2/57.8 mm
```

feet 的前后值一模一样 = viewport 里这份 take **就是 FlowHMR 原始动作**。
也就是说：校正算出来了（float 62.2 → 0.7 mm），然后被闸门整段丢掉。
用户看到的正是「脚根本没锁住」——两脚整段悬空 6 cm、以 0.28 cm/帧滑动。

两个原因，都已修：

1. **`big_sole_gain` 的 slide 条件要求「减半」。** 它本来就只是「大 float 收益不被
   腿的加速度否掉」的挡箭牌，却比 `slide_helps` 更严：`0.14 <= 0.28*0.5` 只在
   恰好一半时成立，于是 jitter 1.88 倍就触发了整段回滚。改成「不变差」
   （`*1.05 + 0.02`，和 `slide_helps` 同一条线），并补上 `float_before > 0`
   （`0.0*4 <= 0.0` 恒真，以前任何贴地片子都能靠这条挡箭牌保住无限的腿动）。
   抽出 `sole_gain_wins()` 并加回归测试，用的就是这组数字。
2. **工程文件把 `live_lock` 带回来了。** 它在存盘时是默认开，而它正是把 slide
   从 0.08 拖到 0.14 的那一遍（参考只在 PEAR 实时预览里跑它）。读工程时现在会
   忽略它并写日志。

同一份 .vp 动作（`probe/vp_dump.py` 抽出来、`replay` 免推理）修复前后：

| 配置 | slide cm/帧 | float mm | sink mm | 回滚 |
|---|---|---|---|---|
| 工程里存着的 take | 0.25/0.28 | 62.2/57.8 | 0.0/0.0 | 是 |
| 修复前 replay（Tight+live） | 0.38/0.42 | 62.2/57.8 | 0.0/0.0 | 是 |
| **修复后 replay（Tight+live）** | **0.08/0.07** | **0.6/0.3** | 3.5/0.0 | 否 |
| **修复后（载入后 live 已关）** | **0.04/0.07** | **0.1/0.0** | 1.2/0.0 | 否 |

另外三台片子不受影响：`2月27日` 0.03/0.05 + 0.0 mm、`短暂` 0.01/0.00 + 0.1 mm、
`tennis` 0.00/0.00 + 0.0 mm，都没有回滚。

### 这一轮另外两个「试过但有数据否掉」的改动

- **时间平滑作为默认**：`replay --smooth`（新加的参数，和 `flow --smooth` 同一个）。
  `2月27日` 上它是明显赢的（jitter 28.6→21.4、ankle 34.7→25.2，脚一点没变差），
  但 `tennis` 上 ankle 48.8→65.7、`短暂` 上直接被闸门回滚。**不做默认**，
  按片子单独试。
- **去掉每帧 IK 的 5 cm 截断**（参考的 `SolveLegs` 没有）：
  脚掌指标全优（三台片子 sink 全 0.0、两台 slide 0.00），但踝抖动各 +4 mm
  （34.7→38.4、34.0→38.1）而 slide 没变。**保留截断**，注释里写了这组数字。

### 锁住之后还剩下什么（这台片子的真实分解）

平滑输入下、**连续三帧都在站立**的帧上：踝二阶差分中位数 **0.2 mm**、
p99 6.3/1.0 mm——比输入本身（11.0/10.0 mm）还稳。所以中段是真的钉死了；
剩下的 ~40 帧（560 帧里）是 stance 进出的**跳变**（10–40 mm），
`ankle 2nd diff >10mm` 只有 37/43 帧。手交接那几帧不是元凶：
关掉交接反而更糟（jitter 28.6→47.5）。

## 13. 2026-11-17 22:15：把审计表里的「权重淡出 / 跳过整帧」改成参考的二值全权重

审计表列的三条（每帧 IK 权重淡出、5 cm 截断、够不到就跳过整帧）里，
后三条其实是同一件事：`RGBMoCapAnatomical.h:53` 对每个接触帧都**解到底**，
`flex = min(acos(...), 160°)` 是在解算器内部夹住的；我们却是
`reach_falloff` 打折 + `continue` 跳过。跳过一帧、旁边一帧全解 = 台阶，
而台阶正好落在 stance 的进出处——和 §12 量到的「尖峰只在进出的 1–2 帧」对得上。

已改成默认全权重（参考行为），`VELOMOCAP_IK_TAPER=1` 可切回旧的打折路径做 A/B：

| 片子 | 旧（打折） | 新（全权重） |
|---|---|---|
| 2月27日 | slide 0.03/0.05, float 0.0, jitter 28.6, ankle 34.7 | **完全相同** |
| 短暂 | 0.01/0.00, 0.1/0.0, 29.0, 34.0 | **完全相同** |
| tennis | 0.00/0.00, 0.0/0.0, 48.8, 48.8 | **完全相同** |

也就是说：**这三条在现有素材上是惰性的**——因为 `reachable_pelvis_drop/lift`
已经把每个站立目标都做成可达，淡出与跳过从未触发（`--floor 0.10` 这种极端
构造下也一样）。所以它不是调参，而是把「目标不可达时」的行为从「跳过整帧」
换成「按参考解到底并内部夹住」——只在真正不可达时才生效，而那时跳过就是台阶。

剩下没动的两条，都是量过、明确否掉的：

- **5 cm 截断**：去掉后脚掌指标全优（三台片子 sink 全 0.0、两台 slide 0.00），
  但踝抖动各 +4 mm、slide 不变。保留（§12 有数字）。
- **导入角色没有目标骨架那一级**（`foot_calibration` / `polish_feet` /
  `stabilize.solve`）：这台工程挂的是 `vrm.vrm`，这是**视图层**的真实差异，
  需要我们自己在角色骨架上重锚脚，工作量比前面几条大，单独一轮做。

## 14. 2026-11-17 22:30：修复后审计 —— 闸门还不看 sink

§12/13 的改动验证通过：`sole_gain_wins(0.28, 0.14, 62.2, 0.7)` 为真；工程载入
强制 `live_lock=false`；全权重 IK 在三台片子上确实惰性；交接关掉后 stance 边缘
的踝 p99 从 57/39 mm 涨到 67/61 mm（所以交接要留）。用 `2月27日` 的 replay 逐项
跑完，剩下的是：

1. **闸门不检查 `sink`。** `FootMetrics.sink_mm` 只出现在注释和一个测试里
   （`correction.rs:4041`），`helps` 只看 float/slide/jitter。实测：

   | replay | slide | float | sink | jitter | 结果 |
   |---|---|---|---|---|---|
   | `--smooth 0.001` | 0.46→0.17/0.21 | 57.8→0.4 | **0.1→29.2/30.1** | 7.5→27.6 | 被回滚 |
   | `--smooth 0.0001` | 0.50→0.15/0.18 | 55.5→0.4 | **4.1→73.3/78.4** | 7.5→22.5 | **放行** |
   | `--smooth 0.0001 --no-leg-ik` | 0.50→0.29 | 55.5→1.5 | **4.1→90.8/91.4** | 7.5→7.2 | — |

   9 cm 沉脚能过闸门，因为 float/slide 都"变好"了。`sole_gain_wins` 和 `helps`
   都要加一条 `sink_after <= sink_before + 0.01`。

2. **5 cm 截断会把"抬回地面"的那一截也削掉。** `apply_leg_pin` 先把 `want`
   （含 `:1630-1637` 的 floor lift）整段截到 5 cm；目标差 9 cm 时，竖直方向只
   给了 5 cm，脚就留在平面下。建议：水平仍截 5 cm，但竖直的 floor-lift 不受截断
   （lift 之后再夹水平分量），这样"宁可脚贴地"而不是"截断后沉进去"。

3. **时间平滑在 app 里没有开关。** `settings.smoothing` 只在日志里打印
   （`app.rs:3026/3041`），UI 没有控件，所以 GUI 永远是 0。它在这台片子上是最大
   的单项收益（踝 42.9→28.7），但现在 (1)(2) 没修之前开了只会回滚或沉脚。
   修完 (1)(2) 后建议加一个"平滑输入"开关，或自动各跑一次、只在不增加 sink 的
   前提下采用。

4. **剩下能看见的抖只有 stance 进出的 1–2 帧。** 连续三帧站立的帧上：
   鞋底二阶差分中位数 0.18/0.24 mm、踝 0.92/1.04 mm、膝 0.47/0.38°；边缘帧
   踝 p99 57/39 mm。关掉交接边缘翻倍，说明交接方向是对的；这段是真实抬脚，
   不是 bug。要再压只能做"进/出各 2 帧的权重坡"这类新东西。

5. 小事：日志时间戳是 UTC（`chrono_like_clock` 用 UNIX epoch，22:16 记成
   14:16），对时差 8 小时的用户看日志不友好；`.vp` 不存 `source`，载入后
   `can_apply_correction` 为 false（`project.rs:739`），每次验证都要重新推理；
   `smooth_weights`/`reach_falloff` 在全权重下已经白算，可删或留作 A/B。

## 14. 2026-11-17 22:35：P1 闸门看 sink、P2 截断不吃 floor-lift、P3 平滑开关、P5 本地时间

按审计给的顺序做完 P1+P2，顺手做了 P3 的开关和 P5 的时间戳。

### P1：闸门完全没看 sink

`sink_mm` 以前只在注释和测试里出现。现在 `helps` 加了 `sink_helps`
（`sink_after <= sink_before + 10 mm`，1 cm 松量：脚"落在"平面上本身就会量出
一两毫米穿透），`sole_gain_wins` 也把 sink 当**否决**而不是收益
（拿 6 cm 悬空换 7 cm 入土不叫改善），`SkipReason::Sink`，日志里也打印
`float/sink/slide/jitter` 四组前后值。

### P2：5 cm 截断把"抬回地面"那一截也削掉了

`apply_leg_pin` 的 `want` 里包含 `leg_ik` 算好的 floor-lift，整段截到 5 cm，
于是目标差 9 cm 时竖直只给 5 cm，脚留在平面下。改成：

- **水平**仍按 5 cm 截（横着拖脚才是要防的）；
- **竖直**单独放宽到 2.5 倍上限（`cap*2.5`），因为竖直那一截就是"抬回地面"。

实测（`2月27日`，默认设置）：`sink 6.8/6.9 -> 0.0/0.0 mm`（原来是 3.1/0.0），
slide/float/jitter/ankle 全不变。审计里那三条 sink 爆表的记录（29.2 / 73.3 /
90.8 mm）现在是 `0.0/0.0`。

### 闸门的第三条腿：ankle

修完 P1/P2 之后 jitter 上限（3x）为了照顾平滑输入放宽到 `3x + 5 mm`，
结果 `tennis --smooth 0.001` 以 `float 0.3->0.0、slide 0.74->0.00` 通过了闸门，
而它的踝抖动是 **27.8 -> 107.2 mm**——闸门当时根本没有踝这一项。
现在 `ankle_helps = ankle_after <= ankle_before*1.5 + 25 mm`，`SkipReason::Ankle`。
25 mm 这个松量是有依据的：钉住脚这件事本身就要腿付 15–25 mm（三台片子实测
+6/+6/+1，平滑输入下 +17/+24）。

### 平滑现在是可选档位（P3）

侧栏"时间平滑：关 / 轻 / 中"（0 / 0.001 / 0.003），会置 `correction_dirty`。
**默认仍然是关**，而且这是量出来的决定：

| 片子 | 关 | 轻(0.001) | 中(0.003) |
|---|---|---|---|
| 2月27日 | ankle 34.7, sink 0.0 | **ankle 24.6**, sink 0.0 | ankle 25.6, sink 0.0 |
| 简短 | ankle 34.0, sink 0.0/0.3 | **ankle 28.4**, sink 0.0/0.2 | 被闸门拒（jitter） |
| tennis | ankle 48.8, sink 0.0/0.1 | **被闸门拒（ankle）** | 被闸门拒（ankle） |

也就是说"轻"在 2/3 片子上赢 6–10 mm，但在 tennis 上会被整体拒绝——拒绝的
结果是**一点校正都没有**，比不开更糟。所以开关给了用户，默认关，tooltip 里
明说"动作快的片子可能被闸门整段拒绝"。

### P5 顺手两条

- **日志时间戳改成本地时间**：`GetLocalTime`（Windows），不再用 UNIX epoch
  取模——以前 22:16 的跑会记成 `[14:16:53]`。
- `smooth_weights` / `reach_falloff` 在全权重下确实白算，但 `VELOMOCAP_IK_TAPER=1`
  的 A/B 还要用，先留着。

### 还没做的

- **VRM 目标骨架那一级**（P5 第一条）：这是剩下最大的缺口，单独一轮。
- **`.vp` 不存 `source`**，载入后 `can_apply_correction` 为 false，每次验证只能
  走 replay。要存 source（或存"未校正的 pose 序列"）才能让工程内直接重解。

## 16. After the sink/ankle gate: where the shaking is (2026-11-18)

The gate fix works — the 22:57 run keeps its pass with `sink 0.0/0.0`,
`float 0.0/0.0`, `slide 0.03/0.05 cm/frame`, and my replay reproduces the
same shape. The remaining complaint ("feet and legs still shake") is **inside
the leg chain**, and it is the pin's own motion.

### Per-joint second difference, p99, source -> corrected

Replay of `2月27日` (`feb-joints.bin` -> `new-d014.bin`, floor 0.014, preset
normal, mesh sole, pose plan). `probe/joint_jitter.py`:

| joint | src | out | delta |
|---|---|---|---|
| toe.R (11) | 18.35 | 39.68 | **+21.3** |
| ankle.R (8) | 18.84 | 38.13 | **+19.3** |
| knee.L (4) | 14.62 | 32.11 | **+17.5** |
| toe.L (10) | 27.16 | 41.48 | **+14.3** |
| ankle.L (7) | 25.51 | 37.55 | **+12.0** |
| knee.R (5) | 12.65 | 24.07 | **+11.4** |
| elbow/wrist/shoulder | — | — | +0.3 each |
| every other joint | — | — | within +-0.5 |

All-joint p99: 94.96 -> 93.49. So the pass adds no noise anywhere except the
three leg joints per side. That is the shaking the user sees, and no sole
metric can see it: the *sole point* is held to 0.3 mm while the foot rotates
around it.

### Why: the ankle rotation is kept, so the ankle has to move

`RGBMoCapAnatomical.h:33` ("Output changes only hip/knee/ankle. Ankle world
direction is retained") is ported faithfully. Consequence, measured with
`probe/foot_turn.py` on planted frames (low sole, slow ankle):

| | src planted | out planted |
|---|---|---|
| foot turn deg/frame p99 (L/R) | 7.48 / 3.26 | 7.02 / 5.45 |
| shin turn deg/frame p99 | 7.27 / 3.17 | 6.12 / 3.51 |
| ankle move mm/frame p99 | 24.68 / 15.45 | 16.98 / 13.24 |

The pin cuts the ankle's *slow* drift by ~30 % and leaves its *high frequency*
untouched: the ankle has to travel 17 mm/frame to keep a foot that is turning
7 deg/frame in place. A character's shoe is rigidly attached to that foot, so
it pivots about the pinned sole — which reads as a trembling foot even though
nothing slides.

### The added shake is the frame-to-frame jump of the correction

`probe/correction_filter.py`: take the applied displacement
`c = out_ankle - src_ankle`, filter it along time with a **zero-phase**
gaussian (the batch path can look ahead; the reverted `smooth_rotation`
one-pole could not, which is why it lagged), and re-measure.

| sigma (frames) | ankle p99 | toe p99 |
|---|---|---|
| 0 (today) | 38.13 | 41.48 |
| 1.0 | 25.27 | 27.27 |
| 1.5 | 24.39 | 25.95 |
| 2.0 | 24.74 | 27.17 |

The sole's own motion on planted frames does not get worse (17.9 -> 18.4-19.2
mm on the joint proxy, within noise). So ~13 mm of ankle shake and ~14 mm of
toe shake are pure per-frame correction jumps, not motion. (This filters joint
*positions* offline; the implementation belongs on the *target offsets* before
`apply_leg_pin` solves them.)

### What the reference has that we do not

- Its **batch SMPL correction is the same design** (`SolveLegs` keeps the ankle
  world rotation; the target is `ankle + (anchor - support) * strength`). So the
  same trembling exists on its SMPL preview — this is not a porting slip.
- Its **character stage** has a second solve that we do not have at all:
  `retarget/pipeline.py:39-58` -> `bounded_stabilizer.solve` (imported by
  `measured.py:4`). A global least squares on the *target's knee flexion* per
  frame with: contact term `contact_sigma = 0.00005` (0.05 mm) to the contact
  goals, temporal term on the flexion's second difference with
  `physical_sigma = 0.008`, and `stable_pole` against knee flips
  (`bounded_stabilizer.py:52-202`). It even reports the metrics for this exact
  symptom — stance-interior flexion rms and boundary p95
  (`bounded_stabilizer.py:204-214`).
- Our character path is world-rotation delta plus a one-time `ground_offset`.
  Everything the pin does to the ankle and the knee is displayed raw.

### The smoothing setting is wired on the wrong side

`crate::temporal::smooth` runs at `correction.rs:471-478`, **before
`detect()`**, so it rewrites the input the contact detector sees. Measured on
the same replay:

- `--smooth 0.001`: ankle gate veto (7.4 -> 40.1 mm), contacts 25 -> 13, no
  correction at all.
- `--smooth 0.003`: passes, and it *is* effective (planted foot turn
  7.5 -> 3.9 deg, ankle 38 -> 17.7, thigh 6.1 -> 2.4) — but it changed the plan
  to 14 stance runs and a root shift of 0 -> 0.182 m, i.e. it did not smooth
  the motion so much as re-detect the contacts.

The reference states it explicitly: `flow_correction.py:3` "No inference,
SMPL-X conversion, upstream smoothing or second target IK pass." Its smoothing
lives in the character-stage solver. Anything that filters before detection
will keep changing the contact plan.

### Order for next round

1. Zero-phase filter on the pin's target offsets (sigma ~1 frame) — removes
   the 13-14 mm the pin adds, leaves the sole metrics alone.
2. Character-stage stabiliser, the `bounded_stabilizer` shape: knee flexion
   temporal term + contact term + stable pole.
3. Then either delete the smoothing setting or re-point it at (1).

## 17. Testing the target-offset filter: it does not hold, and the split says why (2026-11-18)

Section 16 proposed filtering the pin's per-frame target offsets with a
zero-phase gaussian, on the reading that the added shake is the frame-to-frame
jump of the correction and that all of it sits on the ankle, knee and toe. It
was implemented exactly that way — the filter runs on `target - current_ankle`
for every frame of a stance, before `apply_leg_pin`, with σ in frames — and
measured end to end with `--dump-joints` on four clips: the user's `.vp`
recovery (`vp-feb`), `2月27日` (`feb2`), `短暂` (`brief3`), `tennis` (`full`).

### What it costs

`vp-feb`, the clip the proposal came from:

| σ (frames) | slide cm/frame | float mm | sink mm | ankle shake | knee.L p99 | ankle.R p99 |
|---|---|---|---|---|---|---|
| 0 (shipped) | 0.04 / 0.07 | 0.1 / 0.0 | 0.0 / 0.0 | 43.1 | 35.35 | 36.37 |
| 1.0 | 0.14 / 0.15 | 0.3 / 0.2 | **4.6 / 4.1** | 44.7 | 35.41 | 30.98 |
| 1.5 | 0.17 / 0.18 | 0.4 / 0.3 | 7.5 / 5.7 | 46.8 | 35.94 | 31.76 |
| 2.0 | 0.20 / 0.20 | 0.6 / 0.5 | 9.9 / 6.8 | 48.0 | 35.94 | 32.55 |

The pin gives up its exactness (`slide` x3.5, `sink` 0 -> 4.6 mm) and the leg
joints do not get quieter: `knee.L` is unchanged to a hundredth, `ankle.R`
improves 5 mm, and the aggregate `ankle shake` gets *worse*. The other three
clips agree — σ=1.0: `feb2` slide 0.02/0.04 -> 0.04/0.06 with sink 0.0 -> 1.1 mm
and the ankle unchanged; `brief3` slide 0.01/0.00 -> 0.11/0.10, sink -> 1.5 mm;
`full` slide 0.00 -> 0.19/0.12, sink -> 4.6/1.7 mm.

### Why: the shake is not in the stance

`probe/ankle_when.py` splits every joint's second difference by the contact
state the pass itself used (`--dump-soles`):

| joint | planted p99 (src -> out) | edge p99 | swing p99 |
|---|---|---|---|
| ankle.L | 13.22 -> **8.37** | 65.67 | 37.93 |
| ankle.R | 13.86 -> **7.27** | 62.88 | 23.38 |
| toe.L | 13.70 -> **4.74** | 72.53 | 39.60 |
| toe.R | 12.54 -> **2.52** | 62.91 | 25.32 |
| knee.L | 12.16 -> **17.92** | 65.96 | 49.05 |
| knee.R | 11.33 -> 11.47 | 40.83 | 17.30 |

Frame counts: 441/460 planted, 42 edge, 77/58 swing. On planted frames the pin
makes the ankle and the toe **quieter than the source** — the opposite of
"moved onto the ankle". The p99 the proposal measured lands on the 42 edge
frames, whose values are 60-72 mm, and on the swing. A filter applied only
inside a stance can never reach those frames; all it can do is loosen the pin.

The one genuine plant-level regression is the **left knee**, 12.16 -> 17.92 mm,
and it has a different cause: this take's pins run at `d/max = 0.95-0.99` of
the leg's reach (from `VELOMOCAP_DEBUG_PIN`), and near full extension the
two-bone solve converts a millimetre of ankle motion into centimetres of knee
motion. That is the conditioning the reference's `stable_pole` exists for, in
the character layer, not a filter on the target.

### Why the edges are unavoidable on this clip

At the release the pin unwinds the whole correction the stance accumulated
(`[pin] f48..50 TRANSITION offset=(0.016,-0.074,-0.002) .. (0.021,-0.075,0.030)`,
i.e. **71 mm horizontally**), and the swing it has to unwind in is **3 frames**
(the right foot is airborne on frames 48-50 only). `add_transitions` already
uses every frame available and the reference's own rate — `extent =
clamp(|offset|*fps/0.8, 0.12 s, 0.5 s)`, `contact_offsets` — which for 71 mm at
30 fps is 3 frames. Nothing smooths 71 mm over 3 frames.

That is the price of holding a foot the network slides 0.32 cm/frame through a
40-frame stance: the correction grows to the size of the drift, and the release
has to shed it. Weakening the pin trades the slide for the pop; the reference
makes the same trade.

### Decision

- The target-offset filter is **reverted**. `--dump-joints` output is byte-identical
  to `b93132d` on all four clips, so nothing else moved.
- The pre-detection smoothing setting is **removed entirely** (`CorrectionSettings::smoothing`,
  `flow --smooth`, `replay --smooth`, the sidebar combo). It was a re-detection
  in disguise and the reference does no upstream smoothing (`flow_correction.py:3`).
  Section 16's item 3 is done; its item 1 is closed as measured-and-rejected.

### Order for next round, revised

1. Character-stage stabiliser — the reference's `bounded_stabilizer` shape:
   contact term on the ankle goals, temporal term on the knee flexion's second
   difference, `stable_pole`. This is where the only plant-level regression
   (`knee.L` +5.8 mm, the near-straight-leg amplification) is fixed, and the
   reference fixes it there too.
2. Only if a take still complains about the edges: a hand-off budget that can
   start *before* the stance ends, i.e. spend a little slide to buy the ramp.
   Measured tradeoff first — the sole is exact right now.

## 18. The hand-off is computed before the root chain and applied after it (2026-11-18)

Section 17 read the `[pin] f48..50 TRANSITION offset=(0.016,-0.074,-0.002)` line as
"71 mm horizontally" of accumulated stance slide. It is not: `y = -0.074` is
**vertical**, and the horizontal part is `|(0.016, -0.002)| = 16 mm`. That vector
is not slide either. It is the root chain's own translation, leaking into the
hand-off because the hand-off is built before the root moves the body.

### Where it happens

- `foot_lock` builds `pass.targets` in world space (`correction.rs:2731-2738`) and
  calls `add_transitions` at `:2821`, both **before** `grounding_pass` (`:530`),
  `root_height_pass` (`:531`) and `pose_plan_window` (`:533`).
- `add_transitions` stores `residual = pass.targets - current_ankle` (`:2873`), a
  *relative* vector measured on the pre-shift clip.
- A world-space target is invariant under a rigid root translation; a relative
  correction is not: after a shift `s` the same foot needs `residual - s`.
- The reference does exactly that subtraction: `contact_trajectory.py:164`,
  `offsets[active] = (anchors - support - shift) * strength`.
- Measured net root shift on this take: `y = -0.0848 m` median (range
  `-0.019 .. -0.121`). So the `-0.074` above should be about `+0.011 m` -- a
  small lift of the free foot, which is what the reference's own
  "never lower a swing foot" rule (`correction.py:396-407`) asks for. It is
  applied as a 74 mm press instead.

### What it does to the free foot

`--dump-soles` on fixed `2月27日` replay (`n2-soles.txt`), lowest sole patch above
the floor, on the frames the pass itself flags `active = 0`:

| | swing frames | median | p10 | max | within 10 mm of the floor |
|---|---|---|---|---|---|
| before (source) | 89 / 74 | 99.9 / 106.5 mm | 56.2 / 68.8 | 261 / 229 | 0 / 0 |
| after, hand-off on | 89 / 74 | **0.0 / 0.0 mm** | 0.0 / 0.0 | 173 / 137 | **66 / 51** |
| after, `VELOMOCAP_NO_HANDOFF=1` | 89 / 74 | 6.3 / 17.0 mm | -7.5 / 1.8 | 173 / 137 | 55 / 21 |

The correction presses the free foot onto the plane and `keep_sole_on_floor`
stops it there, so it is glued rather than buried. That matches the swing-frame
regression in section 17's own table: `knee.L` 21.08 -> 70.18 mm with the
hand-off, 40.24 mm without it, and `toe.L` 35.45 -> 45.99 / 44.34.

Note the clamp hides the size of the error but not the motion: the leg is still
asked to reach a target 85 mm lower than it should be, which is why a
near-straight swing leg turns it into a centimetre of knee movement.

### The two fixes this points at

1. Make the hand-off shift-invariant: interpolate `pass.targets` (world,
   invariant) instead of the residual and take `target - current` at apply time
   inside `leg_ik`'s transition branch, or move the `add_transitions` call after
   the root chain. Either way the stored quantity must not contain the root's
   translation.
2. Add the reference's free-foot rule as a clamp, not as a wish: the vertical
   part of a transition offset may lift a free sole and never lower it below the
   clearance the source had after the shift.

### Check after the fix, on the same dumps

- swing-frame sole clearance back to roughly `source clearance + shift`
  (~10-15 mm on this take), not 0;
- stance edge p99 no worse than now (25-66 mm), so the hand-off keeps the edge
  benefit it currently has;
- `knee.L` swing p99 near the no-hand-off 40 mm rather than 70;
- `--dump-joints` slide/float/sink unchanged.

Until this is in, the character-stage stabiliser is the wrong next module on
this take: `stable_pole` cannot undo a target that is 85 mm low.

## 19. Both halves of section 18, implemented and measured: one is wrong (2026-11-18)

Section 18 proposed two things: make the hand-off shift-invariant, and clamp the
free foot so it is never lowered. The first was implemented exactly as written —
`foot.root` recorded where the residuals were built, and
`target = current + offset - (root_now - root_then) * hold` in `leg_ik` — and it
is **wrong for this pipeline**. The second holds up and is kept.

### Why the subtraction double-counts

`[pin] f301 leg0 TRANSITION offset=(0.001,-0.009,0.001) shift=(+0.0000,-0.0778,+0.0000)`
on `2月27日`: the residual has already faded to 9 mm while the root has moved
78 mm, so the corrected target lands at `y = 0.153` — **70 mm above the stance
frames on either side of it** (the pin puts those at 0.076-0.085). Result:
`ankle shake 26.9 -> 74.0 mm`, whole-body jitter `21.5 -> 67.1`, and the gate
rolled the pass back on the ankle term.

The cause is that our anchor is not the reference's. The reference computes
`anchors`, `support` and `shift` **once** and then builds the target as
`fk(ankle) + shift + (anchors - support - shift) * strength`. We recompute the
anchor inside `leg_ik` from the **post-shift** clip (`rolling_anchor`), and its
vertical is the floor, so the stance target already carries the subtraction:

```text
reference:  T = fk(ankle) + shift + (A - S - shift) * strength      (shift applied to the pose first)
ours:       T = ankle_now + (A - S_now) * hold,  S_now = S - shift
```

Both give the same `T`. The residual `T - ankle` is therefore a difference of two
quantities that moved together, i.e. it is body-relative and the stance branch
already uses it correctly (measured: at the last stance frame the pin lands
within 2-4 mm of its target, both before and after this change). Subtracting the
root translation a second time asks for a lift nobody needs.

### What the vertical clamp alone buys

`offset[1] = max(offset[1], 0)` in the transition branch — the cast of
`correction.py:396` "Never lower a swing foot". `--dump-soles`, lowest sole patch
above the floor, on the frames the pass flags inactive:

| `2月27日` | swing frames | median | p10 | within 10 mm |
|---|---|---|---|---|
| source | 97 / 79 | 89.1 / 93.5 mm | 45.4 / 62.6 | 0 / 0 |
| before this change | 97 / 79 | **2.5 / 0.0 mm** | 0.0 / 0.0 | **66 / 51** |
| after | 97 / 79 | **6.7 / 17.5 mm** | 0.0 / 1.0 | 53 / 24 |

Sole metrics are untouched (`slide 0.32/0.38 -> 0.04/0.07 cm/frame`,
`float 63.0/60.5 -> 0.1/0.0 mm`, `sink 0.0/0.0 -> 0.0/0.0 mm`), the ankles pay
`43.1 -> 48.7 mm` and the whole-body jitter `37.2 -> 40.2`. The other three takes:
`feb2` identical, `full` better (`48.8 -> 45.8`), `brief3` worse
(`29.0 -> 35.1`, still inside the gate).

There is not much more to recover on this take: its *own* swing clearance over
the planted foot is only ~26 mm (89 mm swing against 63 mm float), so 10-15 mm
absolute is the ceiling, not 90.

### What is still not fixed

- The contact-edge p99 (65-72 mm on the ankles, 41-66 on the knees) is unchanged:
  it is the stance's accumulated *horizontal* correction being unwound, not the
  vertical press.
- `knee.L` on planted frames (12.2 -> 17.9 mm) is the near-straight leg
  (`d/max = 0.95-0.99`), i.e. the character-layer `stable_pole` problem from
  section 16, and this change does not touch it.

## 20. Correction to §19, and the near-extension trade measured (2026-11-18)

§19 concluded that subtracting the root shift from the hand-off "double-counts"
because `leg_ik` re-derives its anchors post-shift. That is too broad, and the
experiment behind it put the subtraction in the wrong place.

`leg_ik` re-derives the anchor per frame (`rolling_anchor` over the current
clip), so the *stance* target is already vertical-correct. `add_transitions`
does not: it reads `pass.targets`, which `foot_lock` wrote once before the root
chain moved anything (`correction.rs:2737`, called at `:520`, root chain at
`:530-533`), so the residual it stores is missing `-s` at the **stance
endpoints**.

The mistake was applying `-s` to the Hermite's *output*, i.e. to a value that
had already faded to 9 mm at f301 — which turns the "fade to zero" point into
+78 mm and puts the target 70 mm above the frames on either side. The shift
belongs on the endpoints, once:

```text
lvalue' = lvalue - s[first-1]      // the stance edge, the only place it exists
rvalue' = rvalue - s[end]
```

On this take the stance-edge residual becomes about `-80 + 78 = -2 mm` (the
same order as the +5 mm this note first estimated), the Hermite fades that to
zero, and nothing else changes. Interpolating world targets instead does not
work either: at the free end the interpolated target is the *pre-shift* foot
position. The two clean forms are endpoint subtraction, or moving
`add_transitions` after the root chain — which then needs `pose_plan_window` to
read `targets`, since it currently reads `pinned`/`offsets`.

The shipped clamp (`offset[1].max(0.0)`) is within ~5 mm of that correct value
on this clip, so it stays; the difference is only visible as the `短暂` jitter
`29.0 -> 35.1`, where the approach can no longer pre-lower the foot and the
stance does the settling instead.

### Near-extension softening: measured, not worth it

`VELOMOCAP_PIN_SOFTEN=<d/max>` (a diagnostic build, not shipped) faded the pin
out as the leg straightened, which is the cheap half of a `stable_pole`. On the
planted frames of `2月27日`:

| | knee.L plant | ankle.L plant | sink | slide |
|---|---|---|---|---|
| shipped clamp | 17.92 | 8.37 | 0.0 / 0.0 mm | 0.04 / 0.07 cm/frame |
| soften at 0.95 | **12.77** | 11.31 | **8.3 / 8.2 mm** | **0.20 / 0.15** |
| soften at 0.93 | gate rolled back (sink) | | | |
| soften at 0.90 | gate rolled back (sink) | | | |

It does fix the knee amplification (17.9 -> 12.8, back to the source's 12.2), and
it costs the sole the one thing the pin exists for: the ankle stops reaching the
plane, so the foot floats 8 mm and skates 3x as much. The exchange rate is bad
because at `d/max >= 0.95` the pin's contribution to the sole is a few
millimetres while its contribution to the knee is ~6 mm of second difference.
Rejected as a default; the character-stage solve is the place to buy this back.

## 21. The character stage: an instrument, the calibration port, and what it says (2026-11-18)

Section 16 and the audits agreed the last missing layer is the *target* skeleton:
our character path copies the clip'"'"'s world rotations (`character.rs`
`delta[canonical] = world · restᵀ`) plus one `ground_offset`, while the reference
runs `foot_calibration.corrected_world`, a two-bone contact solve, and a
`bounded_stabilizer` over the knee flexion. Nothing in the clip'"'"'s own metrics can
see any of that: the SMPL sole is pinned to 0.1 mm, and the character'"'"'s shoe is
bound to its own foot bone.

### The instrument

`velomocap-app --measure-soles <character> <project> <out>` poses every frame of
a saved take on a character and writes the lowest sole vertex per foot, plus two
summaries:

- **sole tilt** on planted frames (lowest sole < 8 mm): the angle between the
  sole band'"'"'s plane normal and the floor. A shoe that pivots about the ankle
  while the foot is planted shows up here and nowhere else.
- **lowest sole** per frame over the whole clip: `p05 / median / p95 / max`, in
  metres above the floor. This is what the eye calls "the foot floats" or "the
  foot sinks".

`Character::sole_vertices` is the vertex set both are measured on: bound mostly
to the ankle/toe bones (canonical 46/47, 50/51) and below the ankle, the same
selection the reference uses for `eval_pins`.

### Baseline, on the user'"'"'s VRM and their own take

`2月27日.vp`, 562 frames, `vrm.vrm` (110148 vertices):

| | planted tilt (L/R) | lowest sole p05 / median / p95 / max |
|---|---|---|
| as shipped | 3.8 / 4.3 deg | -48.9 / **+15.7** / +44.2 / +58.5 mm |
| `VELOMOCAP_FOOT_CALIBRATION=1` | 5.8 / 5.5 deg | -62.1 / **+9.6** / +38.2 / +55.7 mm |

The clip'"'"'s own sole is on the floor by construction, so a planted character
should read ~0. It reads a **+16 mm median with a ±50 mm spread**: the shoe is
neither flat nor still, and the spread is the character'"'"'s own leg geometry, which
no rotation copy can fix — that is stage 2 (a target-side two-bone solve against
the contact points).

### The calibration port, and why it is off by default

`Character::calibrate` is `corrected_world` faithfully: per side, measure the
character'"'"'s own sole frame `F_c` from its vertices and weights, the clip'"'"'s `F_s`
from the neutral body, and premultiply `Q = F_s · F_cᵀ` onto the ankle'"'"'s and
toe'"'"'s delta (`world · restᵀ · Q`, derived in the same way as the reference'"'"'s
`native · C`). The rotation *is* applied and does move the shoe — but on this
pair it makes the shoe **more** tilted (3.8 -> 5.8 deg) and the p05 worse
(-48.9 -> -62.1), for a 6 mm better median.

The defect is upstream of the calibration: the **source** sole frame is wrong on
one foot. Measured with `VELOMOCAP_DEBUG_SOLE=1`:

```text
calib: side 0 source up [-0.0015, 0.99999, -0.0049] forward [0.9238, 0.0032, 0.3829]  Q turns 67.5 deg
calib: side 1 source up [-0.0015, 0.99956, 0.0297]  forward [-0.2648, -0.0290, 0.9639] Q turns 15.6 deg
```

The left foot'"'"'s `forward` is mostly **+x** — sideways — where the right one is
`+z`. Both come from `toe − ankle` with the clip'"'"'s joint indices (7 -> 10,
8 -> 11), so either the body model'"'"'s joint order is not the clip'"'"'s, or the
weight row lookup (`weights[joint * vertices + vertex]`) does not match the
patch on this model. Until that is settled the switch stays off: a 67-degree
rotation on one foot would be visible, and the honest state is "the instrument
is in, the calibration is not trusted yet".

### What this changes for the plan

Stage 1 as ordered ("C + a sole-height constant") cannot reach 2-3 mm on its own:
the constant only shifts the median, and the ±50 mm spread is the missing
target-side solve. The constant was measured too — `ground_offset` already puts
the sole at +16 mm median, and adding the raw foot-thickness difference
(+29 mm) made it +23.9 mm, i.e. worse; it was dropped.

Stage 2 (target-side two-bone contact solve, ~120 lines) is the next real step,
and `--measure-soles` is its acceptance test: planted tilt -> ~0, and the
lowest-sole median -> 0 with a tight spread.

## 22. The cast grew while the take ran (fixed)

Reported: "one person selected at the start, three by the end; I tested the
previous version". Not reproducible on a 4-frame run - it needs a frame where
the picked track is *not* detected.

The old rule, per frame:

* frame 0 kept every track overlapping a seed box;
* every other frame kept the seeded tracks, and - when none of them had a
  detection that frame - *all* the tracks that passed `continues(last_box)`:
  overlap > 0.15, or a centre within 0.6 body sizes of the box the person was
  last seen in. Each accepted id got its own `PersonTrack`.

So one missed detection of the picked body, with three people standing around
it, produced four people in that single frame. That is the "1 -> 3" jump.

Now (`selection_tests`):

* `picks[seed]` holds the identity; `slots[seed]` is the one `PersonTrack` its
  frames land in, whatever the track id becomes.
* While the track coasts (`missing <= max_missing`) the identity is held, so no
  other body can answer for it. A replacement is only looked for once the
  tracker has retired the track (`missing > max_missing`).
* A track owned by another pick cannot be taken over.

Evidence: the legacy rule replayed on the same scripted frames (1 picked body,
3 bystanders, one missed detection) creates 4 people; the new rule creates 1.
That assertion was run once and then deleted, since keeping the old rule in the
tree would mean keeping dead logic around. Real clip, 4 people on screen,
`--select 1`, 30 frames: one seed, one track, one `flow: Person 1`.

Headless runs no longer stay in the window loop: `VELOMOCAP_EXIT_AFTER_CAPTURE=1`
exits after the report. A leftover process held `velomocap-app.exe` locked, so
the next `cargo build` could not link.

Still open from this: when the picked body is retired *and* a bystander is the
best `continues` match, that bystander is adopted into the pick's own slot -
a few frames of the wrong body rather than a second person. ByteTrack's
`with_reid` is what the reference uses to avoid it.
