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
