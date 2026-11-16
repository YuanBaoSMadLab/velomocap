# VeloMocap

**普通 RGB 视频转三维动作捕捉，Rust + C++ 实现，运行时不依赖 Python。**

VeloMocap 把一段手机视频变成可用的三维动作：检测人物、重建人体、估计相机运动、生成时间连贯的动作，
最后把角色导出成一个带动作的 **.glb**——网格、骨骼、材质和动画都在里面，直接拖进 Blender、Unity、Unreal。

> English README: [README.md](README.md)

![VeloMocap 工作区](assets/screenshot.png)

```text
视频 ─► YOLOv8 检测 ─► IoU 跟踪
                        │
                        ▼
      SAM 3D Body（一个 TorchScript 模块，走 LibTorch；无 CUDA 时回退 ONNX）
      躯干 → 双手裁剪 → 融合 → MHR 网格  18 439 顶点 / 36 874 面
                        │  特征：躯干 + 左手 + 右手（3072 维）
                        ▼
      VGGT-Omega 相机（ONNX）─► FlowHMR MMDiT（ONNX）+ Rust 采样器
                        │  SMPL-H  6 890 顶点 / 52 关节
                        ▼
      接触清理 ─► 3D 视口 · 带动画的 .glb
```

| 环节 | 实现 | 实测（RTX 5060 Ti） |
| --- | --- | --- |
| 解码 / 定位 | ffmpeg 管道，任意编码 | 定位一次约 60 ms |
| 人物检测 | YOLOv8x，ONNX Runtime CUDA | 预热后 1 ms/帧 |
| 身份跟踪 | Rust，IoU + 惯性 | 90 帧内 4 人稳定 |
| **SAM 3D Body（完整）** | **LibTorch 加载 TorchScript** | **0.44 秒/人帧**，与 PyTorch 偏差 1.5e-05 |
| SAM 3D Body（回退） | Rust 驱动 11 个 ONNX 图 | 0.26 秒/人帧 |
| **FlowHMR 动作** | VGGT + MMDiT ONNX，采样器 Rust | 20 步欧拉，输出 30 fps |
| 人体模型 | 纯 Rust SMPL-H / SMPL-X | 6 890 顶点，52 关节 |
| 视口 | CPU 光栅化，z-buffer，Gouraud | 14.7 万三角形约 3 ms |
| **导出** | 角色资产 .glb：网格、蒙皮、材质、动作 | 10.5 万顶点 / 67 骨，30.0 MB |

## 特点

* **原生桌面程序**（egui/eframe），布局对齐原版工作台：视频片段、3D 视口、时间轴、
  人物列表、设置侧栏，中文/英文界面。
* **导入 → 检测 → 勾选 → 重建**：导入视频先做一次检测，你勾要捕捉的人，只重建他们。
* **流式播放**：播放时只起一个 ffmpeg 进程顺序读帧，由解码出的帧驱动播放头，画面与动作不会漂移。
* **多人**：每个人有独立身份、颜色和自己的一段动作。
* **导出**：应用里的导出就是下面的「角色资产导出」。`velomocap-cli` 仍可另外写出裸骨架的 BVH
  （Character Studio 骨骼名，统一 `Zrotation Xrotation Yrotation` 编码）和厘米单位的二进制
  FBX 7.4。
* **角色**：可导入 **VRM** 虚拟形象、**虚幻引擎**身体（`models/characters/` 里带 `UE5_Manny_Simple`、`UE5_Quinn_Simple`、
  `Mixamo_Remy`），以及任意 **Mixamo 骨骼** 的 glTF/FBX/GLB；片段会用它自己的材质和贴图
  驱动角色。角色库存在本机，导入一次所有工程共用。
* **角色资产导出**：一个照 DCC 导出面板做的窗口——勾上网格/蒙皮、材质、动画，给每个角色
  单独起文件名，输出目录点「浏览…」选。写出的就是视口里那个模型，一个 **.glb** 装下网格、
  骨骼、材质和动作，导入 Blender / Unity / Unreal 直接得到「骨骼网格体 + 动画序列」，而不是
  只有需要重新蒙皮的动作；贴图就在文件里面，不会掉。
* **MMD（实验性）**：PMX 能加载材质、付与（继承）和变形骨；MMD 用刚体物理驱动的头发、裙摆
  还没有模拟。
* **可自证**：SAM 的比对测试在同一个进程里同时跑 PyTorch 原版和导出模块；导出的 `.glb`
  会被丢进 Blender 重新导入，把重建出来的骨架（绑定姿态和每一帧动作）与文件自身的数据
  逐骨对比。

## 安装

两种方式：

1. **便携版** —— 解压即用，双击 `VeloMocap.cmd`。LibTorch、ONNX Runtime、ffmpeg、模型都在文件夹里，
   不需要安装任何东西，也不需要配置环境变量。
2. **从源码构建** —— 需要 Rust 1.80+ 与 Visual C++ 工具链：

```powershell
cargo build --release -p velomocap-app -p velomocap-cli

# 可选：让 SAM 走 LibTorch（快 2.4 倍，且包含手部）
$env:VELOMOCAP_LIBTORCH = "1"
$env:PATH = "<torch>\lib;$env:PATH"
```

**模型权重不在本仓库内。** `tools/fetch_models.py` 列出了程序会查找的全部文件，
`models/manifest.json` 记录了大小与 SHA-256。SMPL-H 需要自行到
[smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/) 注册下载，放到
`models/body_models/smplh/neutral/model.npz`。

## 使用

```powershell
VeloMocap.cmd                               # 图形界面
VeloMocap.cmd clip.mp4 --capture            # 无界面重建所有人
VeloMocap.cmd clip.mp4 --play               # 打开并播放
VeloMocap.cmd clip.mp4 --detect-only        # 只做检测，打印候选人物
```

![重建出的人体](assets/body-generated.png)

命令行：

```powershell
velomocap-cli flow clip.mp4 --person 1 --frames 60 --bvh out.bvh --fbx out.fbx
velomocap-cli people clip.mp4 --ply out/          # 每人一个 PLY
velomocap-cli ort                                 # 执行后端报告
```

## 已验证

| 检查项 | 结果 |
| --- | --- |
| SAM 3D Body 对比 PyTorch 原版（bf16） | 顶点 1.3e-03 m，2D 关键点 0.6 px，相机 1.0e-03 |
| SAM 3D Body 对比 PyTorch 原版（fp32） | 顶点 1.5e-05 m |
| FlowHMR 特征 | 3072 列全部填充（躯干 + 双手） |
| BVH 导出 | 52 骨、统一 ZXY、Character Studio 命名 |
| FBX 导出 | 52 骨、159 条曲线、厘米单位、重新解析并与源骨架比对 |
| FBX 旋转保真 | 第 0 帧完全一致；12 帧内最大 0.035（矩阵元素） |
| 端到端 | 90 帧 × 4 人，单卡 21 分钟 |

## 仓库结构

```
crates/velomocap-app      egui 工作区、软件视口、多语言
crates/velomocap-motion   捕获、FlowHMR 特征、接触清理、BVH/FBX
crates/velomocap-infer    ONNX 会话、检测器、跟踪器、SAM 链、网格拓扑
crates/velomocap-body     SMPL-H / SMPL-X 解析与蒙皮
crates/velomocap-core     BVH、FBX、接触规划、数学、C++ 内核
crates/velomocap-video    ffmpeg 解码 / 定位 / 流式播放
crates/velomocap-cpp      LibTorch 桥（C ABI）
crates/velomocap-models   模型清单与校验
tools/torch, tools/onnx   离线导出 TorchScript / ONNX 模型的脚本
```

## 许可

**Apache License 2.0** —— 见 [LICENSE](LICENSE)。许可范围**仅限本仓库的代码**；
模型权重、SMPL/SMPL-H 与第三方二进制各自保留其许可，详见 [NOTICE](NOTICE)。

## 标签

`motion-capture` · `pose-estimation` · `3d-human` · `smpl` · `smplx` · `bvh` · `fbx` ·
`rust` · `cpp` · `onnxruntime` · `libtorch` · `gpu` · `computer-vision` ·
`markerless-mocap` · `egui` · `video-to-animation` · `retargeting`
