# 模型文件从哪来、许可证是什么

本仓库的**代码**是 Apache-2.0，**模型权重不是**。每个模型文件都保留它来源项目的
许可证；而且 `models/` 里的文件大多数是这里**转换**出来的（ONNX 导出、TorchScript
追踪、权重 sidecar 打包、参数表抽取）。转换件属于衍生作品，**不会获得新许可证**，仍然
受原权重条款约束。

这个目录里的三份许可证全文，就是随转换件一起分发时必须带上的文本：

| 文件 | 覆盖 | 能否再分发 |
| --- | --- | --- |
| `SAM-LICENSE.txt` | SAM 3D Body 与 MHR 几何（`facebook/sam-3d-body-dinov3`） | 可以，必须随附本协议全文 |
| `FlowHMR-Model-Weights-LICENSE.txt` | FlowHMR 权重（`fafsaf1/flowhmr-0.46B`） | 可以，**仅限非商业**，保留署名 |
| `VGGT-Omega-LICENSE.txt` | VGGT-Omega（`facebook/VGGT-Omega`） | 可以，**仅限非商业研究**，必须随附本协议 |

两个上游**不允许再分发**，本仓库也不包含：

| 文件 | 覆盖 | 能否再分发 |
| --- | --- | --- |
| SMPL-H / SMPL-X（`models/body_models/`） | Max Planck 人体模型 | **不可以**：需要注册，由用户自己下载，见 `SMPL-NOTICE.md` |
| YOLOv8（`models/detector/yolov8*.onnx`） | Ultralytics 人物检测器 | **AGPL-3.0**，见 `ULTRALYTICS-YOLOV8-NOTICE.md` |

## 要点

* **导出的动作是你的。** 许可证约束的是权重，不是用它生成的动作；但 FlowHMR 与
  VGGT-Omega 的非商业条款在部分解读下会延伸到"输出的商业使用"，商业项目请按研究用途
  处理，或取得授权 / 换模型。
* **SAM 3D Body 不是非商业许可。** 它允许商业使用，禁止它列出的贸易管制与军事用途，
  并要求任何副本附带协议全文——所以它在这个目录里。
* **SMPL-H 不能打包。** 注册是逐人的，把 `model.npz` 发给别人就是违反条款。
* **YOLOv8 决定整个产品的许可证。** 带着 AGPL-3.0 权重分发，义务远重于 Apache-2.0；
  本项目的默认做法是不带它，让用户自己导出。

转换脚本、校验方式和出处的逐文件清单见英文版 `README.md`；要发布到 Hugging Face /
魔搭见 `UPLOAD.md`。

## 引用

* SAM 3D Body — Meta，`facebook/sam-3d-body-dinov3`
* FlowHMR — *FlowHMR: Physically Plausible Motion Capture from Video*，arXiv:2610.03691
* VGGT — *VGGT: Visual Geometry Grounded Transformer*，CVPR 2025
* SMPL-H — Romero et al. 2017 / Loper et al. 2015；SMPL-X — Pavlakos et al. 2019
