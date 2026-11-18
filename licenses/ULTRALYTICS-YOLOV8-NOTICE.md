# YOLOv8 (Ultralytics) - not redistributed here

The person detector in this project is an ONNX export of **Ultralytics
YOLOv8x** (and optionally `yolov8x-pose`). Ultralytics publishes YOLOv8 - code
*and* trained weights - under **AGPL-3.0**, with a paid Enterprise licence as
the alternative. See <https://www.ultralytics.com/license>.

That licence is why this project does **not** put `yolov8x.onnx` in the
repository or in a release, and why it should not be uploaded to a model hub
without reading the next paragraph.

## What redistributing it would mean

AGPL-3.0 is a network copyleft licence: if you convey the work or a derivative
of it, you must do so under AGPL-3.0 and provide the corresponding source. A
converted ONNX graph is a derivative of the weights, so publishing it would
mean:

* publishing the whole set under AGPL-3.0 - including the parts of this
  application that a court would read as one work with the detector;
* offering the source of the AGPL component and of the work that uses it, to
  anyone who receives it, including over a network service;
* keeping the Ultralytics copyright and licence notice with it.

If that is not what you want, the alternatives are an Ultralytics Enterprise
licence, or another detector with a permissive licence.

## What this repository does instead

`tools/export_yolo.py` exports the detector locally from the official
Ultralytics checkpoint, and verifies the checkpoint's SHA-256 against the
published release asset before loading it:

```powershell
python tools\export_yolo.py --weights yolov8x.pt --imgsz 640
# -> models/detector/yolov8x.onnx, used by the app if present
```

The user downloads the checkpoint and accepts the AGPL-3.0 terms themselves;
the licence exposure stays with the party that chose to use it, and nothing
AGPL is bundled with the Apache-2.0 code.

The application also accepts any other `.onnx` detector of the same
input/output shape, so a different detector can replace YOLOv8 without code
changes.
