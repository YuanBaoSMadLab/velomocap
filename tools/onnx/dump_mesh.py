"""Write the MHR mesh topology next to the SAM 3D Body graphs.

The exported graphs return vertex positions only. Two checkpoint buffers are
needed to draw and measure a body:

  * `head_pose.faces`                  (36874, 3) triangle indices
  * `head_pose.keypoint_mapping[:70]`  sparse (70, 18566) regressor over
    `[verts; joints]` (18439 vertices, then 127 joints), which is how the
    reference produces `pred_keypoints_3d`

Usage:
    python tools/onnx/dump_mesh.py \
        --ckpt models/flowhmr/sam/model.ckpt \
        --out models/sam3d \
        [--mhr70 path/to/sam_3d_body/metadata/mhr70.py]
"""
import argparse
import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import torch

MHR_ASSET = r'E:\RGBMocap\velomocap\models\flowhmr\sam\assets\mhr_model.pt'
VERTICES = 18439
FACES = 36874
KEYS = 70
TOTAL_POINTS = VERTICES + 127


def keypoint_names(mhr70):
    if mhr70 is None or not Path(mhr70).is_file():
        return []
    spec = importlib.util.spec_from_file_location('mhr70', str(mhr70))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.mhr_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mhr70', type=Path, default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    state = checkpoint.get('state_dict', checkpoint)

    faces = state['head_pose.faces'].to(torch.int64).numpy()
    if faces.shape != (FACES, 3) or faces.max() >= VERTICES:
        raise SystemExit('unexpected face buffer %s' % (faces.shape,))
    mapping = state['head_pose.keypoint_mapping'][:KEYS].float().numpy()
    if mapping.shape[1] != TOTAL_POINTS:
        raise SystemExit('unexpected regressor %s' % (mapping.shape,))

    rows, cols, values = [], [], []
    for row in range(mapping.shape[0]):
        for col in np.nonzero(mapping[row])[0]:
            rows.append(row)
            cols.append(int(col))
            values.append(float(mapping[row, col]))

    # The MHR skeleton itself: parent indices and parent-relative rest offsets.
    mhr = torch.jit.load(MHR_ASSET, map_location='cpu')
    parents = mhr.character_torch.skeleton.joint_parents.to(torch.int64).numpy()
    offsets = mhr.character_torch.skeleton.joint_translation_offsets.float().numpy()
    if parents.shape != (127,) or offsets.shape != (127, 3):
        raise SystemExit('unexpected MHR skeleton buffers')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'sam3d_skeleton.bin').write_bytes(
        struct.pack('<127i', *parents.astype(int).tolist())
        + offsets.astype('<f4').tobytes()
    )
    (args.out / 'sam3d_faces.bin').write_bytes(faces.astype('<u4').tobytes())
    (args.out / 'sam3d_keypoints.bin').write_bytes(
        struct.pack('<%di' % len(rows), *rows)
        + struct.pack('<%di' % len(cols), *cols)
        + struct.pack('<%df' % len(values), *values)
    )
    names = keypoint_names(args.mhr70)
    meta = {
        'vertices': VERTICES,
        'faces': int(faces.shape[0]),
        'face_file': 'sam3d_faces.bin',
        'joints': 127,
        'keypoints': KEYS,
        'keypoint_file': 'sam3d_keypoints.bin',
        'keypoint_nonzeros': len(rows),
        'dtype': 'uint32-le faces, int32-le (rows, cols) + f32 values',
        'skeleton_file': 'sam3d_skeleton.bin',
        'skeleton_joints': 127,
    }
    if names:
        meta['names'] = names
    (args.out / 'sam3d_mesh.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('vertices %d, faces %d, keypoints %d, nnz %d -> %s' % (
        VERTICES, faces.shape[0], KEYS, len(rows), args.out))


if __name__ == '__main__':
    main()
