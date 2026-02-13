# RTMPose Skeleton Viewer

Minimal webcam/video skeleton viewer using RTMPose ONNX models from OpenMMLab.

## 1) Create environment

```bash
cd tools/rtmpose_viewer
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

For CUDA (NVIDIA), install the GPU runtime packages:

```bash
python -m pip install -r requirements-gpu.txt
```

`requirements-gpu.txt` includes ONNX Runtime GPU plus CUDA 12 / cuDNN 9 runtime wheels.

## 2) Download models

This prefetches the detector + RTMPose weights into your cache (`~/.cache/rtmlib/hub/checkpoints` by default):

```bash
python rtmpose_viewer.py --download-only --mode balanced
```

## 3) Run viewer

Webcam:

```bash
python rtmpose_viewer.py --source 0 --mode balanced --device auto
```

Video file:

```bash
python rtmpose_viewer.py --source /path/to/video.mp4 --mode balanced
```

Press `q` or `Esc` to exit.

## 4) Stream to Xenia Kinect Bridge

Run viewer with UDP output enabled:

```bash
python rtmpose_viewer.py --source 0 --mode balanced --nui-udp-target 127.0.0.1:37100
```

Run Xenia with NUI enabled and UDP bridge enabled:

```bash
# Example flags
--allow_nui_initialization=true --nui_sensor_udp_enabled=true --nui_sensor_udp_port=37100 --show_kinect_debug=true
```

## Notes

- `--device auto` prefers CUDA only when CUDA libraries are loadable and at least one CUDA device is accessible, otherwise it falls back to CPU.
- Presets:
  - `lightweight`: fastest, lower accuracy
  - `balanced`: default
  - `performance`: highest accuracy, heavier
- You can override model URLs with `--det-model-url` and `--pose-model-url`.
- NUI UDP packet format (little-endian): `u32 magic='XNUI'`, `u16 version=1`, `u16 joint_count`, `u32 tracked_flag`, `u32 frame_index`, `u64 timestamp_us`, followed by `joint_count` entries of `f32 x, f32 y, f32 z, f32 confidence`.
