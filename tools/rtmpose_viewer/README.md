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

Depth-model tuning (recommended for gesture-heavy titles):

```bash
python rtmpose_viewer.py --source 0 --mode balanced --nui-udp-target 127.0.0.1:37100 \
  --nui-torso-z-baseline 2.0 --nui-z-min 0.8 --nui-z-max 4.0 --nui-z-smoothing 0.35
```

Calibrated-scale depth (body-proportion anchors + camera model):

```bash
# FOV-based focal estimate (good default if you don't have camera calibration)
python rtmpose_viewer.py --source 0 --mode balanced --nui-udp-target 127.0.0.1:37100 \
  --nui-torso-z-baseline 2.0 --nui-z-min 0.8 --nui-z-max 4.0 --nui-z-smoothing 0.35 \
  --nui-camera-fov-deg 70 --nui-assumed-shoulder-width-m 0.41 --nui-assumed-height-m 1.70 \
  --nui-upper-arm-length-m 0.30 --nui-forearm-length-m 0.26

# Use calibrated focal length in pixels when available (overrides --nui-camera-fov-deg)
python rtmpose_viewer.py --source 0 --mode balanced --nui-udp-target 127.0.0.1:37100 \
  --nui-focal-length-px 920 --nui-assumed-shoulder-width-m 0.41 --nui-assumed-height-m 1.70
```

Run Xenia with NUI enabled and UDP bridge enabled:

```bash
# Example flags
--allow_nui_initialization=true --nui_sensor_udp_enabled=true --nui_sensor_udp_port=37100 --show_kinect_debug=true
```

For detailed game/NUI call tracing and a dedicated in-emulator log window:

```bash
--show_kinect_nui_log=true --log_kinect_nui_calls=true
```

To persist the same NUI/XAM call log to a separate file:

```bash
--nui_sensor_log_path=/tmp/xenia_nui_calls.log
```

If a title uses a different argument index for `XamNuiSkeletonScoreUpdate`
output buffers, adjust:

```bash
--nui_skeleton_score_output_arg_index=1
```

Depth safety and smoothing on the Xenia side can be tuned with:

```bash
--nui_skeleton_torso_baseline_z_m=2.0 --nui_skeleton_min_depth_m=0.8 --nui_skeleton_max_depth_m=4.0 --nui_skeleton_depth_smoothing=0.35
```

If the sender provides normalized camera joints (default RTMPose bridge), keep
`--nui_skeleton_input_normalized=true` and tune the 3D conversion with:

```bash
--nui_skeleton_input_normalized=true --nui_skeleton_camera_fov_x_deg=57 --nui_skeleton_camera_fov_y_deg=43
```

## Notes

- `--device auto` prefers CUDA only when CUDA libraries are loadable and at least one CUDA device is accessible, otherwise it falls back to CPU.
- Presets:
  - `lightweight`: fastest, lower accuracy
  - `balanced`: default
  - `performance`: highest accuracy, heavier
- You can override model URLs with `--det-model-url` and `--pose-model-url`.
- NUI depth model outputs per-joint `z` in meters with torso baseline + relative hand depth.
- Calibrated scale estimates torso depth from pose anchors:
  - shoulder-width anchor: `Z ~= f * W / w`
  - optional height anchor: nose-to-ankle pixel span
  - `f` comes from `--nui-focal-length-px` or from `--nui-camera-fov-deg`
- NUI UDP packet format (little-endian): `u32 magic='XNUI'`, `u16 version=1`, `u16 joint_count`, `u32 tracked_flag`, `u32 frame_index`, `u64 timestamp_us`, followed by `joint_count` entries of `f32 x, f32 y, f32 z, f32 confidence`. An optional tail of `u16 frame_width, u16 frame_height` may be appended.
