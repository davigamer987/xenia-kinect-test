#!/usr/bin/env python3
"""Simple RTMPose skeleton viewer.

The script uses RTMPose + YOLOX from OpenMMLab (via rtmlib) and draws a
real-time skeleton overlay on webcam/video frames.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
from pathlib import Path
import socket
import site
import struct
import sys
import time
from typing import Any

DEFAULT_MODELS = {
    "performance": {
        "det": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip"
        ),
        "det_input_size": (640, 640),
        "pose": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-"
            "71d7b7e9_20230629.zip"
        ),
        "pose_input_size": (288, 384),
    },
    "balanced": {
        "det": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
        ),
        "det_input_size": (640, 640),
        "pose": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-"
            "e48f03d0_20230504.zip"
        ),
        "pose_input_size": (192, 256),
    },
    "lightweight": {
        "det": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip"
        ),
        "det_input_size": (416, 416),
        "pose": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-"
            "acd4a1ef_20230504.zip"
        ),
        "pose_input_size": (192, 256),
    },
}

NUI_UDP_MAGIC = 0x584E5549  # "XNUI"
NUI_UDP_VERSION = 1


def parse_source(raw_source: str) -> int | str:
    if raw_source.isdigit():
        return int(raw_source)
    return raw_source


def parse_udp_target(raw_target: str) -> tuple[str, int]:
    if ":" not in raw_target:
        raise ValueError("Expected HOST:PORT for --nui-udp-target.")
    host, port_text = raw_target.rsplit(":", 1)
    if not host:
        raise ValueError("Missing host in --nui-udp-target.")
    port = int(port_text)
    if port < 1 or port > 65535:
        raise ValueError("Port must be in the range 1..65535.")
    return host, port


class NuiDepthEstimator:
    """Calibrated depth model for COCO-17 joints from monocular 2D pose."""

    def __init__(
        self,
        torso_z_baseline: float,
        z_min: float,
        z_max: float,
        smoothing: float,
        assumed_shoulder_width_m: float,
        assumed_height_m: float,
        camera_fov_deg: float,
        focal_length_px: float,
        upper_arm_length_m: float,
        forearm_length_m: float,
    ) -> None:
        self.torso_z_baseline = max(float(torso_z_baseline), 0.1)
        self.z_min = max(float(z_min), 0.1)
        self.z_max = max(float(z_max), self.z_min + 0.1)
        self.smoothing = self._clamp(float(smoothing), 0.0, 1.0)
        self.assumed_shoulder_width_m = max(float(assumed_shoulder_width_m), 0.1)
        self.assumed_height_m = max(float(assumed_height_m), 0.8)
        self.camera_fov_deg = self._clamp(float(camera_fov_deg), 30.0, 140.0)
        self.focal_length_px = max(float(focal_length_px), 0.0)
        self.upper_arm_length_m = max(float(upper_arm_length_m), 0.05)
        self.forearm_length_m = max(float(forearm_length_m), 0.05)
        self._smoothed_torso_z = self.torso_z_baseline
        self._torso_initialized = False
        self._smoothed_depths = [self.torso_z_baseline] * 32
        self._initialized = [False] * 32

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _joint(
        keypoints: Any,
        scores: Any,
        index: int,
    ) -> tuple[float, float, float] | None:
        if index >= len(keypoints) or index >= len(scores):
            return None
        return float(keypoints[index][0]), float(keypoints[index][1]), float(scores[index])

    def _focal_px(self, frame_width: int) -> float:
        if self.focal_length_px > 1.0:
            return self.focal_length_px
        fov_rad = math.radians(self.camera_fov_deg)
        return (float(frame_width) * 0.5) / max(math.tan(fov_rad * 0.5), 1e-6)

    def _estimate_torso_z(
        self,
        keypoints: Any,
        scores: Any,
        frame_width: int,
        confidence_floor: float,
    ) -> float:
        f_px = self._focal_px(frame_width)
        anchors: list[tuple[float, float]] = []

        left_shoulder = self._joint(keypoints, scores, 5)
        right_shoulder = self._joint(keypoints, scores, 6)
        if left_shoulder and right_shoulder:
            shoulder_conf = min(left_shoulder[2], right_shoulder[2])
            shoulder_px = self._distance(left_shoulder, right_shoulder)
            if shoulder_conf >= confidence_floor and shoulder_px >= 12.0:
                z_shoulder = f_px * self.assumed_shoulder_width_m / max(shoulder_px, 1e-6)
                anchors.append((z_shoulder, shoulder_conf))

        nose = self._joint(keypoints, scores, 0)
        left_ankle = self._joint(keypoints, scores, 15)
        right_ankle = self._joint(keypoints, scores, 16)
        ankle_samples = [j for j in (left_ankle, right_ankle) if j and j[2] >= confidence_floor]
        if nose and nose[2] >= confidence_floor and ankle_samples:
            ankle_y = sum(j[1] for j in ankle_samples) / len(ankle_samples)
            body_px = abs(ankle_y - nose[1])
            if body_px >= 40.0:
                height_conf = min(nose[2], sum(j[2] for j in ankle_samples) / len(ankle_samples))
                z_height = f_px * self.assumed_height_m / max(body_px, 1e-6)
                anchors.append((z_height, height_conf * 0.7))

        if not anchors:
            torso_z = self.torso_z_baseline
        else:
            total_weight = sum(w for _, w in anchors)
            if total_weight <= 1e-6:
                torso_z = self.torso_z_baseline
            else:
                torso_z = sum(z * w for z, w in anchors) / total_weight

        torso_z = self._clamp(torso_z, self.z_min, self.z_max)
        if self._torso_initialized:
            torso_z = self._smoothed_torso_z * (1.0 - self.smoothing) + torso_z * self.smoothing
        self._smoothed_torso_z = torso_z
        self._torso_initialized = True
        return torso_z

    def estimate(self, keypoints: Any, scores: Any, frame_width: int, frame_height: int) -> list[float]:
        joint_count = min(len(keypoints), len(scores), 32)
        if joint_count <= 0:
            return []

        confidence_floor = 0.15
        torso_z = self._estimate_torso_z(keypoints, scores, frame_width, confidence_floor)
        f_px = self._focal_px(frame_width)

        raw_depths = [torso_z] * joint_count

        def set_depth(index: int, value: float) -> None:
            if index < joint_count:
                raw_depths[index] = value

        set_depth(0, torso_z - 0.08)  # head / nose
        for idx in (11, 12):
            set_depth(idx, torso_z + 0.06)
        for idx in (13, 14):
            set_depth(idx, torso_z + 0.14)
        for idx in (15, 16):
            set_depth(idx, torso_z + 0.22)

        def apply_arm_depth(shoulder_idx: int, elbow_idx: int, wrist_idx: int) -> None:
            shoulder = self._joint(keypoints, scores, shoulder_idx)
            elbow = self._joint(keypoints, scores, elbow_idx)
            wrist = self._joint(keypoints, scores, wrist_idx)
            if not shoulder or not wrist:
                return
            if shoulder[2] < confidence_floor or wrist[2] < confidence_floor:
                return

            confidence_gate = min(shoulder[2], wrist[2], 1.0)

            def segment_forward(
                a: tuple[float, float, float] | None,
                b: tuple[float, float, float] | None,
                length_m: float,
            ) -> float:
                if not a or not b:
                    return 0.0
                conf = min(a[2], b[2])
                if conf < confidence_floor:
                    return 0.0
                projected_px = self._distance(a, b)
                transverse_m = torso_z * projected_px / max(f_px, 1e-6)
                transverse_m = min(transverse_m, length_m)
                forward_m = math.sqrt(max(length_m * length_m - transverse_m * transverse_m, 0.0))
                conf_scale = self._clamp((conf - confidence_floor) / 0.65, 0.0, 1.0)
                return forward_m * conf_scale

            upper_forward = segment_forward(shoulder, elbow, self.upper_arm_length_m)
            fore_forward = segment_forward(elbow, wrist, self.forearm_length_m)
            forward_total = upper_forward + fore_forward
            confidence_scale = self._clamp((confidence_gate - confidence_floor) / 0.65, 0.0, 1.0)

            if elbow_idx < joint_count:
                raw_depths[elbow_idx] = torso_z - upper_forward
            if wrist_idx < joint_count:
                raw_depths[wrist_idx] = torso_z - forward_total * confidence_scale

        apply_arm_depth(5, 7, 9)
        apply_arm_depth(6, 8, 10)

        smoothed_depths: list[float] = []
        for i in range(joint_count):
            depth = self._clamp(raw_depths[i], self.z_min, self.z_max)
            if self._initialized[i]:
                depth = self._smoothed_depths[i] * (1.0 - self.smoothing) + depth * self.smoothing
            self._smoothed_depths[i] = depth
            self._initialized[i] = True
            smoothed_depths.append(depth)
        return smoothed_depths


def build_nui_udp_payload(
    frame: Any,
    keypoints: Any,
    scores: Any,
    frame_index: int,
    score_threshold: float,
    depth_estimator: NuiDepthEstimator | None,
) -> bytes:
    frame_height, frame_width = frame.shape[:2]

    joint_tuples: list[tuple[float, float, float, float]] = []
    tracked_flag = 0

    if len(keypoints) > 0 and len(scores) > 0:
        best_player = 0
        best_score = -1.0
        for i in range(len(scores)):
            row = scores[i]
            if len(row) == 0:
                continue
            avg = float(sum(float(v) for v in row) / len(row))
            if avg > best_score:
                best_score = avg
                best_player = i

        player_keypoints = keypoints[best_player]
        player_scores = scores[best_player]
        joint_count = min(len(player_keypoints), len(player_scores), 32)
        depth_meters = (
            depth_estimator.estimate(
                player_keypoints, player_scores, frame_width, frame_height
            )
            if depth_estimator is not None
            else [2.0] * joint_count
        )
        for i in range(joint_count):
            x_px = float(player_keypoints[i][0])
            y_px = float(player_keypoints[i][1])
            confidence = float(player_scores[i])

            # Normalize to a camera-like space.
            x_norm = (x_px / max(frame_width, 1)) * 2.0 - 1.0
            y_norm = 1.0 - (y_px / max(frame_height, 1)) * 2.0
            z_norm = float(depth_meters[i]) if i < len(depth_meters) else 2.0

            if confidence >= score_threshold:
                tracked_flag = 1
            joint_tuples.append((x_norm, y_norm, z_norm, confidence))

    timestamp_us = int(time.monotonic_ns() // 1000)
    payload = bytearray(
        struct.pack(
            "<IHHIIQ",
            NUI_UDP_MAGIC,
            NUI_UDP_VERSION,
            len(joint_tuples),
            tracked_flag,
            frame_index,
            timestamp_us,
        )
    )

    for x_norm, y_norm, z_norm, confidence in joint_tuples:
        payload.extend(struct.pack("<ffff", x_norm, y_norm, z_norm, confidence))

    # Optional tail extension for consumers that need source aspect correction.
    frame_width_u16 = max(0, min(int(frame_width), 0xFFFF))
    frame_height_u16 = max(0, min(int(frame_height), 0xFFFF))
    payload.extend(struct.pack("<HH", frame_width_u16, frame_height_u16))

    return bytes(payload)


def _shared_lib_mode() -> int:
    return getattr(os, "RTLD_NOW", 0) | getattr(os, "RTLD_GLOBAL", 0)


def _nvidia_wheel_roots() -> list[Path]:
    roots: list[Path] = []
    for raw in site.getsitepackages() + [site.getusersitepackages()]:
        candidate = Path(raw)
        if candidate.exists():
            roots.append(candidate)
    # Preserve order while deduplicating.
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root)
    return deduped


def _maybe_preload_nvidia_wheel_cuda_libs() -> None:
    # Make CUDA 12 / cuDNN 9 pip wheels usable without manual LD_LIBRARY_PATH.
    if not sys.platform.startswith("linux"):
        return

    relative_libs = [
        "nvidia/cuda_runtime/lib/libcudart.so.12",
        "nvidia/cublas/lib/libcublasLt.so.12",
        "nvidia/cublas/lib/libcublas.so.12",
        "nvidia/cudnn/lib/libcudnn.so.9",
        "nvidia/cufft/lib/libcufft.so.11",
        "nvidia/curand/lib/libcurand.so.10",
        "nvidia/nvjitlink/lib/libnvJitLink.so.12",
    ]

    for rel in relative_libs:
        for root in _nvidia_wheel_roots():
            candidate = root / rel
            if not candidate.exists():
                continue
            ctypes.CDLL(str(candidate), mode=_shared_lib_mode())
            break


def _cuda_device_accessible() -> tuple[bool, str | None]:
    try:
        cudart = ctypes.CDLL("libcudart.so.12", mode=_shared_lib_mode())
    except OSError:
        try:
            cudart = ctypes.CDLL("libcudart.so", mode=_shared_lib_mode())
        except OSError as exc:
            return False, str(exc)

    count = ctypes.c_int(-1)
    rc = cudart.cudaGetDeviceCount(ctypes.byref(count))
    if rc != 0:
        return False, f"cudaGetDeviceCount failed with code {rc}"
    if count.value < 1:
        return False, "No CUDA devices reported by runtime."
    return True, None


def _cuda_provider_runtime_ready() -> tuple[bool, str | None]:
    import onnxruntime as ort

    try:
        _maybe_preload_nvidia_wheel_cuda_libs()
    except OSError as exc:
        return False, str(exc)

    providers = set(ort.get_available_providers())
    if "CUDAExecutionProvider" not in providers:
        return False, "CUDAExecutionProvider is not exposed by onnxruntime."

    if sys.platform.startswith("linux"):
        cuda_provider_lib = Path(ort.__file__).resolve().parent / "capi" / "libonnxruntime_providers_cuda.so"
        if not cuda_provider_lib.exists():
            return False, f"Missing CUDA provider library: {cuda_provider_lib}"
        try:
            # Force dependency resolution now so auto-device can cleanly fallback.
            ctypes.CDLL(str(cuda_provider_lib), mode=_shared_lib_mode())
        except OSError as exc:
            return False, str(exc)

    cuda_ok, reason = _cuda_device_accessible()
    if not cuda_ok:
        return False, reason

    return True, None


def resolve_device(device_arg: str) -> tuple[str, str | None]:
    if device_arg == "cpu":
        return "cpu", None

    if device_arg == "cuda":
        cuda_ok, reason = _cuda_provider_runtime_ready()
        if not cuda_ok:
            raise RuntimeError(f"CUDA requested, but CUDA execution is not usable: {reason}")
        return "cuda", None

    cuda_ok, reason = _cuda_provider_runtime_ready()
    if cuda_ok:
        return "cuda", None
    return "cpu", reason


def make_body_estimator(args: argparse.Namespace):
    from rtmlib import Body

    mode = args.mode
    model_cfg = DEFAULT_MODELS[mode]

    det_url = args.det_model_url or model_cfg["det"]
    pose_url = args.pose_model_url or model_cfg["pose"]

    det_input_size = tuple(args.det_input_size) if args.det_input_size else model_cfg[
        "det_input_size"
    ]
    pose_input_size = (
        tuple(args.pose_input_size) if args.pose_input_size else model_cfg["pose_input_size"]
    )

    return Body(
        det=det_url,
        det_input_size=det_input_size,
        pose=pose_url,
        pose_input_size=pose_input_size,
        mode=mode,
        to_openpose=args.openpose_skeleton,
        backend="onnxruntime",
        device=args.runtime_device,
    )


def open_capture(source: int | str):
    import cv2

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {source!r}")
    return capture


def create_video_writer(output_path: str, capture):
    import cv2

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer: {output_path!r}")
    return writer


def overlay_status(frame: Any, device: str, fps: float) -> None:
    import cv2

    text = f"device={device} fps={fps:5.1f} (q/esc to quit)"
    cv2.putText(
        frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (40, 255, 40),
        2,
        cv2.LINE_AA,
    )


def run_viewer(args: argparse.Namespace) -> int:
    import cv2
    from rtmlib import draw_skeleton

    pose = make_body_estimator(args)
    if args.download_only:
        print("Model files are ready.")
        return 0

    source = parse_source(args.source)
    capture = open_capture(source)
    writer = create_video_writer(args.output, capture) if args.output else None
    nui_socket: socket.socket | None = None
    nui_target: tuple[str, int] | None = None
    nui_depth_estimator: NuiDepthEstimator | None = None

    if args.nui_udp_target:
        nui_target = parse_udp_target(args.nui_udp_target)
        nui_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        nui_depth_estimator = NuiDepthEstimator(
            torso_z_baseline=args.nui_torso_z_baseline,
            z_min=args.nui_z_min,
            z_max=args.nui_z_max,
            smoothing=args.nui_z_smoothing,
            assumed_shoulder_width_m=args.nui_assumed_shoulder_width_m,
            assumed_height_m=args.nui_assumed_height_m,
            camera_fov_deg=args.nui_camera_fov_deg,
            focal_length_px=args.nui_focal_length_px,
            upper_arm_length_m=args.nui_upper_arm_length_m,
            forearm_length_m=args.nui_forearm_length_m,
        )
        print(f"NUI UDP streaming enabled: {nui_target[0]}:{nui_target[1]}")
        print(
            "NUI depth model enabled: "
            f"torso_z={args.nui_torso_z_baseline:.2f}m "
            f"range=[{args.nui_z_min:.2f},{args.nui_z_max:.2f}] "
            f"smoothing={args.nui_z_smoothing:.2f}"
        )
        if args.nui_focal_length_px > 0:
            print(
                "NUI calibrated scale: "
                f"focal_px={args.nui_focal_length_px:.1f} "
                f"shoulder={args.nui_assumed_shoulder_width_m:.3f}m "
                f"height={args.nui_assumed_height_m:.3f}m"
            )
        else:
            print(
                "NUI calibrated scale: "
                f"fov={args.nui_camera_fov_deg:.1f}deg "
                f"shoulder={args.nui_assumed_shoulder_width_m:.3f}m "
                f"height={args.nui_assumed_height_m:.3f}m"
            )

    frame_count = 0
    start_time = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            keypoints, scores = pose(frame)
            rendered = draw_skeleton(
                frame.copy(),
                keypoints,
                scores,
                openpose_skeleton=args.openpose_skeleton,
                kpt_thr=args.kpt_thr,
                radius=args.radius,
                line_width=args.line_width,
            )

            frame_count += 1
            elapsed = max(time.perf_counter() - start_time, 1e-6)
            fps = frame_count / elapsed
            overlay_status(rendered, args.runtime_device, fps)

            if writer is not None:
                writer.write(rendered)

            if nui_socket is not None and nui_target is not None:
                packet = build_nui_udp_payload(
                    frame,
                    keypoints,
                    scores,
                    frame_count,
                    args.nui_score_thr,
                    nui_depth_estimator,
                )
                nui_socket.sendto(packet, nui_target)

            cv2.imshow(args.window_name, rendered)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if nui_socket is not None:
            nui_socket.close()
        cv2.destroyAllWindows()

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RTMPose skeleton viewer (webcam/video).")
    parser.add_argument("--source", default="0", help="Video source index or file path.")
    parser.add_argument(
        "--mode",
        choices=("performance", "balanced", "lightweight"),
        default="balanced",
        help="Model preset for detector + pose.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Runtime device selection for ONNX Runtime.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download model files and exit without opening a camera.",
    )
    parser.add_argument(
        "--det-model-url",
        default=None,
        help="Override detector model URL/path (YOLOX ONNX zip/onnx).",
    )
    parser.add_argument(
        "--pose-model-url",
        default=None,
        help="Override pose model URL/path (RTMPose ONNX zip/onnx).",
    )
    parser.add_argument(
        "--det-input-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("W", "H"),
        help="Detector input size override.",
    )
    parser.add_argument(
        "--pose-input-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("W", "H"),
        help="Pose input size override.",
    )
    parser.add_argument("--kpt-thr", type=float, default=0.43, help="Keypoint score threshold.")
    parser.add_argument("--radius", type=int, default=2, help="Keypoint circle radius.")
    parser.add_argument("--line-width", type=int, default=2, help="Skeleton line width.")
    parser.add_argument(
        "--openpose-skeleton",
        action="store_true",
        help="Render with OpenPose-style topology instead of mmpose-style.",
    )
    parser.add_argument("--window-name", default="RTMPose Viewer", help="Display window title.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output video path (e.g., output.mp4).",
    )
    parser.add_argument(
        "--nui-udp-target",
        default=None,
        help="Optional Kinect bridge target as HOST:PORT (for Xenia NUI UDP input).",
    )
    parser.add_argument(
        "--nui-score-thr",
        type=float,
        default=0.35,
        help="Joint confidence threshold used to mark the outgoing NUI frame as tracked.",
    )
    parser.add_argument(
        "--nui-torso-z-baseline",
        type=float,
        default=2.0,
        help="Baseline torso depth in meters used by the outgoing NUI depth model.",
    )
    parser.add_argument(
        "--nui-z-min",
        type=float,
        default=0.8,
        help="Minimum outgoing NUI joint depth in meters.",
    )
    parser.add_argument(
        "--nui-z-max",
        type=float,
        default=4.0,
        help="Maximum outgoing NUI joint depth in meters.",
    )
    parser.add_argument(
        "--nui-z-smoothing",
        type=float,
        default=0.35,
        help="Depth smoothing factor for outgoing NUI joints (0=no smoothing, 1=no history).",
    )
    parser.add_argument(
        "--nui-assumed-shoulder-width-m",
        type=float,
        default=0.41,
        help="Assumed real shoulder width in meters for calibrated scale.",
    )
    parser.add_argument(
        "--nui-assumed-height-m",
        type=float,
        default=1.70,
        help="Assumed real body height in meters for secondary scale anchor.",
    )
    parser.add_argument(
        "--nui-camera-fov-deg",
        type=float,
        default=70.0,
        help="Horizontal camera field-of-view in degrees (used when focal length is not set).",
    )
    parser.add_argument(
        "--nui-focal-length-px",
        type=float,
        default=0.0,
        help="Optional calibrated focal length in pixels. If >0, overrides --nui-camera-fov-deg.",
    )
    parser.add_argument(
        "--nui-upper-arm-length-m",
        type=float,
        default=0.30,
        help="Assumed upper arm length in meters for relative hand-depth model.",
    )
    parser.add_argument(
        "--nui-forearm-length-m",
        type=float,
        default=0.26,
        help="Assumed forearm length in meters for relative hand-depth model.",
    )
    parser.add_argument(
        "--torch-home",
        default=None,
        help="Optional cache root for downloaded model files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.nui_z_min >= args.nui_z_max:
        raise SystemExit("ERROR: --nui-z-min must be lower than --nui-z-max.")

    if args.torch_home:
        os.environ["TORCH_HOME"] = args.torch_home

    try:
        import onnxruntime as ort

        args.runtime_device, auto_reason = resolve_device(args.device)
        print(f"Using device: {args.runtime_device}")
        print(f"Available providers: {ort.get_available_providers()}")
        if args.device == "auto" and args.runtime_device == "cpu" and auto_reason:
            print(f"CUDA not usable, falling back to CPU: {auto_reason}")
        return run_viewer(args)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
