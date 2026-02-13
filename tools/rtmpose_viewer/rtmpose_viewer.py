#!/usr/bin/env python3
"""Simple RTMPose skeleton viewer.

The script uses RTMPose + YOLOX from OpenMMLab (via rtmlib) and draws a
real-time skeleton overlay on webcam/video frames.
"""

from __future__ import annotations

import argparse
import ctypes
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


def build_nui_udp_payload(
    frame: Any,
    keypoints: Any,
    scores: Any,
    frame_index: int,
    score_threshold: float,
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
        for i in range(joint_count):
            x_px = float(player_keypoints[i][0])
            y_px = float(player_keypoints[i][1])
            confidence = float(player_scores[i])

            # Normalize to a camera-like space.
            x_norm = (x_px / max(frame_width, 1)) * 2.0 - 1.0
            y_norm = 1.0 - (y_px / max(frame_height, 1)) * 2.0
            z_norm = 1.0

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

    if args.nui_udp_target:
        nui_target = parse_udp_target(args.nui_udp_target)
        nui_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"NUI UDP streaming enabled: {nui_target[0]}:{nui_target[1]}")

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
                    frame, keypoints, scores, frame_count, args.nui_score_thr
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
        "--torch-home",
        default=None,
        help="Optional cache root for downloaded model files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

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
