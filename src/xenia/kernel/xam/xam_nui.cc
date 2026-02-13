/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>

#include "third_party/imgui/imgui.h"
#include "xenia/base/logging.h"
#include "xenia/base/threading.h"
#include "xenia/emulator.h"
#include "xenia/kernel/kernel_flags.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/util/shim_utils.h"
#include "xenia/kernel/xam/xam_private.h"
#include "xenia/ui/imgui_dialog.h"
#include "xenia/ui/imgui_drawer.h"
#include "xenia/ui/window.h"
#include "xenia/ui/windowed_app_context.h"
#include "xenia/xbox.h"

#ifdef XE_PLATFORM_WIN32
// NOTE: must be included last as it expects windows.h to already be included.
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include <winsock2.h>  // NOLINT(build/include_order)
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

DEFINE_bool(allow_nui_initialization, false,
            "Enable Kinect/NUI initialization.\n"
            "Set this to true only when testing Kinect titles.",
            "Kernel");
DEFINE_bool(nui_sensor_udp_enabled, false,
            "Enable UDP Kinect sensor bridge (RTMPose sender -> Xenia).",
            "Kernel");
DEFINE_int32(nui_sensor_udp_port, 37100,
             "UDP port for the Kinect sensor bridge listener.", "Kernel");
DEFINE_int32(nui_sensor_frame_timeout_ms, 500,
             "How long a sensor frame stays valid before timing out.",
             "Kernel");
DEFINE_double(nui_sensor_min_joint_confidence, 0.25,
              "Minimum joint confidence that marks a frame as tracked.",
              "Kernel");
DEFINE_bool(
    show_kinect_debug, false,
    "Draw host-side Kinect debug overlay (joints, bones, FPS, latency).",
    "Kernel");

namespace xe {
namespace kernel {
namespace xam {

extern std::atomic<int> xam_dialogs_shown_;

struct X_NUI_DEVICE_STATUS {
  xe::be<uint32_t> unk0;
  xe::be<uint32_t> unk1;
  xe::be<uint32_t> unk2;
  xe::be<uint32_t> status;
  xe::be<uint32_t> unk4;
  xe::be<uint32_t> unk5;
};
static_assert(sizeof(X_NUI_DEVICE_STATUS) == 24, "Size matters");

namespace {

constexpr uint32_t kNuiUdpFrameMagic = 0x584E5549;  // "XNUI"
constexpr uint16_t kNuiUdpFrameVersion = 1;
constexpr size_t kNuiUdpHeaderSize = 24;
constexpr size_t kNuiUdpJointSize = 16;
constexpr size_t kNuiMaxJoints = 32;

struct NuiJoint {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float confidence = 0.0f;
};

struct NuiDebugSnapshot {
  bool has_recent_frame = false;
  bool tracked = false;
  uint32_t frame_index = 0;
  uint64_t sensor_timestamp_us = 0;
  uint64_t host_receive_us = 0;
  float receive_fps = 0.0f;
  uint16_t joint_count = 0;
  std::array<NuiJoint, kNuiMaxJoints> joints = {};
};

// COCO-17 style topology, compatible with the default RTMPose output.
constexpr std::array<std::array<uint8_t, 2>, 18> kNuiCocoBones = {{
    {{0, 1}},
    {{0, 2}},
    {{1, 3}},
    {{2, 4}},
    {{0, 5}},
    {{0, 6}},
    {{5, 6}},
    {{5, 7}},
    {{7, 9}},
    {{6, 8}},
    {{8, 10}},
    {{5, 11}},
    {{6, 12}},
    {{11, 12}},
    {{11, 13}},
    {{13, 15}},
    {{12, 14}},
    {{14, 16}},
}};

#ifdef XE_PLATFORM_WIN32
using NuiSocket = SOCKET;
constexpr NuiSocket kInvalidNuiSocket = INVALID_SOCKET;
#else
using NuiSocket = int;
constexpr NuiSocket kInvalidNuiSocket = -1;
#endif

uint16_t ReadLE16(const uint8_t* src) {
  return static_cast<uint16_t>(src[0]) | (static_cast<uint16_t>(src[1]) << 8);
}

uint32_t ReadLE32(const uint8_t* src) {
  return static_cast<uint32_t>(src[0]) | (static_cast<uint32_t>(src[1]) << 8) |
         (static_cast<uint32_t>(src[2]) << 16) |
         (static_cast<uint32_t>(src[3]) << 24);
}

uint64_t ReadLE64(const uint8_t* src) {
  return static_cast<uint64_t>(ReadLE32(src)) |
         (static_cast<uint64_t>(ReadLE32(src + 4)) << 32);
}

float ReadLEFloat(const uint8_t* src) {
  uint32_t bits = ReadLE32(src);
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint64_t QuerySteadyMicros() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
}

void CloseNuiSocket(NuiSocket* sock) {
  if (!sock) {
    return;
  }
  if (*sock == kInvalidNuiSocket) {
    return;
  }
#ifdef XE_PLATFORM_WIN32
  closesocket(*sock);
#else
  close(*sock);
#endif
  *sock = kInvalidNuiSocket;
}

class NuiUdpSensorService {
 public:
  static NuiUdpSensorService& Get() {
    static NuiUdpSensorService service;
    return service;
  }

  ~NuiUdpSensorService() { Shutdown(); }

  void EnsureRunning() {
    if (!cvars::nui_sensor_udp_enabled) {
      return;
    }
    EnsureDebugOverlayRegistered();
    bool expected = false;
    if (!started_.compare_exchange_strong(expected, true)) {
      return;
    }
    stop_requested_.store(false, std::memory_order_relaxed);
    worker_ = std::thread([this]() { WorkerMain(); });
  }

  bool HasRecentFrame() const {
    if (!cvars::nui_sensor_udp_enabled) {
      return true;
    }

    uint64_t last_us = last_frame_host_us_.load(std::memory_order_relaxed);
    if (!last_us) {
      return false;
    }

    const uint64_t timeout_us =
        static_cast<uint64_t>(std::max(cvars::nui_sensor_frame_timeout_ms, 1)) *
        1000ull;
    return QuerySteadyMicros() - last_us <= timeout_us;
  }

  bool IsTracked() const {
    if (!HasRecentFrame()) {
      return false;
    }
    return tracked_.load(std::memory_order_relaxed);
  }

  NuiDebugSnapshot GetDebugSnapshot() const {
    NuiDebugSnapshot snapshot;
    snapshot.has_recent_frame = HasRecentFrame();
    snapshot.tracked = tracked_.load(std::memory_order_relaxed);
    snapshot.frame_index = last_frame_index_.load(std::memory_order_relaxed);
    snapshot.sensor_timestamp_us =
        last_sensor_timestamp_us_.load(std::memory_order_relaxed);
    snapshot.host_receive_us =
        last_frame_host_us_.load(std::memory_order_relaxed);
    snapshot.receive_fps = receive_fps_.load(std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(frame_mutex_);
    snapshot.joint_count = latest_joint_count_;
    if (snapshot.joint_count) {
      std::copy_n(latest_joints_.cbegin(), snapshot.joint_count,
                  snapshot.joints.begin());
    }
    return snapshot;
  }

 private:
  void EnsureDebugOverlayRegistered() {
    if (!cvars::show_kinect_debug || cvars::headless) {
      return;
    }

    bool expected = false;
    if (!debug_overlay_registered_.compare_exchange_strong(expected, true)) {
      return;
    }

    KernelState* ks = kernel_state();
    if (!ks) {
      debug_overlay_registered_.store(false, std::memory_order_relaxed);
      return;
    }
    const Emulator* emulator = ks->emulator();
    if (!emulator) {
      debug_overlay_registered_.store(false, std::memory_order_relaxed);
      return;
    }
    ui::Window* display_window = emulator->display_window();
    ui::ImGuiDrawer* imgui_drawer = emulator->imgui_drawer();
    if (!display_window || !imgui_drawer) {
      debug_overlay_registered_.store(false, std::memory_order_relaxed);
      return;
    }

    bool attached = display_window->app_context().CallInUIThreadSynchronous(
        [this, imgui_drawer]() {
          imgui_drawer->AddDrawCallback(this, [this,
                                               imgui_drawer](ImGuiIO& io) {
            if (!cvars::show_kinect_debug) {
              imgui_drawer->RemoveDrawCallback(this);
              debug_overlay_registered_.store(false, std::memory_order_relaxed);
              return;
            }
            DrawDebugOverlay(io);
          });
        });
    if (!attached) {
      debug_overlay_registered_.store(false, std::memory_order_relaxed);
    }
  }

  void DrawDebugOverlay(ImGuiIO& io) {
    if (!cvars::show_kinect_debug) {
      return;
    }

    NuiDebugSnapshot snapshot = GetDebugSnapshot();
    const uint64_t now_us = QuerySteadyMicros();

    float age_ms = 0.0f;
    if (snapshot.host_receive_us && now_us >= snapshot.host_receive_us) {
      age_ms = float(now_us - snapshot.host_receive_us) / 1000.0f;
    }

    float latency_ms = age_ms;
    if (snapshot.sensor_timestamp_us &&
        now_us >= snapshot.sensor_timestamp_us) {
      latency_ms = float(now_us - snapshot.sensor_timestamp_us) / 1000.0f;
    }

    ImGui::SetNextWindowPos(ImVec2(12.0f, 12.0f), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.60f);
    ImGui::Begin(
        "Kinect Debug Overlay", nullptr,
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
            ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);
    ImGui::Text("sensor=%s tracked=%s",
                snapshot.has_recent_frame ? "ready" : "no-frame",
                snapshot.tracked ? "yes" : "no");
    ImGui::Text("frame=%u joints=%u", snapshot.frame_index,
                uint32_t(snapshot.joint_count));
    ImGui::Text("rx_fps=%.1f", snapshot.receive_fps);
    ImGui::Text("latency=%.1f ms age=%.1f ms", latency_ms, age_ms);
    ImGui::End();

    if (!snapshot.has_recent_frame || !snapshot.joint_count ||
        !(io.DisplaySize.x > 0.0f) || !(io.DisplaySize.y > 0.0f)) {
      return;
    }

    const float display_width = io.DisplaySize.x;
    const float display_height = io.DisplaySize.y;
    const float confidence_threshold =
        float(cvars::nui_sensor_min_joint_confidence);
    ImDrawList* draw_list = ImGui::GetForegroundDrawList();
    if (!draw_list) {
      return;
    }

    auto joint_to_screen = [display_width,
                            display_height](const NuiJoint& joint) -> ImVec2 {
      // Sender currently emits normalized camera coordinates in [-1, 1].
      bool is_unit = joint.x >= 0.0f && joint.x <= 1.0f && joint.y >= 0.0f &&
                     joint.y <= 1.0f;
      float x = is_unit ? (joint.x * display_width)
                        : ((joint.x * 0.5f + 0.5f) * display_width);
      float y = is_unit ? (joint.y * display_height)
                        : ((0.5f - joint.y * 0.5f) * display_height);
      return ImVec2(x, y);
    };

    if (snapshot.joint_count >= 17) {
      for (const auto& bone : kNuiCocoBones) {
        const uint32_t a = bone[0];
        const uint32_t b = bone[1];
        if (a >= snapshot.joint_count || b >= snapshot.joint_count) {
          continue;
        }
        const NuiJoint& joint_a = snapshot.joints[a];
        const NuiJoint& joint_b = snapshot.joints[b];
        if (joint_a.confidence < confidence_threshold ||
            joint_b.confidence < confidence_threshold) {
          continue;
        }
        draw_list->AddLine(joint_to_screen(joint_a), joint_to_screen(joint_b),
                           IM_COL32(80, 220, 120, 220), 2.0f);
      }
    } else {
      for (uint32_t i = 1; i < snapshot.joint_count; ++i) {
        const NuiJoint& joint_a = snapshot.joints[i - 1];
        const NuiJoint& joint_b = snapshot.joints[i];
        if (joint_a.confidence < confidence_threshold ||
            joint_b.confidence < confidence_threshold) {
          continue;
        }
        draw_list->AddLine(joint_to_screen(joint_a), joint_to_screen(joint_b),
                           IM_COL32(80, 220, 120, 220), 1.5f);
      }
    }

    for (uint32_t i = 0; i < snapshot.joint_count; ++i) {
      const NuiJoint& joint = snapshot.joints[i];
      if (joint.confidence < confidence_threshold) {
        continue;
      }
      const float confidence = std::clamp(joint.confidence, 0.0f, 1.0f);
      const uint8_t red = uint8_t((1.0f - confidence) * 255.0f);
      const uint8_t green = uint8_t(confidence * 255.0f);
      draw_list->AddCircleFilled(joint_to_screen(joint), 4.0f,
                                 IM_COL32(red, green, 255, 255));
    }
  }

  void Shutdown() {
    if (!started_.load(std::memory_order_relaxed)) {
      return;
    }
    stop_requested_.store(true, std::memory_order_relaxed);
    if (worker_.joinable()) {
      worker_.join();
    }
    started_.store(false, std::memory_order_relaxed);
  }

  void WorkerMain() {
    xe::threading::set_name("xam_nui_sensor");

    NuiSocket sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock == kInvalidNuiSocket) {
      XELOGE("NUI sensor: failed to create UDP socket");
      return;
    }

    sockaddr_in bind_addr = {};
    bind_addr.sin_family = AF_INET;
    bind_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    bind_addr.sin_port =
        htons(static_cast<uint16_t>(cvars::nui_sensor_udp_port));

    if (bind(sock, reinterpret_cast<sockaddr*>(&bind_addr),
             sizeof(bind_addr)) != 0) {
      XELOGE("NUI sensor: failed to bind UDP port {}",
             cvars::nui_sensor_udp_port);
      CloseNuiSocket(&sock);
      return;
    }

    XELOGI("NUI sensor: listening on 127.0.0.1:{}", cvars::nui_sensor_udp_port);

    std::array<uint8_t, 4096> packet = {};
    while (!stop_requested_.load(std::memory_order_relaxed)) {
      fd_set read_set;
      FD_ZERO(&read_set);
      FD_SET(sock, &read_set);
      timeval timeout = {0, 100000};  // 100 ms
#ifdef XE_PLATFORM_WIN32
      int select_result = select(0, &read_set, nullptr, nullptr, &timeout);
#else
      int select_result =
          select(sock + 1, &read_set, nullptr, nullptr, &timeout);
#endif
      if (select_result <= 0) {
        continue;
      }

      sockaddr_in from_addr = {};
#ifdef XE_PLATFORM_WIN32
      int from_len = sizeof(from_addr);
#else
      socklen_t from_len = sizeof(from_addr);
#endif
      int recv_size =
          recvfrom(sock, reinterpret_cast<char*>(packet.data()),
                   static_cast<int>(packet.size()), 0,
                   reinterpret_cast<sockaddr*>(&from_addr), &from_len);
      if (recv_size <= 0) {
        continue;
      }

      ParseFrame(packet.data(), static_cast<size_t>(recv_size));
    }

    CloseNuiSocket(&sock);
  }

  void ParseFrame(const uint8_t* data, size_t size) {
    if (size < kNuiUdpHeaderSize) {
      return;
    }

    uint32_t magic = ReadLE32(data + 0);
    uint16_t version = ReadLE16(data + 4);
    uint16_t joint_count = ReadLE16(data + 6);
    uint32_t tracked_flag = ReadLE32(data + 8);
    uint32_t frame_index = ReadLE32(data + 12);
    uint64_t sensor_timestamp_us = ReadLE64(data + 16);

    if (magic != kNuiUdpFrameMagic || version != kNuiUdpFrameVersion) {
      return;
    }

    size_t expected_size =
        kNuiUdpHeaderSize + size_t(joint_count) * kNuiUdpJointSize;
    if (size < expected_size) {
      return;
    }

    const size_t capped_joint_count =
        std::min(size_t(joint_count), kNuiMaxJoints);
    std::array<NuiJoint, kNuiMaxJoints> joints = {};

    bool tracked = tracked_flag != 0;
    for (size_t i = 0; i < capped_joint_count; ++i) {
      const uint8_t* joint_ptr =
          data + kNuiUdpHeaderSize + i * kNuiUdpJointSize;
      NuiJoint joint;
      joint.x = ReadLEFloat(joint_ptr + 0);
      joint.y = ReadLEFloat(joint_ptr + 4);
      joint.z = ReadLEFloat(joint_ptr + 8);
      joint.confidence = ReadLEFloat(joint_ptr + 12);
      joints[i] = joint;
      if (!tracked &&
          joint.confidence >= float(cvars::nui_sensor_min_joint_confidence)) {
        tracked = true;
      }
    }

    const uint64_t host_receive_us = QuerySteadyMicros();
    const uint64_t previous_host_us =
        last_frame_host_us_.load(std::memory_order_relaxed);
    if (previous_host_us && host_receive_us > previous_host_us) {
      const uint64_t delta_us = host_receive_us - previous_host_us;
      if (delta_us < 5000000ull) {
        const float instant_fps = 1000000.0f / float(delta_us);
        const float previous_fps = receive_fps_.load(std::memory_order_relaxed);
        const float smoothed_fps =
            previous_fps > 0.0f ? previous_fps * 0.9f + instant_fps * 0.1f
                                : instant_fps;
        receive_fps_.store(smoothed_fps, std::memory_order_relaxed);
      }
    }

    {
      std::lock_guard<std::mutex> lock(frame_mutex_);
      latest_joint_count_ = uint16_t(capped_joint_count);
      if (capped_joint_count) {
        std::copy_n(joints.cbegin(), capped_joint_count,
                    latest_joints_.begin());
      }
    }
    tracked_.store(tracked, std::memory_order_relaxed);
    last_frame_index_.store(frame_index, std::memory_order_relaxed);
    last_sensor_timestamp_us_.store(sensor_timestamp_us,
                                    std::memory_order_relaxed);
    last_frame_host_us_.store(host_receive_us, std::memory_order_relaxed);
  }

  std::thread worker_;
  std::atomic<bool> started_ = false;
  std::atomic<bool> stop_requested_ = false;
  std::atomic<bool> debug_overlay_registered_ = false;
  std::atomic<bool> tracked_ = false;
  std::atomic<uint32_t> last_frame_index_ = 0;
  std::atomic<uint64_t> last_sensor_timestamp_us_ = 0;
  std::atomic<uint64_t> last_frame_host_us_ = 0;
  std::atomic<float> receive_fps_ = 0.0f;
  mutable std::mutex frame_mutex_;
  uint16_t latest_joint_count_ = 0;
  std::array<NuiJoint, kNuiMaxJoints> latest_joints_ = {};
};

bool IsNuiDeviceConnected() {
  if (!cvars::allow_nui_initialization) {
    return false;
  }
  if (!cvars::nui_sensor_udp_enabled) {
    return true;
  }
  auto& sensor = NuiUdpSensorService::Get();
  sensor.EnsureRunning();
  return sensor.HasRecentFrame();
}

bool IsNuiSkeletonTracked() {
  if (!cvars::allow_nui_initialization) {
    return false;
  }
  if (!cvars::nui_sensor_udp_enabled) {
    return true;
  }
  auto& sensor = NuiUdpSensorService::Get();
  sensor.EnsureRunning();
  return sensor.IsTracked();
}

uint32_t engaged_tracking_id = 0;
uint64_t nui_session_id = 0;

uint32_t XeXamNuiHudCheck(dword_t tracking_id) {
  if (!IsNuiDeviceConnected()) {
    return X_ERROR_ACCESS_DENIED;
  }
  engaged_tracking_id = tracking_id;
  return X_ERROR_SUCCESS;
}

}  // namespace

dword_result_t XamNuiGetDeviceStatus_entry(
    pointer_t<X_NUI_DEVICE_STATUS> status_ptr) {
  if (!status_ptr) {
    return X_E_INVALIDARG;
  }
  const bool connected = IsNuiDeviceConnected();
  status_ptr.Zero();
  status_ptr->status = connected ? 1u : 0u;
  return connected ? X_ERROR_SUCCESS : 0xC0050006;
}
DECLARE_XAM_EXPORT1(XamNuiGetDeviceStatus, kNone, kStub);

dword_result_t XamUserNuiGetUserIndex_entry(unknown_t unk, lpdword_t index) {
  return X_E_NO_SUCH_USER;
}
DECLARE_XAM_EXPORT1(XamUserNuiGetUserIndex, kNone, kStub);

dword_result_t XamUserNuiGetUserIndexForSignin_entry(lpdword_t index) {
  if (!index) {
    return X_E_INVALIDARG;
  }
  if (kernel_state()->user_profile()->signin_state()) {
    *index = 0;
    return X_E_SUCCESS;
  }
  return X_HRESULT_FROM_WIN32(X_ERROR_ACCESS_DENIED);
}
DECLARE_XAM_EXPORT1(XamUserNuiGetUserIndexForSignin, kNone, kImplemented);

dword_result_t XamUserNuiGetUserIndexForBind_entry(lpdword_t index) {
  return X_E_FAIL;
}
DECLARE_XAM_EXPORT1(XamUserNuiGetUserIndexForBind, kNone, kStub);

dword_result_t XamNuiGetDepthCalibration_entry(lpdword_t unk1) {
  return X_STATUS_NO_SUCH_FILE;
}
DECLARE_XAM_EXPORT1(XamNuiGetDepthCalibration, kNone, kStub);

qword_result_t XamNuiSkeletonGetBestSkeletonIndex_entry(int_t unk) {
  return IsNuiSkeletonTracked() ? 0ull : 0xFFFFFFFFFFFFFFFFull;
}
DECLARE_XAM_EXPORT1(XamNuiSkeletonGetBestSkeletonIndex, kNone, kImplemented);

dword_result_t XamNuiCameraTiltGetStatus_entry(lpvoid_t unk) {
  return IsNuiDeviceConnected() ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiCameraTiltGetStatus, kNone, kStub);

dword_result_t XamNuiCameraElevationGetAngle_entry(lpqword_t angle,
                                                   lpdword_t status) {
  if (angle) {
    *angle = 0;
  }
  if (status) {
    *status = 0;
  }
  return IsNuiDeviceConnected() ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiCameraElevationGetAngle, kNone, kStub);

dword_result_t XamNuiCameraGetTiltControllerType_entry() {
  return IsNuiDeviceConnected() ? 1u : 0u;
}
DECLARE_XAM_EXPORT1(XamNuiCameraGetTiltControllerType, kNone, kStub);

dword_result_t XamNuiCameraSetFlags_entry(qword_t unk1, dword_t unk2) {
  return IsNuiDeviceConnected() ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiCameraSetFlags, kNone, kStub);

dword_result_t XamIsNuiUIActive_entry() { return xam_dialogs_shown_ > 0; }
DECLARE_XAM_EXPORT1(XamIsNuiUIActive, kNone, kImplemented);

dword_result_t XamNuiIsDeviceReady_entry() { return IsNuiDeviceConnected(); }
DECLARE_XAM_EXPORT1(XamNuiIsDeviceReady, kNone, kImplemented);

dword_result_t XamIsNuiAutomationEnabled_entry(unknown_t unk1, unknown_t unk2) {
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT2(XamIsNuiAutomationEnabled, kNone, kStub, kHighFrequency);

dword_result_t XamIsNatalPlaybackEnabled_entry(unknown_t unk1, unknown_t unk2) {
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT2(XamIsNatalPlaybackEnabled, kNone, kStub, kHighFrequency);

dword_result_t XamNuiIsChatMicEnabled_entry() { return false; }
DECLARE_XAM_EXPORT1(XamNuiIsChatMicEnabled, kNone, kImplemented);

dword_result_t XamNuiHudSetEngagedTrackingID_entry(dword_t tracking_id) {
  engaged_tracking_id = tracking_id;
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamNuiHudSetEngagedTrackingID, kNone, kImplemented);

qword_result_t XamNuiHudGetEngagedTrackingID_entry() {
  return engaged_tracking_id;
}
DECLARE_XAM_EXPORT1(XamNuiHudGetEngagedTrackingID, kNone, kImplemented);

dword_result_t XamNuiHudIsEnabled_entry() { return IsNuiDeviceConnected(); }
DECLARE_XAM_EXPORT1(XamNuiHudIsEnabled, kNone, kImplemented);

dword_result_t XamNuiHudGetInitializeFlags_entry() { return 0; }
DECLARE_XAM_EXPORT1(XamNuiHudGetInitializeFlags, kNone, kImplemented);

void XamNuiHudGetVersions_entry(lpqword_t unk1, lpqword_t unk2) {
  if (unk1) {
    *unk1 = 0;
  }
  if (unk2) {
    *unk2 = 0;
  }
}
DECLARE_XAM_EXPORT1(XamNuiHudGetVersions, kNone, kImplemented);

dword_result_t XamShowNuiTroubleshooterUI_entry(unknown_t unk1, unknown_t unk2,
                                                unknown_t unk3) {
  if (cvars::headless) {
    return X_ERROR_SUCCESS;
  }

  const Emulator* emulator = kernel_state()->emulator();
  ui::Window* display_window = emulator->display_window();
  ui::ImGuiDrawer* imgui_drawer = emulator->imgui_drawer();
  if (display_window && imgui_drawer) {
    xe::threading::Fence fence;
    if (display_window->app_context().CallInUIThreadSynchronous([&]() {
          xe::ui::ImGuiDialog::ShowMessageBox(
              imgui_drawer, "NUI Troubleshooter",
              "The game has indicated there is a problem with NUI (Kinect).")
              ->Then(&fence);
        })) {
      xam_dialogs_shown_++;
      fence.Wait();
      xam_dialogs_shown_--;
    }
  }

  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamShowNuiTroubleshooterUI, kNone, kStub);

dword_result_t XamShowNuiHardwareRequiredUI_entry(unknown_t unk1) {
  if (unk1 != 0) {
    return X_ERROR_INVALID_PARAMETER;
  }
  return XamShowNuiTroubleshooterUI_entry(0xFF, 0, 0x400000);
}
DECLARE_XAM_EXPORT1(XamShowNuiHardwareRequiredUI, kNone, kImplemented);

dword_result_t XamShowNuiGuideUI_entry(unknown_t unk1, unknown_t unk2) {
  return XeXamNuiHudCheck(0);
}
DECLARE_XAM_EXPORT1(XamShowNuiGuideUI, kNone, kStub);

qword_result_t XamNuiIdentityGetSessionId_entry() {
  if (!nui_session_id) {
    nui_session_id = 0xDEADF00DDEADF00Dull;
  }
  return nui_session_id;
}
DECLARE_XAM_EXPORT1(XamNuiIdentityGetSessionId, kNone, kImplemented);

dword_result_t XamNuiIdentityEnrollForSignIn_entry(dword_t unk1, qword_t unk2,
                                                   qword_t unk3, dword_t unk4) {
  return XamNuiHudIsEnabled_entry() ? X_ERROR_SUCCESS : X_E_FAIL;
}
DECLARE_XAM_EXPORT1(XamNuiIdentityEnrollForSignIn, kNone, kStub);

dword_result_t XamNuiIdentityAbort_entry(dword_t unk) {
  return XamNuiHudIsEnabled_entry() ? X_ERROR_SUCCESS : X_E_FAIL;
}
DECLARE_XAM_EXPORT1(XamNuiIdentityAbort, kNone, kStub);

dword_result_t XamUserNuiEnableBiometric_entry(dword_t user_index,
                                               int_t enable) {
  return X_E_INVALIDARG;
}
DECLARE_XAM_EXPORT1(XamUserNuiEnableBiometric, kNone, kStub);

void XamNuiPlayerEngagementUpdate_entry(qword_t unk1, unknown_t unk2,
                                        lpunknown_t unk3) {}
DECLARE_XAM_EXPORT1(XamNuiPlayerEngagementUpdate, kNone, kStub);

}  // namespace xam
}  // namespace kernel
}  // namespace xe

DECLARE_XAM_EMPTY_REGISTER_EXPORTS(NUI);
