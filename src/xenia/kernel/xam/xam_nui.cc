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
#include <cctype>
#include <cmath>
#include <deque>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "third_party/imgui/imgui.h"
#include "xenia/cpu/function.h"
#include "xenia/cpu/module.h"
#include "xenia/base/filesystem.h"
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
DEFINE_int32(
    nui_sensor_tracking_acquire_frames, 2,
    "Consecutive tracked sensor frames required before reporting tracked.",
    "Kernel");
DEFINE_int32(
    nui_sensor_tracking_hold_ms, 400,
    "How long to hold tracked state after temporary confidence drop.",
    "Kernel");
DEFINE_double(
    nui_sensor_source_aspect_ratio, 4.0 / 3.0,
    "Fallback RTMPose sender aspect ratio used by the Kinect debug overlay "
    "when frame dimensions are unavailable.",
    "Kernel");
DEFINE_bool(
    show_kinect_debug, false,
    "Draw host-side Kinect debug overlay (joints, bones, FPS, latency).",
    "Kernel");
DEFINE_bool(
    show_kinect_nui_log, false,
    "Draw a dedicated Kinect NUI call log window in the host UI.",
    "Kernel");
DEFINE_bool(
    log_kinect_nui_calls, true,
    "Write Kinect/NUI bridge call details to xenia.log.",
    "Kernel");
DEFINE_path(
    nui_sensor_log_path, "",
    "Optional host path for a dedicated Kinect/NUI call log file.",
    "Kernel");
DEFINE_int32(
    nui_sensor_log_frame_interval, 30,
    "How often to log received UDP skeleton frames (in frame count).",
    "Kernel");
DEFINE_int32(
    nui_skeleton_score_output_arg_index, 1,
    "Argument index treated as an output frame pointer in "
    "XamNuiSkeletonScoreUpdate (-1 disables writing).",
    "Kernel");
DEFINE_int32(
    nui_device_status_value, 2,
    "Connected status value reported by XamNuiGetDeviceStatus.",
    "Kernel");
DEFINE_int32(
    nui_hud_initialize_flags, 0x9,
    "Initialize flags reported by XamNuiHudGetInitializeFlags (default enables "
    "depth + skeleton style paths).",
    "Kernel");
DEFINE_bool(
    nui_chat_mic_enabled, false,
    "Reported return value for XamNuiIsChatMicEnabled.",
    "Kernel");
DEFINE_bool(
    nui_trace_all_xam_import_calls, true,
    "Trace all xam extern dispatches that look Kinect-related (NUI/Natal) "
    "and append them to the Kinect NUI log.",
    "Kernel");
DEFINE_bool(
    nui_trace_all_xam_calls, true,
    "Trace every xam extern dispatch/undefined call into the Kinect NUI log "
    "(very noisy, useful for call-path debugging).",
    "Kernel");
DEFINE_double(
    nui_skeleton_torso_baseline_z_m, 2.0,
    "Fallback torso depth (meters) used when incoming joint depth is invalid.",
    "Kernel");
DEFINE_double(
    nui_skeleton_min_depth_m, 0.8,
    "Minimum accepted Kinect skeleton depth in meters.",
    "Kernel");
DEFINE_double(
    nui_skeleton_max_depth_m, 4.0,
    "Maximum accepted Kinect skeleton depth in meters.",
    "Kernel");
DEFINE_double(
    nui_skeleton_depth_smoothing, 0.35,
    "Depth smoothing factor for incoming Kinect joints (0=no smoothing, 1=no history).",
    "Kernel");
DEFINE_bool(
    nui_skeleton_input_normalized, true,
    "Interpret incoming UDP joint x/y as normalized camera coordinates in "
    "[-1, 1] and convert them to Kinect-like meters.",
    "Kernel");
DEFINE_double(
    nui_skeleton_camera_fov_x_deg, 57.0,
    "Horizontal FOV used to convert normalized joint x into meters.",
    "Kernel");
DEFINE_double(
    nui_skeleton_camera_fov_y_deg, 43.0,
    "Vertical FOV used to convert normalized joint y into meters.",
    "Kernel");
DEFINE_int32(
    nui_sensor_color_output_width, 640,
    "Target color texture width presented to XAM NUI camera consumers.",
    "Kernel");
DEFINE_int32(
    nui_sensor_color_output_height, 480,
    "Target color texture height presented to XAM NUI camera consumers.",
    "Kernel");
DEFINE_int32(
    nui_sensor_color_format_fourcc, 0x32595559,
    "Color format FourCC reported by XAM NUI camera info (default YUY2).",
    "Kernel");
DEFINE_int32(
    nui_sensor_color_nominal_fps, 30,
    "Nominal RGB camera FPS reported by XAM NUI camera info.",
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
constexpr size_t kNuiUdpFrameSizeTailSize = 4;
constexpr size_t kNuiUdpMaxPacketSize = 65536;
constexpr uint32_t kNuiUdpColorMagic = 0x31424752;  // "RGB1"
constexpr size_t kNuiUdpColorHeaderSize = 16;
constexpr uint8_t kNuiUdpColorFormatRgb24 = 1;
constexpr size_t kNuiUdpMaxColorPayloadSize = 60000;
constexpr uint32_t kNuiColorFormatYUY2 = 0x32595559;  // "YUY2"
constexpr uint32_t kNuiColorFormatRGB3 = 0x33424752;  // "RGB3"
constexpr uint32_t kNuiColorFormatBGRX = 0x58524742;  // "BGRX"
constexpr uint32_t kNuiBridgeRevision = 3;
constexpr float kPi = 3.14159265358979323846f;

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
  uint16_t frame_width = 0;
  uint16_t frame_height = 0;
  uint16_t joint_count = 0;
  std::array<NuiJoint, kNuiMaxJoints> joints = {};
};

struct NuiColorFrame {
  uint16_t width = 0;
  uint16_t height = 0;
  uint32_t frame_index = 0;
  uint64_t sensor_timestamp_us = 0;
  uint64_t host_receive_us = 0;
  std::vector<uint8_t> rgb24 = {};
};

constexpr size_t kNuiLogLineLimit = 256;
constexpr uint32_t kNuiSkeletonCount = 6;
constexpr uint32_t kNuiSkeletonPositionCount = 20;

enum NuiSkeletonTrackingState : uint32_t {
  kNuiSkeletonNotTracked = 0,
  kNuiSkeletonPositionOnly = 1,
  kNuiSkeletonTracked = 2,
};

enum NuiSkeletonPositionTrackingState : uint32_t {
  kNuiSkeletonPositionNotTracked = 0,
  kNuiSkeletonPositionInferred = 1,
  kNuiSkeletonPositionTracked = 2,
};

enum NuiSkeletonPositionIndex : uint32_t {
  kNuiSkeletonPositionHipCenter = 0,
  kNuiSkeletonPositionSpine = 1,
  kNuiSkeletonPositionShoulderCenter = 2,
  kNuiSkeletonPositionHead = 3,
  kNuiSkeletonPositionShoulderLeft = 4,
  kNuiSkeletonPositionElbowLeft = 5,
  kNuiSkeletonPositionWristLeft = 6,
  kNuiSkeletonPositionHandLeft = 7,
  kNuiSkeletonPositionShoulderRight = 8,
  kNuiSkeletonPositionElbowRight = 9,
  kNuiSkeletonPositionWristRight = 10,
  kNuiSkeletonPositionHandRight = 11,
  kNuiSkeletonPositionHipLeft = 12,
  kNuiSkeletonPositionKneeLeft = 13,
  kNuiSkeletonPositionAnkleLeft = 14,
  kNuiSkeletonPositionFootLeft = 15,
  kNuiSkeletonPositionHipRight = 16,
  kNuiSkeletonPositionKneeRight = 17,
  kNuiSkeletonPositionAnkleRight = 18,
  kNuiSkeletonPositionFootRight = 19,
};

struct X_NUI_VECTOR4 {
  xe::be<float> x;
  xe::be<float> y;
  xe::be<float> z;
  xe::be<float> w;
};
static_assert(sizeof(X_NUI_VECTOR4) == 16, "Size matters");

struct X_NUI_SKELETON_DATA {
  xe::be<uint32_t> tracking_state;
  xe::be<uint32_t> tracking_id;
  xe::be<uint32_t> enrollment_index;
  xe::be<uint32_t> user_index;
  X_NUI_VECTOR4 position;
  std::array<X_NUI_VECTOR4, kNuiSkeletonPositionCount> joints;
  std::array<xe::be<uint32_t>, kNuiSkeletonPositionCount> joint_states;
  xe::be<uint32_t> quality_flags;
};
static_assert(sizeof(X_NUI_SKELETON_DATA) == 436, "Unexpected skeleton size");

struct X_NUI_SKELETON_FRAME {
  xe::be<uint64_t> timestamp;
  xe::be<uint32_t> frame_number;
  xe::be<uint32_t> flags;
  X_NUI_VECTOR4 floor_clip_plane;
  X_NUI_VECTOR4 normal_to_gravity;
  std::array<X_NUI_SKELETON_DATA, kNuiSkeletonCount> skeletons;
};
static_assert(sizeof(X_NUI_SKELETON_FRAME) == 2664, "Unexpected frame size");

struct NuiJointSample {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float confidence = 0.0f;
  bool valid = false;
};

std::mutex g_nui_log_mutex;
std::deque<std::string> g_nui_log_lines;
std::atomic<uint64_t> g_nui_log_line_index = 0;
std::mutex g_nui_log_file_mutex;
std::filesystem::path g_nui_log_file_path;
FILE* g_nui_log_file = nullptr;
bool g_nui_log_file_open_failed = false;

bool ContainsNoCase(const std::string_view text, const std::string_view needle) {
  if (needle.empty()) {
    return true;
  }
  if (needle.size() > text.size()) {
    return false;
  }
  for (size_t i = 0; i + needle.size() <= text.size(); ++i) {
    bool matched = true;
    for (size_t j = 0; j < needle.size(); ++j) {
      const unsigned char a = unsigned(text[i + j]);
      const unsigned char b = unsigned(needle[j]);
      if (std::tolower(a) != std::tolower(b)) {
        matched = false;
        break;
      }
    }
    if (matched) {
      return true;
    }
  }
  return false;
}

bool IsXamModuleName(const std::string_view name) {
  return ContainsNoCase(name, "xam.xex");
}

bool LooksLikeKinectCall(const std::string_view name) {
  return ContainsNoCase(name, "nui") || ContainsNoCase(name, "natal") ||
         ContainsNoCase(name, "kinect");
}

void CloseNuiLogFileLocked() {
  if (!g_nui_log_file) {
    return;
  }
  std::fflush(g_nui_log_file);
  std::fclose(g_nui_log_file);
  g_nui_log_file = nullptr;
}

void AppendNuiLogFile(const std::string& line) {
  const std::filesystem::path desired_path = cvars::nui_sensor_log_path;
  std::lock_guard<std::mutex> lock(g_nui_log_file_mutex);

  if (desired_path.empty()) {
    CloseNuiLogFileLocked();
    g_nui_log_file_path.clear();
    g_nui_log_file_open_failed = false;
    return;
  }

  if (desired_path != g_nui_log_file_path || !g_nui_log_file) {
    CloseNuiLogFileLocked();
    g_nui_log_file_path = desired_path;
    g_nui_log_file_open_failed = false;

    xe::filesystem::CreateParentFolder(desired_path);
    g_nui_log_file = xe::filesystem::OpenFile(desired_path, "at");
    if (!g_nui_log_file) {
      if (!g_nui_log_file_open_failed) {
        XELOGE("NUI: failed to open dedicated log file {}",
               xe::path_to_utf8(desired_path));
        g_nui_log_file_open_failed = true;
      }
      return;
    }
    XELOGI("NUI: writing dedicated call log to {}",
           xe::path_to_utf8(desired_path));
  }

  if (!g_nui_log_file) {
    return;
  }

  std::fwrite(line.data(), sizeof(char), line.size(), g_nui_log_file);
  std::fputc('\n', g_nui_log_file);
  std::fflush(g_nui_log_file);
}

void AddNuiLogLine(const std::string& message) {
  const uint64_t line_index =
      g_nui_log_line_index.fetch_add(1, std::memory_order_relaxed) + 1;
  const std::string line = fmt::format("[{:06}] {}", line_index, message);
  {
    std::lock_guard<std::mutex> lock(g_nui_log_mutex);
    g_nui_log_lines.emplace_back(line);
    while (g_nui_log_lines.size() > kNuiLogLineLimit) {
      g_nui_log_lines.pop_front();
    }
  }
  AppendNuiLogFile(line);
  if (cvars::log_kinect_nui_calls) {
    XELOGI("NUI: {}", message);
  }
}

void TraceXamExternCall(const xe::cpu::Function* function, bool undefined) {
  if (!cvars::nui_trace_all_xam_import_calls || !function) {
    return;
  }
  const xe::cpu::Module* module = function->module();
  if (!module || !IsXamModuleName(module->name())) {
    return;
  }
  const std::string_view function_name = function->name();
  const bool unknown_xam = undefined && ContainsNoCase(function_name, "__xam_");
  const bool trace_all_xam = cvars::nui_trace_all_xam_calls;
  if (!trace_all_xam && !LooksLikeKinectCall(function_name) && !unknown_xam) {
    return;
  }
  AddNuiLogLine(fmt::format("XAM extern {}: {} @ {:08X}",
                            undefined ? "undefined" : "dispatch",
                            function_name, function->address()));
}

template <typename... Args>
void LogNui(const char* format, Args&&... args) {
  AddNuiLogLine(fmt::format(format, std::forward<Args>(args)...));
}

bool IsLikelyGuestAddress(uint32_t guest_address) {
  return guest_address != 0 && kernel_memory()->LookupHeap(guest_address);
}

bool WriteGuestU32(uint32_t guest_address, uint32_t value) {
  if (!IsLikelyGuestAddress(guest_address)) {
    return false;
  }
  uint8_t* host_ptr = kernel_memory()->TranslateVirtual<uint8_t*>(guest_address);
  if (!host_ptr) {
    return false;
  }
  xe::store_and_swap<uint32_t>(host_ptr, value);
  return true;
}

bool WriteGuestF32(uint32_t guest_address, float value) {
  if (!IsLikelyGuestAddress(guest_address)) {
    return false;
  }
  uint8_t* host_ptr = kernel_memory()->TranslateVirtual<uint8_t*>(guest_address);
  if (!host_ptr) {
    return false;
  }
  xe::store_and_swap<float>(host_ptr, value);
  return true;
}

bool WriteGuestBytes(uint32_t guest_address, const void* src, size_t size) {
  if (!src || !size || !IsLikelyGuestAddress(guest_address)) {
    return false;
  }
  uint8_t* host_ptr = kernel_memory()->TranslateVirtual<uint8_t*>(guest_address);
  if (!host_ptr) {
    return false;
  }
  std::memcpy(host_ptr, src, size);
  return true;
}

std::vector<uint32_t> CollectLikelyOutputPointers(
    const std::array<uint32_t, 8>& args) {
  std::vector<uint32_t> pointers;
  pointers.reserve(args.size());
  for (uint32_t arg : args) {
    if (!IsLikelyGuestAddress(arg)) {
      continue;
    }
    bool duplicate = false;
    for (uint32_t existing : pointers) {
      if (existing == arg) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate) {
      pointers.push_back(arg);
    }
  }
  return pointers;
}

std::vector<std::string> CopyNuiLogLines() {
  std::lock_guard<std::mutex> lock(g_nui_log_mutex);
  return std::vector<std::string>(g_nui_log_lines.begin(), g_nui_log_lines.end());
}

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

float ClampDepthMeters(float depth_meters) {
  const float min_depth =
      std::max(float(cvars::nui_skeleton_min_depth_m), 0.1f);
  const float max_depth =
      std::max(float(cvars::nui_skeleton_max_depth_m), min_depth + 0.1f);
  if (!std::isfinite(depth_meters) || depth_meters <= 0.0f) {
    depth_meters = float(cvars::nui_skeleton_torso_baseline_z_m);
  }
  return std::clamp(depth_meters, min_depth, max_depth);
}

float TanHalfFovRadians(double fov_degrees) {
  const float clamped = std::clamp(float(fov_degrees), 1.0f, 179.0f);
  const float radians = clamped * (kPi / 180.0f);
  return std::tan(radians * 0.5f);
}

std::string FourCCToString(uint32_t fourcc) {
  std::string out(4, '.');
  out[0] = std::isprint(int((fourcc >> 0) & 0xFF))
               ? char((fourcc >> 0) & 0xFF)
               : '.';
  out[1] = std::isprint(int((fourcc >> 8) & 0xFF))
               ? char((fourcc >> 8) & 0xFF)
               : '.';
  out[2] = std::isprint(int((fourcc >> 16) & 0xFF))
               ? char((fourcc >> 16) & 0xFF)
               : '.';
  out[3] = std::isprint(int((fourcc >> 24) & 0xFF))
               ? char((fourcc >> 24) & 0xFF)
               : '.';
  return out;
}

uint8_t ClampColorToByte(int value) {
  return uint8_t(std::clamp(value, 0, 255));
}

uint8_t RgbToY(uint8_t r, uint8_t g, uint8_t b) {
  const int y = ((66 * int(r) + 129 * int(g) + 25 * int(b) + 128) >> 8) + 16;
  return ClampColorToByte(y);
}

uint8_t RgbToU(uint8_t r, uint8_t g, uint8_t b) {
  const int u = ((-38 * int(r) - 74 * int(g) + 112 * int(b) + 128) >> 8) + 128;
  return ClampColorToByte(u);
}

uint8_t RgbToV(uint8_t r, uint8_t g, uint8_t b) {
  const int v = ((112 * int(r) - 94 * int(g) - 18 * int(b) + 128) >> 8) + 128;
  return ClampColorToByte(v);
}

bool ConvertBgraToYuy2(const std::vector<uint8_t>& bgra, uint32_t width,
                       uint32_t height, std::vector<uint8_t>* out_yuy2) {
  if (!out_yuy2 || !width || !height || (width & 1u)) {
    return false;
  }
  const size_t bgra_size = size_t(width) * size_t(height) * 4u;
  if (bgra.size() < bgra_size) {
    return false;
  }
  out_yuy2->resize(size_t(width) * size_t(height) * 2u);
  uint8_t* dst = out_yuy2->data();
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; x += 2) {
      const size_t i0 = (size_t(y) * width + x) * 4u;
      const size_t i1 = i0 + 4u;
      const uint8_t b0 = bgra[i0 + 0];
      const uint8_t g0 = bgra[i0 + 1];
      const uint8_t r0 = bgra[i0 + 2];
      const uint8_t b1 = bgra[i1 + 0];
      const uint8_t g1 = bgra[i1 + 1];
      const uint8_t r1 = bgra[i1 + 2];

      const uint8_t y0 = RgbToY(r0, g0, b0);
      const uint8_t y1 = RgbToY(r1, g1, b1);
      const uint8_t u0 = RgbToU(r0, g0, b0);
      const uint8_t u1 = RgbToU(r1, g1, b1);
      const uint8_t v0 = RgbToV(r0, g0, b0);
      const uint8_t v1 = RgbToV(r1, g1, b1);

      const uint8_t u = uint8_t((uint32_t(u0) + uint32_t(u1)) / 2u);
      const uint8_t v = uint8_t((uint32_t(v0) + uint32_t(v1)) / 2u);
      const size_t o = (size_t(y) * width + x) * 2u;
      dst[o + 0] = y0;
      dst[o + 1] = u;
      dst[o + 2] = y1;
      dst[o + 3] = v;
    }
  }
  return true;
}

void SetVector(X_NUI_VECTOR4* vector, float x, float y, float z, float w) {
  assert_not_null(vector);
  vector->x = x;
  vector->y = y;
  vector->z = z;
  vector->w = w;
}

NuiJointSample GetCocoJoint(const NuiDebugSnapshot& snapshot, uint32_t index,
                            float confidence_threshold) {
  NuiJointSample sample;
  sample.z = ClampDepthMeters(float(cvars::nui_skeleton_torso_baseline_z_m));
  if (index >= snapshot.joint_count) {
    return sample;
  }
  const NuiJoint& joint = snapshot.joints[index];
  sample.x = joint.x;
  sample.y = joint.y;
  sample.z = ClampDepthMeters(joint.z);
  sample.confidence = joint.confidence;
  sample.valid = joint.confidence >= confidence_threshold;
  return sample;
}

NuiJointSample AverageJoints(const NuiJointSample& a, const NuiJointSample& b) {
  NuiJointSample out;
  out.x = (a.x + b.x) * 0.5f;
  out.y = (a.y + b.y) * 0.5f;
  out.z = (a.z + b.z) * 0.5f;
  out.confidence = (a.confidence + b.confidence) * 0.5f;
  out.valid = a.valid && b.valid;
  return out;
}

void FillJoint(X_NUI_SKELETON_DATA* skeleton, NuiSkeletonPositionIndex index,
               const NuiJointSample& sample) {
  assert_not_null(skeleton);
  SetVector(&skeleton->joints[index], sample.x, sample.y,
            ClampDepthMeters(sample.z), 1.0f);
  skeleton->joint_states[index] = sample.valid ? kNuiSkeletonPositionTracked
                                               : kNuiSkeletonPositionInferred;
}

void BuildSkeletonFrame(const NuiDebugSnapshot& snapshot,
                        X_NUI_SKELETON_FRAME* out_frame) {
  assert_not_null(out_frame);
  *out_frame = X_NUI_SKELETON_FRAME{};

  out_frame->timestamp = snapshot.sensor_timestamp_us;
  out_frame->frame_number = snapshot.frame_index;
  out_frame->flags = snapshot.tracked ? 1u : 0u;
  SetVector(&out_frame->floor_clip_plane, 0.0f, 1.0f, 0.0f, 0.0f);
  SetVector(&out_frame->normal_to_gravity, 0.0f, 1.0f, 0.0f, 0.0f);

  for (auto& skeleton : out_frame->skeletons) {
    skeleton.tracking_state = kNuiSkeletonNotTracked;
    skeleton.enrollment_index = 0xFFFFFFFFu;
    skeleton.user_index = 0xFFFFFFFFu;
  }

  if (!snapshot.has_recent_frame || !snapshot.tracked || !snapshot.joint_count) {
    return;
  }

  constexpr uint32_t kCocoNose = 0;
  constexpr uint32_t kCocoLeftShoulder = 5;
  constexpr uint32_t kCocoRightShoulder = 6;
  constexpr uint32_t kCocoLeftElbow = 7;
  constexpr uint32_t kCocoRightElbow = 8;
  constexpr uint32_t kCocoLeftWrist = 9;
  constexpr uint32_t kCocoRightWrist = 10;
  constexpr uint32_t kCocoLeftHip = 11;
  constexpr uint32_t kCocoRightHip = 12;
  constexpr uint32_t kCocoLeftKnee = 13;
  constexpr uint32_t kCocoRightKnee = 14;
  constexpr uint32_t kCocoLeftAnkle = 15;
  constexpr uint32_t kCocoRightAnkle = 16;

  const float confidence_threshold =
      float(cvars::nui_sensor_min_joint_confidence);
  const NuiJointSample head =
      GetCocoJoint(snapshot, kCocoNose, confidence_threshold);
  const NuiJointSample shoulder_left =
      GetCocoJoint(snapshot, kCocoLeftShoulder, confidence_threshold);
  const NuiJointSample shoulder_right =
      GetCocoJoint(snapshot, kCocoRightShoulder, confidence_threshold);
  const NuiJointSample elbow_left =
      GetCocoJoint(snapshot, kCocoLeftElbow, confidence_threshold);
  const NuiJointSample elbow_right =
      GetCocoJoint(snapshot, kCocoRightElbow, confidence_threshold);
  const NuiJointSample wrist_left =
      GetCocoJoint(snapshot, kCocoLeftWrist, confidence_threshold);
  const NuiJointSample wrist_right =
      GetCocoJoint(snapshot, kCocoRightWrist, confidence_threshold);
  const NuiJointSample hip_left =
      GetCocoJoint(snapshot, kCocoLeftHip, confidence_threshold);
  const NuiJointSample hip_right =
      GetCocoJoint(snapshot, kCocoRightHip, confidence_threshold);
  const NuiJointSample knee_left =
      GetCocoJoint(snapshot, kCocoLeftKnee, confidence_threshold);
  const NuiJointSample knee_right =
      GetCocoJoint(snapshot, kCocoRightKnee, confidence_threshold);
  const NuiJointSample ankle_left =
      GetCocoJoint(snapshot, kCocoLeftAnkle, confidence_threshold);
  const NuiJointSample ankle_right =
      GetCocoJoint(snapshot, kCocoRightAnkle, confidence_threshold);

  const bool input_normalized = cvars::nui_skeleton_input_normalized;
  const float tan_half_fov_x = TanHalfFovRadians(cvars::nui_skeleton_camera_fov_x_deg);
  const float tan_half_fov_y = TanHalfFovRadians(cvars::nui_skeleton_camera_fov_y_deg);
  auto to_skeleton_space = [input_normalized, tan_half_fov_x, tan_half_fov_y](
                               NuiJointSample sample) -> NuiJointSample {
    sample.z = ClampDepthMeters(sample.z);
    if (input_normalized) {
      sample.x = sample.x * sample.z * tan_half_fov_x;
      sample.y = sample.y * sample.z * tan_half_fov_y;
    }
    return sample;
  };

  const NuiJointSample head_s = to_skeleton_space(head);
  const NuiJointSample shoulder_left_s = to_skeleton_space(shoulder_left);
  const NuiJointSample shoulder_right_s = to_skeleton_space(shoulder_right);
  const NuiJointSample elbow_left_s = to_skeleton_space(elbow_left);
  const NuiJointSample elbow_right_s = to_skeleton_space(elbow_right);
  const NuiJointSample wrist_left_s = to_skeleton_space(wrist_left);
  const NuiJointSample wrist_right_s = to_skeleton_space(wrist_right);
  const NuiJointSample hip_left_s = to_skeleton_space(hip_left);
  const NuiJointSample hip_right_s = to_skeleton_space(hip_right);
  const NuiJointSample knee_left_s = to_skeleton_space(knee_left);
  const NuiJointSample knee_right_s = to_skeleton_space(knee_right);
  const NuiJointSample ankle_left_s = to_skeleton_space(ankle_left);
  const NuiJointSample ankle_right_s = to_skeleton_space(ankle_right);

  const NuiJointSample shoulder_center =
      AverageJoints(shoulder_left_s, shoulder_right_s);
  const NuiJointSample hip_center = AverageJoints(hip_left_s, hip_right_s);
  const NuiJointSample spine = AverageJoints(hip_center, shoulder_center);

  NuiJointSample safe_hip_center = hip_center;
  if (!safe_hip_center.valid) {
    safe_hip_center.x = 0.0f;
    safe_hip_center.y = 0.0f;
    safe_hip_center.z =
        ClampDepthMeters(float(cvars::nui_skeleton_torso_baseline_z_m));
    safe_hip_center.valid = true;
  }
  NuiJointSample safe_head = head_s;
  if (!safe_head.valid) {
    safe_head.x = safe_hip_center.x;
    safe_head.y = safe_hip_center.y - 0.35f;
    safe_head.z = safe_hip_center.z - 0.08f;
    safe_head.valid = true;
  }
  NuiJointSample safe_hip_left = hip_left_s;
  if (!safe_hip_left.valid) {
    safe_hip_left.x = safe_hip_center.x - 0.10f;
    safe_hip_left.y = safe_hip_center.y;
    safe_hip_left.z = safe_hip_center.z + 0.05f;
    safe_hip_left.valid = true;
  }
  NuiJointSample safe_hip_right = hip_right_s;
  if (!safe_hip_right.valid) {
    safe_hip_right.x = safe_hip_center.x + 0.10f;
    safe_hip_right.y = safe_hip_center.y;
    safe_hip_right.z = safe_hip_center.z + 0.05f;
    safe_hip_right.valid = true;
  }
  NuiJointSample safe_hand_left = wrist_left_s;
  if (!safe_hand_left.valid) {
    safe_hand_left.x = safe_hip_center.x - 0.20f;
    safe_hand_left.y = safe_hip_center.y + 0.10f;
    safe_hand_left.z = safe_hip_center.z - 0.25f;
    safe_hand_left.valid = true;
  }
  NuiJointSample safe_hand_right = wrist_right_s;
  if (!safe_hand_right.valid) {
    safe_hand_right.x = safe_hip_center.x + 0.20f;
    safe_hand_right.y = safe_hip_center.y + 0.10f;
    safe_hand_right.z = safe_hip_center.z - 0.25f;
    safe_hand_right.valid = true;
  }

  X_NUI_SKELETON_DATA& skeleton = out_frame->skeletons[0];
  skeleton.tracking_state = kNuiSkeletonTracked;
  skeleton.tracking_id = 1;
  skeleton.enrollment_index = 0;
  skeleton.user_index = 0;

  SetVector(&skeleton.position, safe_hip_center.x, safe_hip_center.y,
            safe_hip_center.z, 1.0f);
  FillJoint(&skeleton, kNuiSkeletonPositionHipCenter, safe_hip_center);
  FillJoint(&skeleton, kNuiSkeletonPositionSpine, spine);
  FillJoint(&skeleton, kNuiSkeletonPositionShoulderCenter, shoulder_center);
  FillJoint(&skeleton, kNuiSkeletonPositionHead, safe_head);
  FillJoint(&skeleton, kNuiSkeletonPositionShoulderLeft, shoulder_left_s);
  FillJoint(&skeleton, kNuiSkeletonPositionElbowLeft, elbow_left_s);
  FillJoint(&skeleton, kNuiSkeletonPositionWristLeft, wrist_left_s);
  FillJoint(&skeleton, kNuiSkeletonPositionHandLeft, safe_hand_left);
  FillJoint(&skeleton, kNuiSkeletonPositionShoulderRight, shoulder_right_s);
  FillJoint(&skeleton, kNuiSkeletonPositionElbowRight, elbow_right_s);
  FillJoint(&skeleton, kNuiSkeletonPositionWristRight, wrist_right_s);
  FillJoint(&skeleton, kNuiSkeletonPositionHandRight, safe_hand_right);
  FillJoint(&skeleton, kNuiSkeletonPositionHipLeft, safe_hip_left);
  FillJoint(&skeleton, kNuiSkeletonPositionKneeLeft, knee_left_s);
  FillJoint(&skeleton, kNuiSkeletonPositionAnkleLeft, ankle_left_s);
  FillJoint(&skeleton, kNuiSkeletonPositionFootLeft, ankle_left_s);
  FillJoint(&skeleton, kNuiSkeletonPositionHipRight, safe_hip_right);
  FillJoint(&skeleton, kNuiSkeletonPositionKneeRight, knee_right_s);
  FillJoint(&skeleton, kNuiSkeletonPositionAnkleRight, ankle_right_s);
  FillJoint(&skeleton, kNuiSkeletonPositionFootRight, ankle_right_s);
  // Keep core joints tracked so retail title gating logic sees a stable body.
  skeleton.joint_states[kNuiSkeletonPositionHead] = kNuiSkeletonPositionTracked;
  skeleton.joint_states[kNuiSkeletonPositionHipCenter] =
      kNuiSkeletonPositionTracked;
  skeleton.joint_states[kNuiSkeletonPositionHipLeft] =
      kNuiSkeletonPositionTracked;
  skeleton.joint_states[kNuiSkeletonPositionHipRight] =
      kNuiSkeletonPositionTracked;
  skeleton.joint_states[kNuiSkeletonPositionHandLeft] =
      kNuiSkeletonPositionTracked;
  skeleton.joint_states[kNuiSkeletonPositionHandRight] =
      kNuiSkeletonPositionTracked;
  skeleton.tracking_state = kNuiSkeletonTracked;
}

class NuiUdpSensorService {
 public:
  static NuiUdpSensorService& Get() {
    static NuiUdpSensorService service;
    return service;
  }

  ~NuiUdpSensorService() { Shutdown(); }

  void EnsureRunning() {
    EnsureExternCallHook();
    EnsureDebugOverlayRegistered();
    bool logged = startup_logged_.exchange(true, std::memory_order_relaxed);
    if (!logged) {
      LogNui(
          "NUI bridge revision={} status_default={} acquire_frames={} hold_ms={} color_default={}({:08X})",
          kNuiBridgeRevision, uint32_t(std::max(cvars::nui_device_status_value, 0)),
          uint32_t(std::max(cvars::nui_sensor_tracking_acquire_frames, 0)),
          uint32_t(std::max(cvars::nui_sensor_tracking_hold_ms, 0)),
          FourCCToString(uint32_t(std::max(cvars::nui_sensor_color_format_fourcc, 0))),
          uint32_t(std::max(cvars::nui_sensor_color_format_fourcc, 0)));
    }
    if (!cvars::nui_sensor_udp_enabled) {
      return;
    }
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
    snapshot.frame_width = last_frame_width_.load(std::memory_order_relaxed);
    snapshot.frame_height = last_frame_height_.load(std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(frame_mutex_);
    snapshot.joint_count = latest_joint_count_;
    if (snapshot.joint_count) {
      std::copy_n(latest_joints_.cbegin(), snapshot.joint_count,
                  snapshot.joints.begin());
    }
    return snapshot;
  }

  bool HasRecentColorFrame() const {
    if (!cvars::nui_sensor_udp_enabled) {
      return true;
    }
    const uint64_t last_us = last_color_host_us_.load(std::memory_order_relaxed);
    if (!last_us) {
      return false;
    }
    const uint64_t timeout_us =
        static_cast<uint64_t>(std::max(cvars::nui_sensor_frame_timeout_ms, 1)) *
        1000ull;
    return QuerySteadyMicros() - last_us <= timeout_us;
  }

  bool BuildLatestColorBgra(uint16_t target_width, uint16_t target_height,
                            std::vector<uint8_t>* out_bgra,
                            uint32_t* out_frame_index) const {
    if (!out_bgra || !target_width || !target_height) {
      return false;
    }

    NuiColorFrame color;
    {
      std::lock_guard<std::mutex> lock(color_mutex_);
      if (latest_color_frame_.rgb24.empty() || !latest_color_frame_.width ||
          !latest_color_frame_.height) {
        return false;
      }
      color = latest_color_frame_;
    }
    if (!HasRecentColorFrame()) {
      return false;
    }

    const uint32_t src_w = color.width;
    const uint32_t src_h = color.height;
    const uint32_t dst_w = target_width;
    const uint32_t dst_h = target_height;
    const uint64_t dst_size =
        uint64_t(dst_w) * uint64_t(dst_h) * uint64_t(4);
    if (dst_size > size_t(-1)) {
      return false;
    }
    out_bgra->resize(size_t(dst_size));

    const uint8_t* src = color.rgb24.data();
    uint8_t* dst = out_bgra->data();
    for (uint32_t y = 0; y < dst_h; ++y) {
      const uint32_t sy = std::min((uint64_t(y) * src_h) / dst_h, uint64_t(src_h - 1));
      for (uint32_t x = 0; x < dst_w; ++x) {
        const uint32_t sx =
            std::min((uint64_t(x) * src_w) / dst_w, uint64_t(src_w - 1));
        const size_t src_index = (size_t(sy) * src_w + sx) * 3;
        const size_t dst_index = (size_t(y) * dst_w + x) * 4;
        const uint8_t r = src[src_index + 0];
        const uint8_t g = src[src_index + 1];
        const uint8_t b = src[src_index + 2];
        dst[dst_index + 0] = b;
        dst[dst_index + 1] = g;
        dst[dst_index + 2] = r;
        dst[dst_index + 3] = 0xFF;
      }
    }
    if (out_frame_index) {
      *out_frame_index = color.frame_index;
    }
    return true;
  }

 private:
  void EnsureExternCallHook() {
    const auto current_hook = xe::cpu::GetExternCallTraceHook();
    if (cvars::nui_trace_all_xam_import_calls) {
      if (current_hook != &TraceXamExternCall) {
        xe::cpu::SetExternCallTraceHook(&TraceXamExternCall);
        AddNuiLogLine("Enabled XAM extern Kinect call tracing hook");
      }
    } else if (current_hook == &TraceXamExternCall) {
      xe::cpu::SetExternCallTraceHook(nullptr);
      AddNuiLogLine("Disabled XAM extern Kinect call tracing hook");
    }
  }

  void EnsureDebugOverlayRegistered() {
    if ((!cvars::show_kinect_debug && !cvars::show_kinect_nui_log) ||
        cvars::headless) {
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
            if (!cvars::show_kinect_debug && !cvars::show_kinect_nui_log) {
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
    if (!cvars::show_kinect_debug && !cvars::show_kinect_nui_log) {
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

    if (cvars::show_kinect_debug) {
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
      if (snapshot.frame_width && snapshot.frame_height) {
        ImGui::Text("source=%ux%u", uint32_t(snapshot.frame_width),
                    uint32_t(snapshot.frame_height));
      } else {
        ImGui::Text("source=unknown");
      }
      ImGui::End();
    }

    if (cvars::show_kinect_nui_log) {
      ImGui::SetNextWindowPos(ImVec2(12.0f, 108.0f), ImGuiCond_FirstUseEver);
      ImGui::SetNextWindowSize(ImVec2(760.0f, 340.0f), ImGuiCond_FirstUseEver);
      ImGui::SetNextWindowBgAlpha(0.65f);
      if (ImGui::Begin("Kinect NUI Call Log", nullptr)) {
        if (ImGui::Button("Clear")) {
          std::lock_guard<std::mutex> lock(g_nui_log_mutex);
          g_nui_log_lines.clear();
        }
        ImGui::SameLine();
        ImGui::Text("entries=%llu",
                    static_cast<unsigned long long>(
                        g_nui_log_line_index.load(std::memory_order_relaxed)));
        ImGui::Separator();
        ImGui::BeginChild("kinect_nui_log_scroll", ImVec2(0.0f, 0.0f), false,
                          ImGuiWindowFlags_HorizontalScrollbar);
        const auto lines = CopyNuiLogLines();
        for (const std::string& line : lines) {
          ImGui::TextUnformatted(line.c_str());
        }
        ImGui::SetScrollHereY(1.0f);
        ImGui::EndChild();
      }
      ImGui::End();
    }

    if (!cvars::show_kinect_debug) {
      return;
    }

    if (!snapshot.has_recent_frame || !snapshot.joint_count ||
        !(io.DisplaySize.x > 0.0f) || !(io.DisplaySize.y > 0.0f)) {
      return;
    }

    const float display_width = io.DisplaySize.x;
    const float display_height = io.DisplaySize.y;
    float source_aspect = 0.0f;
    if (snapshot.frame_width && snapshot.frame_height) {
      source_aspect =
          float(snapshot.frame_width) / float(snapshot.frame_height);
    }
    if (!(source_aspect > 0.0f)) {
      source_aspect = std::max(float(cvars::nui_sensor_source_aspect_ratio),
                               0.1f);
    }
    const float display_aspect = display_width / display_height;
    float content_x = 0.0f;
    float content_y = 0.0f;
    float content_width = display_width;
    float content_height = display_height;
    if (display_aspect > source_aspect) {
      content_width = display_height * source_aspect;
      content_x = (display_width - content_width) * 0.5f;
    } else {
      content_height = display_width / source_aspect;
      content_y = (display_height - content_height) * 0.5f;
    }
    const float confidence_threshold =
        float(cvars::nui_sensor_min_joint_confidence);
    ImDrawList* draw_list = ImGui::GetForegroundDrawList();
    if (!draw_list) {
      return;
    }

    auto joint_to_screen = [content_x, content_y, content_width,
                            content_height](const NuiJoint& joint) -> ImVec2 {
      // Sender currently emits normalized camera coordinates in [-1, 1].
      float x_norm = joint.x * 0.5f + 0.5f;
      float y_norm = 0.5f - joint.y * 0.5f;
      float x = content_x + x_norm * content_width;
      float y = content_y + y_norm * content_height;
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
    if (xe::cpu::GetExternCallTraceHook() == &TraceXamExternCall) {
      xe::cpu::SetExternCallTraceHook(nullptr);
      AddNuiLogLine("Disabled XAM extern Kinect call tracing hook");
    }
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
    AddNuiLogLine("NUI UDP sensor worker started");

    NuiSocket sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock == kInvalidNuiSocket) {
      XELOGE("NUI sensor: failed to create UDP socket");
      AddNuiLogLine("Failed to create NUI UDP socket");
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
      LogNui("Failed to bind NUI UDP port {}", cvars::nui_sensor_udp_port);
      CloseNuiSocket(&sock);
      return;
    }

    XELOGI("NUI sensor: listening on 127.0.0.1:{}", cvars::nui_sensor_udp_port);
    LogNui("Listening for RTMPose UDP frames at 127.0.0.1:{}",
           cvars::nui_sensor_udp_port);

    std::array<uint8_t, kNuiUdpMaxPacketSize> packet = {};
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
    AddNuiLogLine("NUI UDP sensor worker stopped");
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
    uint16_t frame_width = 0;
    uint16_t frame_height = 0;
    size_t extension_offset = expected_size;
    if (size >= expected_size + kNuiUdpFrameSizeTailSize) {
      frame_width = ReadLE16(data + expected_size + 0);
      frame_height = ReadLE16(data + expected_size + 2);
      extension_offset += kNuiUdpFrameSizeTailSize;
    }

    bool has_color_extension = false;
    uint16_t color_width = 0;
    uint16_t color_height = 0;
    std::vector<uint8_t> color_rgb24;
    if (size >= extension_offset + kNuiUdpColorHeaderSize) {
      const uint32_t color_magic = ReadLE32(data + extension_offset + 0);
      if (color_magic == kNuiUdpColorMagic) {
        color_width = ReadLE16(data + extension_offset + 4);
        color_height = ReadLE16(data + extension_offset + 6);
        const uint8_t color_format = data[extension_offset + 8];
        const uint32_t color_payload_size = ReadLE32(data + extension_offset + 12);
        const size_t payload_offset = extension_offset + kNuiUdpColorHeaderSize;
        const bool payload_in_range =
            size >= payload_offset && color_payload_size <= (size - payload_offset);
        const bool payload_valid =
            color_payload_size <= kNuiUdpMaxColorPayloadSize && color_width &&
            color_height && color_format == kNuiUdpColorFormatRgb24;
        if (payload_in_range && payload_valid) {
          color_rgb24.resize(color_payload_size);
          std::memcpy(color_rgb24.data(), data + payload_offset, color_payload_size);
          has_color_extension = true;
        } else {
          const uint32_t dropped =
              dropped_color_extension_count_.fetch_add(1, std::memory_order_relaxed) + 1;
          if (dropped <= 4 || (dropped % 120u) == 0u) {
            LogNui(
                "RX color dropped frame={} reason={}{} fmt={} wh={}x{} payload={} max={}",
                frame_index, payload_in_range ? "" : "range",
                payload_valid ? "" : (payload_in_range ? "invalid" : "+invalid"),
                uint32_t(color_format), uint32_t(color_width),
                uint32_t(color_height), color_payload_size,
                uint32_t(kNuiUdpMaxColorPayloadSize));
          }
        }
      }
    }

    const size_t capped_joint_count =
        std::min(size_t(joint_count), kNuiMaxJoints);
    std::array<NuiJoint, kNuiMaxJoints> joints = {};

    bool tracked_raw = tracked_flag != 0;
    uint32_t confident_joint_count = 0;
    const float depth_smoothing =
        std::clamp(float(cvars::nui_skeleton_depth_smoothing), 0.0f, 1.0f);
    for (size_t i = 0; i < capped_joint_count; ++i) {
      const uint8_t* joint_ptr =
          data + kNuiUdpHeaderSize + i * kNuiUdpJointSize;
      NuiJoint joint;
      joint.x = ReadLEFloat(joint_ptr + 0);
      joint.y = ReadLEFloat(joint_ptr + 4);
      joint.z = ReadLEFloat(joint_ptr + 8);
      joint.confidence = ReadLEFloat(joint_ptr + 12);
      joint.z = ClampDepthMeters(joint.z);
      if (depth_initialized_[i]) {
        joint.z =
            smoothed_depths_[i] * (1.0f - depth_smoothing) +
            joint.z * depth_smoothing;
      }
      smoothed_depths_[i] = joint.z;
      depth_initialized_[i] = true;
      joints[i] = joint;
      if (joint.confidence >= float(cvars::nui_sensor_min_joint_confidence)) {
        ++confident_joint_count;
        tracked_raw = true;
      }
    }
    const bool previous_tracked = tracked_.load(std::memory_order_relaxed);

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

    if (tracked_raw) {
      ++consecutive_tracked_frames_;
      last_tracked_host_us_.store(host_receive_us, std::memory_order_relaxed);
    } else {
      consecutive_tracked_frames_ = 0;
    }
    const uint32_t acquire_frames =
        uint32_t(std::max(cvars::nui_sensor_tracking_acquire_frames, 1));
    const uint64_t hold_us =
        uint64_t(std::max(cvars::nui_sensor_tracking_hold_ms, 0)) * 1000ull;
    bool tracked = previous_tracked;
    if (tracked_raw) {
      tracked = previous_tracked || (consecutive_tracked_frames_ >= acquire_frames);
    } else {
      if (previous_tracked && hold_us > 0) {
        const uint64_t last_tracked_us =
            last_tracked_host_us_.load(std::memory_order_relaxed);
        tracked = last_tracked_us && host_receive_us >= last_tracked_us &&
                  (host_receive_us - last_tracked_us) <= hold_us;
      } else {
        tracked = false;
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
    last_frame_width_.store(frame_width, std::memory_order_relaxed);
    last_frame_height_.store(frame_height, std::memory_order_relaxed);
    if (has_color_extension) {
      std::lock_guard<std::mutex> lock(color_mutex_);
      latest_color_frame_.width = color_width;
      latest_color_frame_.height = color_height;
      latest_color_frame_.frame_index = frame_index;
      latest_color_frame_.sensor_timestamp_us = sensor_timestamp_us;
      latest_color_frame_.host_receive_us = host_receive_us;
      latest_color_frame_.rgb24 = std::move(color_rgb24);
      last_color_host_us_.store(host_receive_us, std::memory_order_relaxed);
      bool expected = false;
      if (received_any_color_frame_.compare_exchange_strong(
              expected, true, std::memory_order_relaxed)) {
        LogNui("RX first color frame={} rgb={}x{} bytes={}", frame_index,
               uint32_t(color_width), uint32_t(color_height),
               uint32_t(latest_color_frame_.rgb24.size()));
      }
    }

    const int log_interval = std::max(cvars::nui_sensor_log_frame_interval, 0);
    if (log_interval &&
        ((frame_index % uint32_t(log_interval)) == 0 || tracked != previous_tracked)) {
      const NuiJoint nose = capped_joint_count > 0 ? joints[0] : NuiJoint{};
      const NuiJoint hip_left = capped_joint_count > 11 ? joints[11] : NuiJoint{};
      const NuiJoint hip_right = capped_joint_count > 12 ? joints[12] : NuiJoint{};
      LogNui(
          "RX frame={} tracked={} raw={} conf={} acquire={} hold_ms={} joints={} src={}x{} nose=({:.3f},{:.3f},{:.3f}|{:.2f}) hips=({:.3f},{:.3f})/({:.3f},{:.3f})",
          frame_index, tracked ? "yes" : "no", tracked_raw ? "yes" : "no",
          confident_joint_count, acquire_frames,
          uint32_t(std::max(cvars::nui_sensor_tracking_hold_ms, 0)),
          uint32_t(capped_joint_count),
          uint32_t(frame_width), uint32_t(frame_height), nose.x, nose.y, nose.z,
          nose.confidence, hip_left.x, hip_left.y, hip_right.x, hip_right.y);
      if (has_color_extension) {
        LogNui("RX color frame={} rgb={}x{} bytes={}", frame_index,
               uint32_t(color_width), uint32_t(color_height),
               uint32_t(latest_color_frame_.rgb24.size()));
      }
    }
  }

  std::thread worker_;
  std::atomic<bool> started_ = false;
  std::atomic<bool> stop_requested_ = false;
  std::atomic<bool> startup_logged_ = false;
  std::atomic<bool> debug_overlay_registered_ = false;
  std::atomic<bool> tracked_ = false;
  std::atomic<uint32_t> last_frame_index_ = 0;
  std::atomic<uint64_t> last_sensor_timestamp_us_ = 0;
  std::atomic<uint64_t> last_frame_host_us_ = 0;
  std::atomic<uint64_t> last_tracked_host_us_ = 0;
  std::atomic<uint16_t> last_frame_width_ = 0;
  std::atomic<uint16_t> last_frame_height_ = 0;
  std::atomic<float> receive_fps_ = 0.0f;
  uint32_t consecutive_tracked_frames_ = 0;
  mutable std::mutex frame_mutex_;
  uint16_t latest_joint_count_ = 0;
  std::array<NuiJoint, kNuiMaxJoints> latest_joints_ = {};
  std::array<float, kNuiMaxJoints> smoothed_depths_ = {};
  std::array<bool, kNuiMaxJoints> depth_initialized_ = {};
  mutable std::mutex color_mutex_;
  NuiColorFrame latest_color_frame_ = {};
  std::atomic<uint64_t> last_color_host_us_ = 0;
  std::atomic<uint32_t> dropped_color_extension_count_ = 0;
  std::atomic<bool> received_any_color_frame_ = false;
};

uint32_t engaged_tracking_id = 0;
uint32_t engaged_enrollment_index = 0;
uint64_t nui_session_id = 0;
std::atomic<bool> nui_force_device_off = false;
uint32_t nui_system_gesture_control = 1;

std::mutex g_nui_color_texture_mutex;
uint32_t g_nui_color_texture_guest_ptr = 0;
uint32_t g_nui_color_texture_capacity = 0;
uint32_t g_nui_color_texture_width = 0;
uint32_t g_nui_color_texture_height = 0;
uint32_t g_nui_color_texture_frame_index = 0;
uint32_t g_nui_color_texture_format_fourcc = kNuiColorFormatYUY2;
std::atomic<uint32_t> g_nui_color_miss_count = 0;
std::atomic<uint32_t> g_nui_color_update_count = 0;

bool IsNuiDeviceConnected() {
  if (!cvars::allow_nui_initialization) {
    return false;
  }
  if (nui_force_device_off.load(std::memory_order_relaxed)) {
    return false;
  }
  if (!cvars::nui_sensor_udp_enabled) {
    return true;
  }
  // Treat the virtual Kinect as connected whenever the bridge is enabled.
  auto& sensor = NuiUdpSensorService::Get();
  sensor.EnsureRunning();
  return true;
}

bool IsNuiSkeletonTracked() {
  if (!cvars::allow_nui_initialization) {
    return false;
  }
  if (nui_force_device_off.load(std::memory_order_relaxed)) {
    return false;
  }
  if (!cvars::nui_sensor_udp_enabled) {
    return true;
  }
  auto& sensor = NuiUdpSensorService::Get();
  sensor.EnsureRunning();
  return sensor.IsTracked();
}

bool UpdateNuiColorTextureFromSensor(uint32_t* out_guest_ptr, uint32_t* out_width,
                                     uint32_t* out_height,
                                     uint32_t* out_frame_index,
                                     uint32_t* out_format_fourcc) {
  NuiUdpSensorService& sensor = NuiUdpSensorService::Get();
  sensor.EnsureRunning();
  uint16_t target_width =
      uint16_t(std::clamp(cvars::nui_sensor_color_output_width, 16, 1920));
  const uint16_t target_height =
      uint16_t(std::clamp(cvars::nui_sensor_color_output_height, 16, 1080));

  uint32_t output_format =
      uint32_t(std::max(cvars::nui_sensor_color_format_fourcc, 0));
  if (!output_format) {
    output_format = kNuiColorFormatYUY2;
  }
  if (output_format == kNuiColorFormatYUY2 && (target_width & 1u)) {
    target_width = uint16_t(std::max(16, int(target_width & ~1u)));
  }

  std::vector<uint8_t> bgra;
  uint32_t frame_index = 0;
  const bool has_fresh_color =
      sensor.BuildLatestColorBgra(target_width, target_height, &bgra, &frame_index);
  if (!has_fresh_color) {
    const uint32_t miss =
        g_nui_color_miss_count.fetch_add(1, std::memory_order_relaxed) + 1;
    {
      std::lock_guard<std::mutex> lock(g_nui_color_texture_mutex);
      if (g_nui_color_texture_guest_ptr) {
        if (out_guest_ptr) {
          *out_guest_ptr = g_nui_color_texture_guest_ptr;
        }
        if (out_width) {
          *out_width = g_nui_color_texture_width;
        }
        if (out_height) {
          *out_height = g_nui_color_texture_height;
        }
        if (out_frame_index) {
          *out_frame_index = g_nui_color_texture_frame_index;
        }
        if (out_format_fourcc) {
          *out_format_fourcc = g_nui_color_texture_format_fourcc;
        }
        if (miss <= 4 || (miss % 120u) == 0u) {
          LogNui(
              "NUI color: using cached texture frame={} fmt={}({:08X}) wh={}x{}",
              g_nui_color_texture_frame_index,
              FourCCToString(g_nui_color_texture_format_fourcc),
              g_nui_color_texture_format_fourcc, g_nui_color_texture_width,
              g_nui_color_texture_height);
        }
        return true;
      }
    }
    if (miss <= 4 || (miss % 120u) == 0u) {
      LogNui(
          "NUI color: no RGB frame available yet (fmt={}({:08X}) target={}x{})",
          FourCCToString(output_format), output_format, uint32_t(target_width),
          uint32_t(target_height));
    }
    return false;
  }

  std::vector<uint8_t> texture_bytes;
  if (output_format == kNuiColorFormatYUY2) {
    if (!ConvertBgraToYuy2(bgra, target_width, target_height, &texture_bytes)) {
      LogNui("NUI color: failed YUY2 conversion for {}x{}", uint32_t(target_width),
             uint32_t(target_height));
      return false;
    }
  } else if (output_format == kNuiColorFormatRGB3) {
    texture_bytes.resize(size_t(target_width) * size_t(target_height) * 3u);
    for (size_t i = 0, j = 0; i + 3 < bgra.size(); i += 4, j += 3) {
      texture_bytes[j + 0] = bgra[i + 2];
      texture_bytes[j + 1] = bgra[i + 1];
      texture_bytes[j + 2] = bgra[i + 0];
    }
  } else {
    if (output_format != kNuiColorFormatBGRX) {
      LogNui(
          "NUI color: unsupported format {}({:08X}), using BGRX texture payload",
          FourCCToString(output_format), output_format);
    }
    output_format = kNuiColorFormatBGRX;
    texture_bytes = std::move(bgra);
  }

  const size_t required_size = texture_bytes.size();
  if (!required_size) {
    return false;
  }

  std::lock_guard<std::mutex> lock(g_nui_color_texture_mutex);
  if (!g_nui_color_texture_guest_ptr ||
      g_nui_color_texture_capacity < required_size) {
    const uint32_t guest_ptr =
        kernel_memory()->SystemHeapAlloc(uint32_t(required_size), 0x100);
    if (!guest_ptr) {
      return false;
    }
    g_nui_color_texture_guest_ptr = guest_ptr;
    g_nui_color_texture_capacity = uint32_t(required_size);
  }
  if (!WriteGuestBytes(g_nui_color_texture_guest_ptr, texture_bytes.data(),
                       texture_bytes.size())) {
    return false;
  }
  g_nui_color_texture_width = target_width;
  g_nui_color_texture_height = target_height;
  g_nui_color_texture_frame_index = frame_index;
  g_nui_color_texture_format_fourcc = output_format;

  const uint32_t update_count =
      g_nui_color_update_count.fetch_add(1, std::memory_order_relaxed) + 1;
  if (update_count <= 4 || (update_count % 120u) == 0u) {
    LogNui(
        "NUI color: updated frame={} fmt={}({:08X}) wh={}x{} bytes={}",
        frame_index, FourCCToString(output_format), output_format,
        uint32_t(target_width), uint32_t(target_height),
        uint32_t(texture_bytes.size()));
  }

  if (out_guest_ptr) {
    *out_guest_ptr = g_nui_color_texture_guest_ptr;
  }
  if (out_width) {
    *out_width = g_nui_color_texture_width;
  }
  if (out_height) {
    *out_height = g_nui_color_texture_height;
  }
  if (out_frame_index) {
    *out_frame_index = g_nui_color_texture_frame_index;
  }
  if (out_format_fourcc) {
    *out_format_fourcc = g_nui_color_texture_format_fourcc;
  }
  return true;
}

uint32_t XeXamNuiHudCheck(dword_t tracking_id) {
  if (!IsNuiDeviceConnected()) {
    LogNui("XeXamNuiHudCheck({:08X}) -> access denied",
           uint32_t(tracking_id));
    return X_ERROR_ACCESS_DENIED;
  }
  engaged_tracking_id = tracking_id;
  LogNui("XeXamNuiHudCheck({:08X}) -> success", uint32_t(tracking_id));
  return X_ERROR_SUCCESS;
}

}  // namespace

dword_result_t XamNuiGetDeviceStatus_entry(
    pointer_t<X_NUI_DEVICE_STATUS> status_ptr) {
  if (!status_ptr) {
    AddNuiLogLine("XamNuiGetDeviceStatus(status_ptr=null)");
    return X_E_INVALIDARG;
  }
  const bool connected = IsNuiDeviceConnected();
  const bool tracked = IsNuiSkeletonTracked();
  const uint32_t status_value =
      connected ? uint32_t(std::max(cvars::nui_device_status_value, 2)) : 0u;
  status_ptr.Zero();
  status_ptr->unk0 = status_value;
  status_ptr->unk1 = connected ? 1u : 0u;
  status_ptr->unk2 = tracked ? 1u : 0u;
  status_ptr->status = status_value;
  status_ptr->unk4 = connected ? 1u : 0u;
  status_ptr->unk5 = tracked ? 1u : 0u;
  LogNui(
      "XamNuiGetDeviceStatus(status_ptr={:08X}) -> connected={} tracked={} fields=[{:08X},{:08X},{:08X},{:08X},{:08X},{:08X}]",
      status_ptr.guest_address(), connected ? "yes" : "no",
      tracked ? "yes" : "no",
      uint32_t(status_ptr->unk0), uint32_t(status_ptr->unk1),
      uint32_t(status_ptr->unk2), uint32_t(status_ptr->status),
      uint32_t(status_ptr->unk4), uint32_t(status_ptr->unk5));
  return connected ? X_ERROR_SUCCESS : 0xC0050006;
}
DECLARE_XAM_EXPORT1(XamNuiGetDeviceStatus, kNone, kStub);

dword_result_t XamUserNuiGetUserIndex_entry(unknown_t unk, lpdword_t index) {
  if (!index) {
    AddNuiLogLine("XamUserNuiGetUserIndex(index_ptr=null)");
    return X_E_INVALIDARG;
  }
  if (!IsNuiDeviceConnected() || !IsNuiSkeletonTracked()) {
    LogNui("XamUserNuiGetUserIndex(unk={:08X}, index_ptr={:08X}) -> no user",
           uint32_t(unk), index.guest_address());
    return X_E_NO_SUCH_USER;
  }
  *index = 0;
  LogNui("XamUserNuiGetUserIndex(unk={:08X}, index_ptr={:08X}) -> 0",
         uint32_t(unk), index.guest_address());
  return X_E_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamUserNuiGetUserIndex, kNone, kImplemented);

dword_result_t XamUserNuiGetUserIndexForSignin_entry(lpdword_t index) {
  if (!index) {
    AddNuiLogLine("XamUserNuiGetUserIndexForSignin(index_ptr=null)");
    return X_E_INVALIDARG;
  }
  if (IsNuiDeviceConnected() && IsNuiSkeletonTracked()) {
    *index = 0;
    LogNui("XamUserNuiGetUserIndexForSignin(index_ptr={:08X}) -> 0",
           index.guest_address());
    return X_E_SUCCESS;
  }
  LogNui("XamUserNuiGetUserIndexForSignin(index_ptr={:08X}) -> no user",
         index.guest_address());
  return X_E_NO_SUCH_USER;
}
DECLARE_XAM_EXPORT1(XamUserNuiGetUserIndexForSignin, kNone, kImplemented);

dword_result_t XamUserNuiGetUserIndexForBind_entry(lpdword_t index) {
  if (!index) {
    AddNuiLogLine("XamUserNuiGetUserIndexForBind(index_ptr=null)");
    return X_E_INVALIDARG;
  }
  if (!IsNuiDeviceConnected() || !IsNuiSkeletonTracked()) {
    LogNui("XamUserNuiGetUserIndexForBind(index_ptr={:08X}) -> no user",
           index.guest_address());
    return X_E_NO_SUCH_USER;
  }
  *index = 0;
  LogNui("XamUserNuiGetUserIndexForBind(index_ptr={:08X}) -> 0",
         index.guest_address());
  return X_E_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamUserNuiGetUserIndexForBind, kNone, kImplemented);

dword_result_t XamNuiGetDepthCalibration_entry(lpdword_t unk1) {
  const bool connected = IsNuiDeviceConnected();
  if (unk1) {
    *unk1 = connected ? 1u : 0u;
  }
  LogNui("XamNuiGetDepthCalibration(ptr={:08X}) -> {}",
         unk1.guest_address(), connected ? "success" : "not found");
  return connected ? X_ERROR_SUCCESS : X_STATUS_NO_SUCH_FILE;
}
DECLARE_XAM_EXPORT1(XamNuiGetDepthCalibration, kNone, kStub);

dword_result_t XamNuiStoreDepthCalibration_entry(
    unknown_t arg0, unknown_t arg1, unknown_t arg2, unknown_t arg3) {
  const bool connected = IsNuiDeviceConnected();
  LogNui(
      "XamNuiStoreDepthCalibration({:08X}, {:08X}, {:08X}, {:08X}) -> {}",
      uint32_t(arg0), uint32_t(arg1), uint32_t(arg2), uint32_t(arg3),
      connected ? "success" : "not-connected");
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiStoreDepthCalibration, kNone, kImplemented);

dword_result_t XamNuiGetLoadedDepthCalibration_entry(
    unknown_t arg0, unknown_t arg1, unknown_t arg2, unknown_t arg3) {
  const std::array<uint32_t, 4> args = {
      uint32_t(arg0), uint32_t(arg1), uint32_t(arg2), uint32_t(arg3)};
  const bool connected = IsNuiDeviceConnected();
  std::array<uint32_t, 8> args8 = {args[0], args[1], args[2], args[3], 0, 0, 0, 0};
  const auto output_ptrs = CollectLikelyOutputPointers(args8);
  if (!output_ptrs.empty()) {
    WriteGuestU32(output_ptrs[0], connected ? 1u : 0u);
  }
  LogNui(
      "XamNuiGetLoadedDepthCalibration({:08X}, {:08X}, {:08X}, {:08X}) -> {}",
      args[0], args[1], args[2], args[3],
      connected ? "success" : "not-connected");
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiGetLoadedDepthCalibration, kNone, kImplemented);

dword_result_t XamNuiSetForceDeviceOff_entry(dword_t force_off) {
  const bool off = force_off != 0;
  nui_force_device_off.store(off, std::memory_order_relaxed);
  LogNui("XamNuiSetForceDeviceOff({})", off ? "1" : "0");
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamNuiSetForceDeviceOff, kNone, kImplemented);

dword_result_t XamNuiEnableChatMic_entry(int_t enable) {
  cvars::nui_chat_mic_enabled = enable != 0;
  LogNui("XamNuiEnableChatMic(enable={}) -> {}", int32_t(enable),
         cvars::nui_chat_mic_enabled ? "enabled" : "disabled");
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamNuiEnableChatMic, kNone, kImplemented);

qword_result_t XamNuiSkeletonGetBestSkeletonIndex_entry(int_t unk) {
  const bool tracked = IsNuiSkeletonTracked();
  LogNui("XamNuiSkeletonGetBestSkeletonIndex(unk={:08X}) -> {}",
         uint32_t(unk), tracked ? "0" : "FFFFFFFFFFFFFFFF");
  return tracked ? 0ull : 0xFFFFFFFFFFFFFFFFull;
}
DECLARE_XAM_EXPORT1(XamNuiSkeletonGetBestSkeletonIndex, kNone, kImplemented);

dword_result_t XamNuiSkeletonScoreUpdate_entry(
    unknown_t arg0, unknown_t arg1, unknown_t arg2, unknown_t arg3,
    unknown_t arg4, unknown_t arg5, unknown_t arg6, unknown_t arg7) {
  const std::array<uint32_t, 8> args = {
      uint32_t(arg0), uint32_t(arg1), uint32_t(arg2), uint32_t(arg3),
      uint32_t(arg4), uint32_t(arg5), uint32_t(arg6), uint32_t(arg7),
  };
  LogNui(
      "XamNuiSkeletonScoreUpdate(args=[{:08X},{:08X},{:08X},{:08X},{:08X},{:08X},{:08X},{:08X}])",
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);

  int output_arg_index =
      std::clamp(cvars::nui_skeleton_score_output_arg_index, -1, 7);
  if (output_arg_index < 0) {
    AddNuiLogLine(
        "XamNuiSkeletonScoreUpdate output disabled by cvar "
        "nui_skeleton_score_output_arg_index");
    return X_ERROR_SUCCESS;
  }

  uint32_t output_ptr = args[size_t(output_arg_index)];
  if (!IsLikelyGuestAddress(output_ptr)) {
    for (size_t i = 0; i < args.size(); ++i) {
      if (!IsLikelyGuestAddress(args[i])) {
        continue;
      }
      output_arg_index = int(i);
      output_ptr = args[i];
      break;
    }
  }
  if (!IsLikelyGuestAddress(output_ptr)) {
    LogNui("XamNuiSkeletonScoreUpdate: no plausible output pointer found");
    return X_ERROR_SUCCESS;
  }

  auto* out_frame = kernel_memory()->TranslateVirtual<X_NUI_SKELETON_FRAME*>(
      output_ptr);
  if (!out_frame) {
    LogNui("XamNuiSkeletonScoreUpdate: output pointer {:08X} is not mapped",
           output_ptr);
    return X_ERROR_SUCCESS;
  }
  NuiUdpSensorService& sensor = NuiUdpSensorService::Get();
  sensor.EnsureRunning();
  const NuiDebugSnapshot snapshot = sensor.GetDebugSnapshot();
  BuildSkeletonFrame(snapshot, out_frame);
  if (engaged_tracking_id) {
    out_frame->skeletons[0].tracking_id = engaged_tracking_id;
  }

  const auto& skel = out_frame->skeletons[0];
  const auto& head = skel.joints[kNuiSkeletonPositionHead];
  const auto& hip = skel.joints[kNuiSkeletonPositionHipCenter];
  LogNui(
      "XamNuiSkeletonScoreUpdate wrote arg{}={:08X}: tracked={} frame={} tracking_id={} head=({:.3f},{:.3f},{:.3f}) hip=({:.3f},{:.3f},{:.3f})",
      output_arg_index, output_ptr, uint32_t(skel.tracking_state),
      uint32_t(out_frame->frame_number),
      uint32_t(skel.tracking_id), float(head.x), float(head.y), float(head.z),
      float(hip.x), float(hip.y), float(hip.z));

  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamNuiSkeletonScoreUpdate, kNone, kImplemented);

dword_result_t XamNuiGetTrueColorInfo_entry(
    unknown_t arg0, unknown_t arg1, unknown_t arg2, unknown_t arg3,
    unknown_t arg4, unknown_t arg5, unknown_t arg6, unknown_t arg7) {
  const std::array<uint32_t, 8> args = {
      uint32_t(arg0), uint32_t(arg1), uint32_t(arg2), uint32_t(arg3),
      uint32_t(arg4), uint32_t(arg5), uint32_t(arg6), uint32_t(arg7),
  };
  const bool connected = IsNuiDeviceConnected();
  uint32_t texture_ptr = 0;
  uint32_t width = uint32_t(std::clamp(cvars::nui_sensor_color_output_width, 16, 1920));
  uint32_t height =
      uint32_t(std::clamp(cvars::nui_sensor_color_output_height, 16, 1080));
  uint32_t frame_index = 0;
  uint32_t format_fourcc =
      uint32_t(std::max(cvars::nui_sensor_color_format_fourcc, 0));
  const bool has_color = connected &&
                         UpdateNuiColorTextureFromSensor(&texture_ptr, &width, &height,
                                                         &frame_index,
                                                         &format_fourcc);
  const auto output_ptrs = CollectLikelyOutputPointers(args);
  if (!output_ptrs.empty()) {
    WriteGuestU32(output_ptrs[0], width);
  }
  if (output_ptrs.size() > 1) {
    WriteGuestU32(output_ptrs[1], height);
  }
  if (output_ptrs.size() > 2) {
    WriteGuestU32(output_ptrs[2],
                  uint32_t(std::max(cvars::nui_sensor_color_nominal_fps, 1)));
  }
  if (output_ptrs.size() > 3) {
    WriteGuestU32(output_ptrs[3], format_fourcc);
  }
  if (output_ptrs.size() > 4) {
    WriteGuestU32(output_ptrs[4], texture_ptr);
  }
  LogNui(
      "XamNuiGetTrueColorInfo(args=[{:08X},{:08X},{:08X},{:08X},{:08X},{:08X},{:08X},{:08X}]) -> connected={} color={} wh={}x{} fps={} format={:08X} tex={:08X} frame={}",
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
      connected ? "yes" : "no", has_color ? "yes" : "no", width, height,
      uint32_t(std::max(cvars::nui_sensor_color_nominal_fps, 1)),
      format_fourcc, texture_ptr, frame_index);
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiGetTrueColorInfo, kNone, kImplemented);

dword_result_t XamNuiGetCameraIntrinsics_entry(
    unknown_t arg0, unknown_t arg1, unknown_t arg2, unknown_t arg3,
    unknown_t arg4, unknown_t arg5, unknown_t arg6, unknown_t arg7) {
  const std::array<uint32_t, 8> args = {
      uint32_t(arg0), uint32_t(arg1), uint32_t(arg2), uint32_t(arg3),
      uint32_t(arg4), uint32_t(arg5), uint32_t(arg6), uint32_t(arg7),
  };
  const bool connected = IsNuiDeviceConnected();
  const float fx = 525.0f;
  const float fy = 525.0f;
  const float cx = 319.5f;
  const float cy = 239.5f;
  const auto output_ptrs = CollectLikelyOutputPointers(args);
  if (!output_ptrs.empty()) {
    const uint32_t base = output_ptrs[0];
    WriteGuestF32(base + 0, fx);
    WriteGuestF32(base + 4, fy);
    WriteGuestF32(base + 8, cx);
    WriteGuestF32(base + 12, cy);
  }
  if (output_ptrs.size() > 1) {
    WriteGuestF32(output_ptrs[1], fx);
  }
  if (output_ptrs.size() > 2) {
    WriteGuestF32(output_ptrs[2], fy);
  }
  if (output_ptrs.size() > 3) {
    WriteGuestF32(output_ptrs[3], cx);
  }
  if (output_ptrs.size() > 4) {
    WriteGuestF32(output_ptrs[4], cy);
  }
  LogNui(
      "XamNuiGetCameraIntrinsics(args=[{:08X},{:08X},{:08X},{:08X},{:08X},{:08X},{:08X},{:08X}]) -> connected={} fx={:.2f} fy={:.2f} cx={:.2f} cy={:.2f}",
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
      connected ? "yes" : "no", fx, fy, cx, cy);
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiGetCameraIntrinsics, kNone, kImplemented);

dword_result_t XamNuiIdentityGetColorTexture_entry(
    unknown_t arg0, unknown_t arg1, unknown_t arg2, unknown_t arg3,
    unknown_t arg4, unknown_t arg5, unknown_t arg6, unknown_t arg7) {
  const std::array<uint32_t, 8> args = {
      uint32_t(arg0), uint32_t(arg1), uint32_t(arg2), uint32_t(arg3),
      uint32_t(arg4), uint32_t(arg5), uint32_t(arg6), uint32_t(arg7),
  };
  const bool connected = IsNuiDeviceConnected();
  uint32_t texture_ptr = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t frame_index = 0;
  const bool has_color = connected &&
                         UpdateNuiColorTextureFromSensor(&texture_ptr, &width, &height,
                                                         &frame_index, nullptr);
  const auto output_ptrs = CollectLikelyOutputPointers(args);
  if (!output_ptrs.empty()) {
    WriteGuestU32(output_ptrs[0], texture_ptr);
  }
  if (output_ptrs.size() > 1) {
    WriteGuestU32(output_ptrs[1], width);
  }
  if (output_ptrs.size() > 2) {
    WriteGuestU32(output_ptrs[2], height);
  }
  if (output_ptrs.size() > 3) {
    WriteGuestU32(output_ptrs[3], frame_index);
  }
  LogNui(
      "XamNuiIdentityGetColorTexture(args=[{:08X},{:08X},{:08X},{:08X},{:08X},{:08X},{:08X},{:08X}]) -> connected={} color={} tex={:08X} wh={}x{} frame={}",
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
      connected ? "yes" : "no", has_color ? "yes" : "no", texture_ptr, width,
      height, frame_index);
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiIdentityGetColorTexture, kNone, kImplemented);

dword_result_t XamNuiNatalCameraUpdateStarting_entry(
    unknown_t arg0, unknown_t arg1, unknown_t arg2, unknown_t arg3,
    unknown_t arg4, unknown_t arg5, unknown_t arg6, unknown_t arg7) {
  const bool connected = IsNuiDeviceConnected();
  uint32_t texture_ptr = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t frame_index = 0;
  const bool has_color = connected &&
                         UpdateNuiColorTextureFromSensor(&texture_ptr, &width, &height,
                                                         &frame_index, nullptr);
  LogNui(
      "XamNuiNatalCameraUpdateStarting(args=[{:08X},{:08X},{:08X},{:08X},{:08X},{:08X},{:08X},{:08X}]) -> connected={} color={} tex={:08X} wh={}x{} frame={}",
      uint32_t(arg0), uint32_t(arg1), uint32_t(arg2), uint32_t(arg3),
      uint32_t(arg4), uint32_t(arg5), uint32_t(arg6), uint32_t(arg7),
      connected ? "yes" : "no", has_color ? "yes" : "no", texture_ptr, width,
      height, frame_index);
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiNatalCameraUpdateStarting, kNone, kImplemented);

dword_result_t XamNuiNatalCameraUpdateComplete_entry(
    unknown_t arg0, unknown_t arg1, unknown_t arg2, unknown_t arg3,
    unknown_t arg4, unknown_t arg5, unknown_t arg6, unknown_t arg7) {
  const bool connected = IsNuiDeviceConnected();
  uint32_t texture_ptr = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t frame_index = 0;
  const bool has_color = connected &&
                         UpdateNuiColorTextureFromSensor(&texture_ptr, &width, &height,
                                                         &frame_index, nullptr);
  LogNui(
      "XamNuiNatalCameraUpdateComplete(args=[{:08X},{:08X},{:08X},{:08X},{:08X},{:08X},{:08X},{:08X}]) -> connected={} color={} tex={:08X} wh={}x{} frame={}",
      uint32_t(arg0), uint32_t(arg1), uint32_t(arg2), uint32_t(arg3),
      uint32_t(arg4), uint32_t(arg5), uint32_t(arg6), uint32_t(arg7),
      connected ? "yes" : "no", has_color ? "yes" : "no", texture_ptr, width,
      height, frame_index);
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiNatalCameraUpdateComplete, kNone, kImplemented);

dword_result_t XamNuiCameraRememberFloor_entry(
    unknown_t arg0, unknown_t arg1, unknown_t arg2) {
  const bool connected = IsNuiDeviceConnected();
  LogNui("XamNuiCameraRememberFloor({:08X}, {:08X}, {:08X}) -> {}",
         uint32_t(arg0), uint32_t(arg1), uint32_t(arg2),
         connected ? "success" : "not-connected");
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiCameraRememberFloor, kNone, kImplemented);

dword_result_t XamNuiCameraTiltSetCallback_entry(
    unknown_t callback, unknown_t context, unknown_t flags) {
  const bool connected = IsNuiDeviceConnected();
  LogNui(
      "XamNuiCameraTiltSetCallback(cb={:08X}, ctx={:08X}, flags={:08X}) -> {}",
      uint32_t(callback), uint32_t(context), uint32_t(flags),
      connected ? "success" : "not-connected");
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiCameraTiltSetCallback, kNone, kImplemented);

dword_result_t XamNuiCameraTiltReportStatus_entry(unknown_t status, unknown_t unk) {
  const bool connected = IsNuiDeviceConnected();
  LogNui("XamNuiCameraTiltReportStatus(status={:08X}, {:08X}) -> {}",
         uint32_t(status), uint32_t(unk),
         connected ? "success" : "not-connected");
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiCameraTiltReportStatus, kNone, kImplemented);

dword_result_t XamNuiCameraAdjustTilt_entry(int_t direction, int_t magnitude) {
  const bool connected = IsNuiDeviceConnected();
  LogNui("XamNuiCameraAdjustTilt(direction={}, magnitude={}) -> {}",
         int32_t(direction), int32_t(magnitude),
         connected ? "success" : "not-connected");
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiCameraAdjustTilt, kNone, kImplemented);

dword_result_t XamNuiCameraElevationSetAngle_entry(int_t angle) {
  const bool connected = IsNuiDeviceConnected();
  LogNui("XamNuiCameraElevationSetAngle(angle={}) -> {}",
         int32_t(angle), connected ? "success" : "not-connected");
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiCameraElevationSetAngle, kNone, kImplemented);

dword_result_t XamNuiCameraElevationAutoTilt_entry(dword_t enable) {
  const bool connected = IsNuiDeviceConnected();
  LogNui("XamNuiCameraElevationAutoTilt(enable={}) -> {}",
         uint32_t(enable), connected ? "success" : "not-connected");
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiCameraElevationAutoTilt, kNone, kImplemented);

dword_result_t XamNuiCameraElevationReverseAutoTilt_entry(dword_t enable) {
  const bool connected = IsNuiDeviceConnected();
  LogNui("XamNuiCameraElevationReverseAutoTilt(enable={}) -> {}",
         uint32_t(enable), connected ? "success" : "not-connected");
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiCameraElevationReverseAutoTilt, kNone, kImplemented);

dword_result_t XamNuiCameraElevationStopMovement_entry() {
  const bool connected = IsNuiDeviceConnected();
  LogNui("XamNuiCameraElevationStopMovement() -> {}",
         connected ? "success" : "not-connected");
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiCameraElevationStopMovement, kNone, kImplemented);

dword_result_t XamNuiCameraElevationSetCallback_entry(
    unknown_t callback, unknown_t context, unknown_t flags) {
  const bool connected = IsNuiDeviceConnected();
  LogNui(
      "XamNuiCameraElevationSetCallback(cb={:08X}, ctx={:08X}, flags={:08X}) -> {}",
      uint32_t(callback), uint32_t(context), uint32_t(flags),
      connected ? "success" : "not-connected");
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiCameraElevationSetCallback, kNone, kImplemented);

dword_result_t XamNuiCameraTiltGetStatus_entry(lpvoid_t unk) {
  const bool connected = IsNuiDeviceConnected();
  LogNui("XamNuiCameraTiltGetStatus(ptr={:08X}) -> {}",
         unk.guest_address(), connected ? "success" : "not-connected");
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiCameraTiltGetStatus, kNone, kStub);

dword_result_t XamNuiCameraElevationGetAngle_entry(lpqword_t angle,
                                                   lpdword_t status) {
  const bool connected = IsNuiDeviceConnected();
  if (angle) {
    *angle = 0;
  }
  if (status) {
    *status = 0;
  }
  LogNui(
      "XamNuiCameraElevationGetAngle(angle_ptr={:08X}, status_ptr={:08X}) -> {}",
      angle.guest_address(), status.guest_address(),
      connected ? "success" : "not-connected");
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiCameraElevationGetAngle, kNone, kStub);

dword_result_t XamNuiCameraGetTiltControllerType_entry() {
  const bool connected = IsNuiDeviceConnected();
  const uint32_t result = connected ? 1u : 0u;
  LogNui("XamNuiCameraGetTiltControllerType() -> {:08X}", result);
  return result;
}
DECLARE_XAM_EXPORT1(XamNuiCameraGetTiltControllerType, kNone, kStub);

dword_result_t XamNuiCameraSetFlags_entry(qword_t unk1, dword_t unk2) {
  const bool connected = IsNuiDeviceConnected();
  LogNui("XamNuiCameraSetFlags({:016X}, {:08X}) -> {}",
         uint64_t(unk1), uint32_t(unk2),
         connected ? "success" : "not-connected");
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiCameraSetFlags, kNone, kStub);

dword_result_t XamIsNuiUIActive_entry() { return xam_dialogs_shown_ > 0; }
DECLARE_XAM_EXPORT1(XamIsNuiUIActive, kNone, kImplemented);

dword_result_t XamNuiIsDeviceReady_entry() {
  const bool connected = IsNuiDeviceConnected();
  LogNui("XamNuiIsDeviceReady() -> {}", connected ? "1" : "0");
  return connected;
}
DECLARE_XAM_EXPORT1(XamNuiIsDeviceReady, kNone, kImplemented);

dword_result_t XamIsNuiAutomationEnabled_entry(unknown_t unk1, unknown_t unk2) {
  LogNui("XamIsNuiAutomationEnabled({:08X}, {:08X}) -> success",
         uint32_t(unk1), uint32_t(unk2));
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT2(XamIsNuiAutomationEnabled, kNone, kStub, kHighFrequency);

dword_result_t XamIsNatalPlaybackEnabled_entry(unknown_t unk1, unknown_t unk2) {
  LogNui("XamIsNatalPlaybackEnabled({:08X}, {:08X}) -> success",
         uint32_t(unk1), uint32_t(unk2));
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT2(XamIsNatalPlaybackEnabled, kNone, kStub, kHighFrequency);

dword_result_t XamNuiIsChatMicEnabled_entry() {
  LogNui("XamNuiIsChatMicEnabled() -> {}", cvars::nui_chat_mic_enabled ? "1"
                                                                        : "0");
  return cvars::nui_chat_mic_enabled ? 1u : 0u;
}
DECLARE_XAM_EXPORT1(XamNuiIsChatMicEnabled, kNone, kImplemented);

dword_result_t XamNuiHudSetEngagedTrackingID_entry(dword_t tracking_id) {
  engaged_tracking_id = tracking_id;
  LogNui("XamNuiHudSetEngagedTrackingID({:08X})", uint32_t(tracking_id));
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamNuiHudSetEngagedTrackingID, kNone, kImplemented);

qword_result_t XamNuiHudGetEngagedTrackingID_entry() {
  LogNui("XamNuiHudGetEngagedTrackingID() -> {:08X}", engaged_tracking_id);
  return engaged_tracking_id;
}
DECLARE_XAM_EXPORT1(XamNuiHudGetEngagedTrackingID, kNone, kImplemented);

dword_result_t XamNuiHudIsEnabled_entry() {
  const bool connected = IsNuiDeviceConnected();
  LogNui("XamNuiHudIsEnabled() -> {}", connected ? "1" : "0");
  return connected ? 1u : 0u;
}
DECLARE_XAM_EXPORT1(XamNuiHudIsEnabled, kNone, kImplemented);

dword_result_t XamNuiHudGetInitializeFlags_entry() {
  const bool connected = IsNuiDeviceConnected();
  const uint32_t flags =
      connected ? uint32_t(std::max(cvars::nui_hud_initialize_flags, 0)) : 0u;
  LogNui("XamNuiHudGetInitializeFlags() -> {:08X}", flags);
  return flags;
}
DECLARE_XAM_EXPORT1(XamNuiHudGetInitializeFlags, kNone, kImplemented);

void XamNuiHudGetVersions_entry(lpqword_t unk1, lpqword_t unk2) {
  const bool connected = IsNuiDeviceConnected();
  const uint64_t version = connected ? 0x0001000000000000ull : 0ull;
  if (unk1) {
    *unk1 = version;
  }
  if (unk2) {
    *unk2 = version;
  }
  LogNui("XamNuiHudGetVersions(ptr1={:08X}, ptr2={:08X}) -> {:016X}",
         unk1.guest_address(), unk2.guest_address(), version);
}
DECLARE_XAM_EXPORT1(XamNuiHudGetVersions, kNone, kImplemented);

qword_result_t XamNuiHudGetEngagedEnrollmentIndex_entry() {
  LogNui("XamNuiHudGetEngagedEnrollmentIndex() -> {:08X}",
         engaged_enrollment_index);
  return engaged_enrollment_index;
}
DECLARE_XAM_EXPORT1(XamNuiHudGetEngagedEnrollmentIndex, kNone, kImplemented);

dword_result_t XamNuiHudEnableInputFilter_entry(dword_t enable) {
  const bool connected = IsNuiDeviceConnected();
  LogNui("XamNuiHudEnableInputFilter(enable={}) -> {}", uint32_t(enable),
         connected ? "success" : "not-connected");
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiHudEnableInputFilter, kNone, kImplemented);

dword_result_t XamNuiHudInterpretFrame_entry(
    unknown_t arg0, unknown_t arg1, unknown_t arg2, unknown_t arg3,
    unknown_t arg4, unknown_t arg5, unknown_t arg6, unknown_t arg7) {
  const std::array<uint32_t, 8> args = {
      uint32_t(arg0), uint32_t(arg1), uint32_t(arg2), uint32_t(arg3),
      uint32_t(arg4), uint32_t(arg5), uint32_t(arg6), uint32_t(arg7),
  };
  const bool connected = IsNuiDeviceConnected();
  const bool tracked = IsNuiSkeletonTracked();
  const auto output_ptrs = CollectLikelyOutputPointers(args);
  if (!output_ptrs.empty()) {
    WriteGuestU32(output_ptrs[0], tracked ? 1u : 0u);
  }
  if (output_ptrs.size() > 1) {
    WriteGuestU32(output_ptrs[1], engaged_tracking_id);
  }
  LogNui(
      "XamNuiHudInterpretFrame(args=[{:08X},{:08X},{:08X},{:08X},{:08X},{:08X},{:08X},{:08X}]) -> connected={} tracked={}",
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
      connected ? "yes" : "no", tracked ? "yes" : "no");
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiHudInterpretFrame, kNone, kImplemented);

dword_result_t XamNuiGetDeviceSerialNumber_entry(
    unknown_t arg0, unknown_t arg1, unknown_t arg2, unknown_t arg3) {
  const std::array<uint32_t, 8> args = {
      uint32_t(arg0), uint32_t(arg1), uint32_t(arg2), uint32_t(arg3), 0, 0, 0, 0};
  const bool connected = IsNuiDeviceConnected();
  static constexpr const char kSerial[] = "XNUI-EMU-0001";
  const auto output_ptrs = CollectLikelyOutputPointers(args);
  if (!output_ptrs.empty()) {
    WriteGuestBytes(output_ptrs[0], kSerial, sizeof(kSerial));
  }
  if (output_ptrs.size() > 1) {
    WriteGuestU32(output_ptrs[1], uint32_t(sizeof(kSerial) - 1));
  }
  LogNui(
      "XamNuiGetDeviceSerialNumber({:08X}, {:08X}, {:08X}, {:08X}) -> connected={} serial={}",
      uint32_t(arg0), uint32_t(arg1), uint32_t(arg2), uint32_t(arg3),
      connected ? "yes" : "no", kSerial);
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiGetDeviceSerialNumber, kNone, kImplemented);

dword_result_t XamNuiGetFanRate_entry(lpdword_t out_rate) {
  const bool connected = IsNuiDeviceConnected();
  if (out_rate) {
    *out_rate = 0;
  }
  LogNui("XamNuiGetFanRate(ptr={:08X}) -> {}", out_rate.guest_address(),
         connected ? "success" : "not-connected");
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiGetFanRate, kNone, kImplemented);

dword_result_t XamNuiGetSupportString_entry(
    unknown_t arg0, unknown_t arg1, unknown_t arg2, unknown_t arg3) {
  const std::array<uint32_t, 8> args = {
      uint32_t(arg0), uint32_t(arg1), uint32_t(arg2), uint32_t(arg3), 0, 0, 0, 0};
  const bool connected = IsNuiDeviceConnected();
  static constexpr const char kSupport[] = "Xenia Kinect Bridge RGB1";
  const auto output_ptrs = CollectLikelyOutputPointers(args);
  if (!output_ptrs.empty()) {
    WriteGuestBytes(output_ptrs[0], kSupport, sizeof(kSupport));
  }
  if (output_ptrs.size() > 1) {
    WriteGuestU32(output_ptrs[1], uint32_t(sizeof(kSupport) - 1));
  }
  LogNui(
      "XamNuiGetSupportString({:08X}, {:08X}, {:08X}, {:08X}) -> connected={} support={}",
      uint32_t(arg0), uint32_t(arg1), uint32_t(arg2), uint32_t(arg3),
      connected ? "yes" : "no", kSupport);
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiGetSupportString, kNone, kImplemented);

dword_result_t XamNuiGetSystemGestureControl_entry(lpdword_t out_value) {
  const bool connected = IsNuiDeviceConnected();
  if (out_value) {
    *out_value = nui_system_gesture_control;
  }
  LogNui("XamNuiGetSystemGestureControl(ptr={:08X}) -> {:08X}",
         out_value.guest_address(), nui_system_gesture_control);
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiGetSystemGestureControl, kNone, kImplemented);

dword_result_t XamShowNuiTroubleshooterUI_entry(unknown_t unk1, unknown_t unk2,
                                                unknown_t unk3) {
  LogNui("XamShowNuiTroubleshooterUI({:08X}, {:08X}, {:08X})", uint32_t(unk1),
         uint32_t(unk2), uint32_t(unk3));
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
  LogNui("XamShowNuiGuideUI({:08X}, {:08X})", uint32_t(unk1), uint32_t(unk2));
  return XeXamNuiHudCheck(0);
}
DECLARE_XAM_EXPORT1(XamShowNuiGuideUI, kNone, kStub);

dword_result_t XamNuiIdentityGetEnrollmentInfo_entry(
    unknown_t arg0, unknown_t arg1, unknown_t arg2, unknown_t arg3,
    unknown_t arg4, unknown_t arg5, unknown_t arg6, unknown_t arg7) {
  const std::array<uint32_t, 8> args = {
      uint32_t(arg0), uint32_t(arg1), uint32_t(arg2), uint32_t(arg3),
      uint32_t(arg4), uint32_t(arg5), uint32_t(arg6), uint32_t(arg7),
  };
  const bool connected = IsNuiDeviceConnected();
  const bool tracked = IsNuiSkeletonTracked();
  const auto output_ptrs = CollectLikelyOutputPointers(args);
  if (!output_ptrs.empty()) {
    WriteGuestU32(output_ptrs[0], engaged_enrollment_index);
  }
  if (output_ptrs.size() > 1) {
    WriteGuestU32(output_ptrs[1], engaged_tracking_id);
  }
  if (output_ptrs.size() > 2) {
    WriteGuestU32(output_ptrs[2], tracked ? 1u : 0u);
  }
  if (output_ptrs.size() > 3) {
    WriteGuestU32(output_ptrs[3], connected ? 1u : 0u);
  }
  LogNui(
      "XamNuiIdentityGetEnrollmentInfo(args=[{:08X},{:08X},{:08X},{:08X},{:08X},{:08X},{:08X},{:08X}]) -> connected={} tracked={} enrollment={} tracking_id={:08X}",
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
      connected ? "yes" : "no", tracked ? "yes" : "no",
      engaged_enrollment_index, engaged_tracking_id);
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiIdentityGetEnrollmentInfo, kNone, kImplemented);

dword_result_t XamNuiIdentityUnenroll_entry(
    unknown_t arg0, unknown_t arg1, unknown_t arg2, unknown_t arg3,
    unknown_t arg4, unknown_t arg5, unknown_t arg6, unknown_t arg7) {
  const std::array<uint32_t, 8> args = {
      uint32_t(arg0), uint32_t(arg1), uint32_t(arg2), uint32_t(arg3),
      uint32_t(arg4), uint32_t(arg5), uint32_t(arg6), uint32_t(arg7),
  };
  const bool connected = IsNuiDeviceConnected();
  engaged_enrollment_index = 0;
  const auto output_ptrs = CollectLikelyOutputPointers(args);
  if (!output_ptrs.empty()) {
    WriteGuestU32(output_ptrs[0], connected ? 1u : 0u);
  }
  LogNui(
      "XamNuiIdentityUnenroll(args=[{:08X},{:08X},{:08X},{:08X},{:08X},{:08X},{:08X},{:08X}]) -> {}",
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
      connected ? "success" : "not-connected");
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiIdentityUnenroll, kNone, kImplemented);

dword_result_t XamNuiIdentityGetQualityFlags_entry(
    unknown_t arg0, unknown_t arg1, unknown_t arg2, unknown_t arg3,
    unknown_t arg4, unknown_t arg5, unknown_t arg6, unknown_t arg7) {
  const std::array<uint32_t, 8> args = {
      uint32_t(arg0), uint32_t(arg1), uint32_t(arg2), uint32_t(arg3),
      uint32_t(arg4), uint32_t(arg5), uint32_t(arg6), uint32_t(arg7),
  };
  const bool connected = IsNuiDeviceConnected();
  const bool tracked = IsNuiSkeletonTracked();
  const uint32_t quality_flags = tracked ? 0u : 1u;
  const auto output_ptrs = CollectLikelyOutputPointers(args);
  if (!output_ptrs.empty()) {
    WriteGuestU32(output_ptrs[0], quality_flags);
  }
  if (output_ptrs.size() > 1) {
    WriteGuestU32(output_ptrs[1], tracked ? 1u : 0u);
  }
  LogNui(
      "XamNuiIdentityGetQualityFlags(args=[{:08X},{:08X},{:08X},{:08X},{:08X},{:08X},{:08X},{:08X}]) -> connected={} tracked={} flags={:08X}",
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
      connected ? "yes" : "no", tracked ? "yes" : "no", quality_flags);
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiIdentityGetQualityFlags, kNone, kImplemented);

dword_result_t XamNuiIdentityGetQualityFlagsMessage_entry(
    unknown_t arg0, unknown_t arg1, unknown_t arg2, unknown_t arg3,
    unknown_t arg4, unknown_t arg5, unknown_t arg6, unknown_t arg7) {
  const std::array<uint32_t, 8> args = {
      uint32_t(arg0), uint32_t(arg1), uint32_t(arg2), uint32_t(arg3),
      uint32_t(arg4), uint32_t(arg5), uint32_t(arg6), uint32_t(arg7),
  };
  const bool connected = IsNuiDeviceConnected();
  const bool tracked = IsNuiSkeletonTracked();
  const char* message = tracked ? "tracking-ok" : "tracking-searching";
  const auto output_ptrs = CollectLikelyOutputPointers(args);
  if (!output_ptrs.empty()) {
    WriteGuestBytes(output_ptrs[0], message, std::strlen(message) + 1);
  }
  if (output_ptrs.size() > 1) {
    WriteGuestU32(output_ptrs[1], uint32_t(std::strlen(message)));
  }
  if (output_ptrs.size() > 2) {
    WriteGuestU32(output_ptrs[2], tracked ? 0u : 1u);
  }
  LogNui(
      "XamNuiIdentityGetQualityFlagsMessage(args=[{:08X},{:08X},{:08X},{:08X},{:08X},{:08X},{:08X},{:08X}]) -> connected={} tracked={} msg={}",
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
      connected ? "yes" : "no", tracked ? "yes" : "no", message);
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiIdentityGetQualityFlagsMessage, kNone, kImplemented);

dword_result_t XamNuiIdentityIdentifyWithBiometric_entry(
    unknown_t arg0, unknown_t arg1, unknown_t arg2, unknown_t arg3,
    unknown_t arg4, unknown_t arg5, unknown_t arg6, unknown_t arg7) {
  const std::array<uint32_t, 8> args = {
      uint32_t(arg0), uint32_t(arg1), uint32_t(arg2), uint32_t(arg3),
      uint32_t(arg4), uint32_t(arg5), uint32_t(arg6), uint32_t(arg7),
  };
  const bool connected = IsNuiDeviceConnected();
  const bool tracked = IsNuiSkeletonTracked();
  if (tracked && !engaged_tracking_id) {
    engaged_tracking_id = 1;
  }
  const auto output_ptrs = CollectLikelyOutputPointers(args);
  if (!output_ptrs.empty()) {
    WriteGuestU32(output_ptrs[0], tracked ? 0u : 0xFFFFFFFFu);
  }
  if (output_ptrs.size() > 1) {
    WriteGuestU32(output_ptrs[1], engaged_tracking_id);
  }
  if (output_ptrs.size() > 2) {
    WriteGuestU32(output_ptrs[2], tracked ? 1u : 0u);
  }
  LogNui(
      "XamNuiIdentityIdentifyWithBiometric(args=[{:08X},{:08X},{:08X},{:08X},{:08X},{:08X},{:08X},{:08X}]) -> connected={} tracked={} tracking_id={:08X}",
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
      connected ? "yes" : "no", tracked ? "yes" : "no", engaged_tracking_id);
  return connected ? X_ERROR_SUCCESS : X_E_DEVICE_NOT_CONNECTED;
}
DECLARE_XAM_EXPORT1(XamNuiIdentityIdentifyWithBiometric, kNone, kImplemented);

qword_result_t XamNuiIdentityGetSessionId_entry() {
  if (!nui_session_id) {
    nui_session_id = 0xDEADF00DDEADF00Dull;
  }
  LogNui("XamNuiIdentityGetSessionId() -> {:016X}", nui_session_id);
  return nui_session_id;
}
DECLARE_XAM_EXPORT1(XamNuiIdentityGetSessionId, kNone, kImplemented);

dword_result_t XamNuiIdentityEnrollForSignIn_entry(dword_t unk1, qword_t unk2,
                                                   qword_t unk3, dword_t unk4) {
  const bool enabled = XamNuiHudIsEnabled_entry() != 0;
  LogNui(
      "XamNuiIdentityEnrollForSignIn({:08X}, {:016X}, {:016X}, {:08X}) -> {}",
      uint32_t(unk1), uint64_t(unk2), uint64_t(unk3), uint32_t(unk4),
      enabled ? "success" : "fail");
  return enabled ? X_ERROR_SUCCESS : X_E_FAIL;
}
DECLARE_XAM_EXPORT1(XamNuiIdentityEnrollForSignIn, kNone, kStub);

dword_result_t XamNuiIdentityAbort_entry(dword_t unk) {
  const bool enabled = XamNuiHudIsEnabled_entry() != 0;
  LogNui("XamNuiIdentityAbort({:08X}) -> {}", uint32_t(unk),
         enabled ? "success" : "fail");
  return enabled ? X_ERROR_SUCCESS : X_E_FAIL;
}
DECLARE_XAM_EXPORT1(XamNuiIdentityAbort, kNone, kStub);

dword_result_t XamUserNuiEnableBiometric_entry(dword_t user_index,
                                               int_t enable) {
  const bool connected = IsNuiDeviceConnected();
  const bool valid_user = user_index == 0;
  const uint32_t result =
      (connected && valid_user) ? X_ERROR_SUCCESS : X_E_NO_SUCH_USER;
  LogNui("XamUserNuiEnableBiometric(user_index={}, enable={}) -> {:08X}",
         uint32_t(user_index), int32_t(enable), result);
  return result;
}
DECLARE_XAM_EXPORT1(XamUserNuiEnableBiometric, kNone, kStub);

void XamNuiPlayerEngagementUpdate_entry(qword_t unk1, unknown_t unk2,
                                        lpunknown_t unk3) {
  LogNui("XamNuiPlayerEngagementUpdate({:016X}, {:08X}, ptr={:08X})",
         uint64_t(unk1), uint32_t(unk2), unk3.guest_address());
}
DECLARE_XAM_EXPORT1(XamNuiPlayerEngagementUpdate, kNone, kStub);

}  // namespace xam
}  // namespace kernel
}  // namespace xe

DECLARE_XAM_EMPTY_REGISTER_EXPORTS(NUI);
