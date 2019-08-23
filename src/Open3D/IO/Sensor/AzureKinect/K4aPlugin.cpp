// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <k4a/k4a.h>
#include <k4arecord/playback.h>
#include <k4arecord/record.h>
#include <cstring>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <link.h>
#endif

#include "Open3D/IO/Sensor/AzureKinect/K4aPlugin.h"
#include "Open3D/IO/Sensor/AzureKinect/PluginMacros.h"
#include "Open3D/Utility/Console.h"

namespace open3d {
namespace io {
namespace k4a_plugin {

#ifdef _WIN32

// clang-format off
static const std::vector<std::string> k4a_lib_path_hints = {
    "",
    "C:\\Program Files\\Azure Kinect SDK v1.2.0\\sdk\\windows-desktop\\amd64\\release\\bin\\"
};
// clang-format on
static const std::string k4a_lib_name = "k4a.dll";
static const std::string k4arecord_lib_name = "k4arecord.dll";

static HINSTANCE GetDynamicLibHandle(const std::string& lib_name) {
    static std::unordered_map<std::string, HINSTANCE> map_lib_name_to_handle;

    if (map_lib_name_to_handle.count(lib_name) == 0) {
        HINSTANCE handle = NULL;
        for (const std::string& k4a_lib_path_hint : k4a_lib_path_hints) {
            std::string full_path = k4a_lib_path_hint + lib_name;
            handle = LoadLibrary(TEXT(full_path.c_str()));
            if (handle != NULL) {
                utility::LogInfo("Loaded {}\n", full_path);
                break;
            }
        }
        if (handle == NULL) {
            utility::LogFatal("Cannot load {}\n", lib_name);
        }
        map_lib_name_to_handle[lib_name] = handle;
    }
    return map_lib_name_to_handle.at(lib_name);
}

#define DEFINE_BRIDGED_FUNC_WITH_COUNT(lib_name, return_type, f_name, \
                                       num_args, ...)                 \
    return_type f_name(EXTRACT_TYPES_PARAMS(num_args, __VA_ARGS__)) { \
        typedef return_type (*f_type)(                                \
                EXTRACT_TYPES_PARAMS(num_args, __VA_ARGS__));         \
        static f_type f = nullptr;                                    \
                                                                      \
        if (!f) {                                                     \
            f = (f_type)GetProcAddress(GetDynamicLibHandle(lib_name), \
                                       #f_name);                      \
            if (f == nullptr) {                                       \
                utility::LogFatal("Cannot load func {}\n", #f_name);  \
            } else {                                                  \
                utility::LogInfo("Loaded func {}\n", #f_name);        \
            }                                                         \
        }                                                             \
        return f(EXTRACT_PARAMS(num_args, __VA_ARGS__));              \
    }

#else

static const std::string k4a_lib_name = "k4arecord.lib";
static const std::string k4arecord_lib_name = "libk4arecord.so";

static void* GetDynamicLibHandle(const std::string& lib_name) {
    static std::unordered_map<std::string, void*> map_lib_name_to_handle;

    if (map_lib_name_to_handle.count(lib_name) == 0) {
        void* handle = dlopen(lib_name.c_str(), RTLD_LAZY);
        if (!handle) {
            utility::LogFatal("Cannot load {}\n", dlerror());
        } else {
            utility::LogInfo("Loaded {}\n", lib_name);
            struct link_map* map = nullptr;
            if (!dlinfo(handle, RTLD_DI_LINKMAP, &map)) {
                if (map != nullptr) {
                    utility::LogInfo("Library path {}\n", map->l_name);
                } else {
                    utility::LogWarning("Cannot get link_map\n");
                }
            } else {
                utility::LogWarning("Cannot get dlinfo\n");
            }
        }
        map_lib_name_to_handle[lib_name] = handle;
    }
    return map_lib_name_to_handle.at(lib_name);
}

#define DEFINE_BRIDGED_FUNC_WITH_COUNT(lib_name, return_type, f_name,          \
                                       num_args, ...)                          \
    return_type f_name(EXTRACT_TYPES_PARAMS(num_args, __VA_ARGS__)) {          \
        typedef return_type (*f_type)(                                         \
                EXTRACT_TYPES_PARAMS(num_args, __VA_ARGS__));                  \
        static f_type f = nullptr;                                             \
                                                                               \
        if (!f) {                                                              \
            f = (f_type)dlsym(GetDynamicLibHandle(lib_name), #f_name);         \
            if (!f) {                                                          \
                utility::LogFatal("Cannot load {}: {}\n", #f_name, dlerror()); \
            }                                                                  \
        }                                                                      \
        return f(EXTRACT_PARAMS(num_args, __VA_ARGS__));                       \
    }

#endif

#define DEFINE_BRIDGED_FUNC(lib_name, return_type, f_name, ...)   \
    DEFINE_BRIDGED_FUNC_WITH_COUNT(lib_name, return_type, f_name, \
                                   COUNT_ARGS(__VA_ARGS__), __VA_ARGS__)

////////////////////////////////////////////////////////////////////////////////

DEFINE_BRIDGED_FUNC(k4arecord_lib_name,
                    k4a_result_t,
                    k4a_record_create,
                    const char*,
                    path,
                    k4a_device_t,
                    device,
                    const k4a_device_configuration_t,
                    device_config,
                    k4a_record_t*,
                    recording_handle)

DEFINE_BRIDGED_FUNC(k4arecord_lib_name,
                    k4a_result_t,
                    k4a_record_add_tag,
                    k4a_record_t,
                    recording_handle,
                    const char*,
                    name,
                    const char*,
                    value)

DEFINE_BRIDGED_FUNC(k4arecord_lib_name,
                    k4a_result_t,
                    k4a_record_add_imu_track,
                    k4a_record_t,
                    recording_handle)

DEFINE_BRIDGED_FUNC(k4arecord_lib_name,
                    k4a_result_t,
                    k4a_record_write_header,
                    k4a_record_t,
                    recording_handle)

DEFINE_BRIDGED_FUNC(k4arecord_lib_name,
                    k4a_result_t,
                    k4a_record_write_capture,
                    k4a_record_t,
                    recording_handle,
                    k4a_capture_t,
                    capture_handle)

DEFINE_BRIDGED_FUNC(k4arecord_lib_name,
                    k4a_result_t,
                    k4a_record_write_imu_sample,
                    k4a_record_t,
                    recording_handle,
                    k4a_imu_sample_t,
                    imu_sample)

DEFINE_BRIDGED_FUNC(k4arecord_lib_name,
                    k4a_result_t,
                    k4a_record_flush,
                    k4a_record_t,
                    recording_handle)

DEFINE_BRIDGED_FUNC(k4arecord_lib_name,
                    void,
                    k4a_record_close,
                    k4a_record_t,
                    recording_handle)

////////////////////////////////////////////////////////////////////////////////

DEFINE_BRIDGED_FUNC(k4arecord_lib_name,
                    k4a_result_t,
                    k4a_playback_open,
                    const char*,
                    path,
                    k4a_playback_t*,
                    playback_handle)
DEFINE_BRIDGED_FUNC(k4arecord_lib_name,
                    k4a_buffer_result_t,
                    k4a_playback_get_raw_calibration,
                    k4a_playback_t,
                    playback_handle,
                    uint8_t*,
                    data,
                    size_t*,
                    data_size)
DEFINE_BRIDGED_FUNC(k4arecord_lib_name,
                    k4a_result_t,
                    k4a_playback_get_calibration,
                    k4a_playback_t,
                    playback_handle,
                    k4a_calibration_t*,
                    calibration)
DEFINE_BRIDGED_FUNC(k4arecord_lib_name,
                    k4a_result_t,
                    k4a_playback_get_record_configuration,
                    k4a_playback_t,
                    playback_handle,
                    k4a_record_configuration_t*,
                    config)
DEFINE_BRIDGED_FUNC(k4arecord_lib_name,
                    k4a_buffer_result_t,
                    k4a_playback_get_tag,
                    k4a_playback_t,
                    playback_handle,
                    const char*,
                    name,
                    char*,
                    value,
                    size_t*,
                    value_size)
DEFINE_BRIDGED_FUNC(k4arecord_lib_name,
                    k4a_result_t,
                    k4a_playback_set_color_conversion,
                    k4a_playback_t,
                    playback_handle,
                    k4a_image_format_t,
                    target_format)
DEFINE_BRIDGED_FUNC(k4arecord_lib_name,
                    k4a_stream_result_t,
                    k4a_playback_get_next_capture,
                    k4a_playback_t,
                    playback_handle,
                    k4a_capture_t*,
                    capture_handle)
DEFINE_BRIDGED_FUNC(k4arecord_lib_name,
                    k4a_stream_result_t,
                    k4a_playback_get_previous_capture,
                    k4a_playback_t,
                    playback_handle,
                    k4a_capture_t*,
                    capture_handle)
DEFINE_BRIDGED_FUNC(k4arecord_lib_name,
                    k4a_stream_result_t,
                    k4a_playback_get_next_imu_sample,
                    k4a_playback_t,
                    playback_handle,
                    k4a_imu_sample_t*,
                    imu_sample)
DEFINE_BRIDGED_FUNC(k4arecord_lib_name,
                    k4a_stream_result_t,
                    k4a_playback_get_previous_imu_sample,
                    k4a_playback_t,
                    playback_handle,
                    k4a_imu_sample_t*,
                    imu_sample)
DEFINE_BRIDGED_FUNC(k4arecord_lib_name,
                    k4a_result_t,
                    k4a_playback_seek_timestamp,
                    k4a_playback_t,
                    playback_handle,
                    int64_t,
                    offset_usec,
                    k4a_playback_seek_origin_t,
                    origin)
DEFINE_BRIDGED_FUNC(k4arecord_lib_name,
                    uint64_t,
                    k4a_playback_get_last_timestamp_usec,
                    k4a_playback_t,
                    playback_handle)
DEFINE_BRIDGED_FUNC(k4arecord_lib_name,
                    void,
                    k4a_playback_close,
                    k4a_playback_t,
                    playback_handle)

////////////////////////////////////////////////////////////////////////////////

DEFINE_BRIDGED_FUNC(k4a_lib_name, uint32_t, k4a_device_get_installed_count)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_result_t,
                    k4a_set_debug_message_handler,
                    k4a_logging_message_cb_t*,
                    message_cb,
                    void*,
                    message_cb_context,
                    k4a_log_level_t,
                    min_level)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_result_t,
                    k4a_device_open,
                    uint32_t,
                    index,
                    k4a_device_t*,
                    device_handle)

DEFINE_BRIDGED_FUNC(
        k4a_lib_name, void, k4a_device_close, k4a_device_t, device_handle)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_wait_result_t,
                    k4a_device_get_capture,
                    k4a_device_t,
                    device_handle,
                    k4a_capture_t*,
                    capture_handle,
                    int32_t,
                    timeout_in_ms)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_wait_result_t,
                    k4a_device_get_imu_sample,
                    k4a_device_t,
                    device_handle,
                    k4a_imu_sample_t*,
                    imu_sample,
                    int32_t,
                    timeout_in_ms)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_result_t,
                    k4a_capture_create,
                    k4a_capture_t*,
                    capture_handle)

DEFINE_BRIDGED_FUNC(
        k4a_lib_name, void, k4a_capture_release, k4a_capture_t, capture_handle)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    void,
                    k4a_capture_reference,
                    k4a_capture_t,
                    capture_handle)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_image_t,
                    k4a_capture_get_color_image,
                    k4a_capture_t,
                    capture_handle)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_image_t,
                    k4a_capture_get_depth_image,
                    k4a_capture_t,
                    capture_handle)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_image_t,
                    k4a_capture_get_ir_image,
                    k4a_capture_t,
                    capture_handle)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    void,
                    k4a_capture_set_color_image,
                    k4a_capture_t,
                    capture_handle,
                    k4a_image_t,
                    image_handle)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    void,
                    k4a_capture_set_depth_image,
                    k4a_capture_t,
                    capture_handle,
                    k4a_image_t,
                    image_handle)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    void,
                    k4a_capture_set_ir_image,
                    k4a_capture_t,
                    capture_handle,
                    k4a_image_t,
                    image_handle)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    void,
                    k4a_capture_set_temperature_c,
                    k4a_capture_t,
                    capture_handle,
                    float,
                    temperature_c)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    float,
                    k4a_capture_get_temperature_c,
                    k4a_capture_t,
                    capture_handle)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_result_t,
                    k4a_image_create,
                    k4a_image_format_t,
                    format,
                    int,
                    width_pixels,
                    int,
                    height_pixels,
                    int,
                    stride_bytes,
                    k4a_image_t*,
                    image_handle)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_result_t,
                    k4a_image_create_from_buffer,
                    k4a_image_format_t,
                    format,
                    int,
                    width_pixels,
                    int,
                    height_pixels,
                    int,
                    stride_bytes,
                    uint8_t*,
                    buffer,
                    size_t,
                    buffer_size,
                    k4a_memory_destroy_cb_t*,
                    buffer_release_cb,
                    void*,
                    buffer_release_cb_context,
                    k4a_image_t*,
                    image_handle)

DEFINE_BRIDGED_FUNC(
        k4a_lib_name, uint8_t*, k4a_image_get_buffer, k4a_image_t, image_handle)

DEFINE_BRIDGED_FUNC(
        k4a_lib_name, size_t, k4a_image_get_size, k4a_image_t, image_handle)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_image_format_t,
                    k4a_image_get_format,
                    k4a_image_t,
                    image_handle)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    int,
                    k4a_image_get_width_pixels,
                    k4a_image_t,
                    image_handle)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    int,
                    k4a_image_get_height_pixels,
                    k4a_image_t,
                    image_handle)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    int,
                    k4a_image_get_stride_bytes,
                    k4a_image_t,
                    image_handle)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    uint64_t,
                    k4a_image_get_timestamp_usec,
                    k4a_image_t,
                    image_handle)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    uint64_t,
                    k4a_image_get_exposure_usec,
                    k4a_image_t,
                    image_handle)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    uint32_t,
                    k4a_image_get_white_balance,
                    k4a_image_t,
                    image_handle)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    uint32_t,
                    k4a_image_get_iso_speed,
                    k4a_image_t,
                    image_handle)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    void,
                    k4a_image_set_timestamp_usec,
                    k4a_image_t,
                    image_handle,
                    uint64_t,
                    timestamp_usec)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    void,
                    k4a_image_set_exposure_time_usec,
                    k4a_image_t,
                    image_handle,
                    uint64_t,
                    exposure_usec)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    void,
                    k4a_image_set_white_balance,
                    k4a_image_t,
                    image_handle,
                    uint32_t,
                    white_balance)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    void,
                    k4a_image_set_iso_speed,
                    k4a_image_t,
                    image_handle,
                    uint32_t,
                    iso_speed)

DEFINE_BRIDGED_FUNC(
        k4a_lib_name, void, k4a_image_reference, k4a_image_t, image_handle)

DEFINE_BRIDGED_FUNC(
        k4a_lib_name, void, k4a_image_release, k4a_image_t, image_handle)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_result_t,
                    k4a_device_start_cameras,
                    k4a_device_t,
                    device_handle,
                    k4a_device_configuration_t*,
                    config)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    void,
                    k4a_device_stop_cameras,
                    k4a_device_t,
                    device_handle)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_result_t,
                    k4a_device_start_imu,
                    k4a_device_t,
                    device_handle)

DEFINE_BRIDGED_FUNC(
        k4a_lib_name, void, k4a_device_stop_imu, k4a_device_t, device_handle)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_buffer_result_t,
                    k4a_device_get_serialnum,
                    k4a_device_t,
                    device_handle,
                    char*,
                    serial_number,
                    size_t*,
                    serial_number_size)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_result_t,
                    k4a_device_get_version,
                    k4a_device_t,
                    device_handle,
                    k4a_hardware_version_t*,
                    version)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_result_t,
                    k4a_device_get_color_control_capabilities,
                    k4a_device_t,
                    device_handle,
                    k4a_color_control_command_t,
                    command,
                    bool*,
                    supports_auto,
                    int32_t*,
                    min_value,
                    int32_t*,
                    max_value,
                    int32_t*,
                    step_value,
                    int32_t*,
                    default_value,
                    k4a_color_control_mode_t*,
                    default_mode)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_result_t,
                    k4a_device_get_color_control,
                    k4a_device_t,
                    device_handle,
                    k4a_color_control_command_t,
                    command,
                    k4a_color_control_mode_t*,
                    mode,
                    int32_t*,
                    value)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_result_t,
                    k4a_device_set_color_control,
                    k4a_device_t,
                    device_handle,
                    k4a_color_control_command_t,
                    command,
                    k4a_color_control_mode_t,
                    mode,
                    int32_t,
                    value)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_buffer_result_t,
                    k4a_device_get_raw_calibration,
                    k4a_device_t,
                    device_handle,
                    uint8_t*,
                    data,
                    size_t*,
                    data_size)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_result_t,
                    k4a_device_get_calibration,
                    k4a_device_t,
                    device_handle,
                    const k4a_depth_mode_t,
                    depth_mode,
                    const k4a_color_resolution_t,
                    color_resolution,
                    k4a_calibration_t*,
                    calibration)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_result_t,
                    k4a_device_get_sync_jack,
                    k4a_device_t,
                    device_handle,
                    bool*,
                    sync_in_jack_connected,
                    bool*,
                    sync_out_jack_connected)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_result_t,
                    k4a_calibration_get_from_raw,
                    char*,
                    raw_calibration,
                    size_t,
                    raw_calibration_size,
                    const k4a_depth_mode_t,
                    depth_mode,
                    const k4a_color_resolution_t,
                    color_resolution,
                    k4a_calibration_t*,
                    calibration)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_result_t,
                    k4a_calibration_3d_to_3d,
                    const k4a_calibration_t*,
                    calibration,
                    const k4a_float3_t*,
                    source_point3d_mm,
                    const k4a_calibration_type_t,
                    source_camera,
                    const k4a_calibration_type_t,
                    target_camera,
                    k4a_float3_t*,
                    target_point3d_mm)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_result_t,
                    k4a_calibration_2d_to_3d,
                    const k4a_calibration_t*,
                    calibration,
                    const k4a_float2_t*,
                    source_point2d,
                    const float,
                    source_depth_mm,
                    const k4a_calibration_type_t,
                    source_camera,
                    const k4a_calibration_type_t,
                    target_camera,
                    k4a_float3_t*,
                    target_point3d_mm,
                    int*,
                    valid)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_result_t,
                    k4a_calibration_3d_to_2d,
                    const k4a_calibration_t*,
                    calibration,
                    const k4a_float3_t*,
                    source_point3d_mm,
                    const k4a_calibration_type_t,
                    source_camera,
                    const k4a_calibration_type_t,
                    target_camera,
                    k4a_float2_t*,
                    target_point2d,
                    int*,
                    valid)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_result_t,
                    k4a_calibration_2d_to_2d,
                    const k4a_calibration_t*,
                    calibration,
                    const k4a_float2_t*,
                    source_point2d,
                    const float,
                    source_depth_mm,
                    const k4a_calibration_type_t,
                    source_camera,
                    const k4a_calibration_type_t,
                    target_camera,
                    k4a_float2_t*,
                    target_point2d,
                    int*,
                    valid)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_transformation_t,
                    k4a_transformation_create,
                    const k4a_calibration_t*,
                    calibration)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    void,
                    k4a_transformation_destroy,
                    k4a_transformation_t,
                    transformation_handle)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_result_t,
                    k4a_transformation_depth_image_to_color_camera,
                    k4a_transformation_t,
                    transformation_handle,
                    const k4a_image_t,
                    depth_image,
                    k4a_image_t,
                    transformed_depth_image)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_result_t,
                    k4a_transformation_color_image_to_depth_camera,
                    k4a_transformation_t,
                    transformation_handle,
                    const k4a_image_t,
                    depth_image,
                    const k4a_image_t,
                    color_image,
                    k4a_image_t,
                    transformed_color_image)

DEFINE_BRIDGED_FUNC(k4a_lib_name,
                    k4a_result_t,
                    k4a_transformation_depth_image_to_point_cloud,
                    k4a_transformation_t,
                    transformation_handle,
                    const k4a_image_t,
                    depth_image,
                    const k4a_calibration_type_t,
                    camera,
                    k4a_image_t,
                    xyz_image)

}  // namespace k4a_plugin
}  // namespace io
}  // namespace open3d
