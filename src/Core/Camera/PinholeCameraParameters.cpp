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

#include "PinholeCameraParameters.h"

#include <json/json.h>
#include <Core/Utility/Console.h>

namespace open3d {

PinholeCameraParameters::PinholeCameraParameters() {}

PinholeCameraParameters::~PinholeCameraParameters() {}

bool PinholeCameraParameters::ConvertToJsonValue(Json::Value &value) const {
    Json::Value trajectory_array;
    value["class_name"] = "PinholeCameraParameters";
    value["version_major"] = 1;
    value["version_minor"] = 0;
    if (EigenMatrix4dToJsonArray(extrinsic_, value["extrinsic"]) == false) {
        return false;
    }
    if (intrinsic_.ConvertToJsonValue(value["intrinsic"]) == false) {
        return false;
    }
    return true;
}

bool PinholeCameraParameters::ConvertFromJsonValue(const Json::Value &value) {
    if (value.isObject() == false) {
        PrintWarning(
                "PinholeCameraParameters read JSON failed: unsupported json "
                "format.\n");
        return false;
    }
    if (value.get("class_name", "").asString() != "PinholeCameraParameters" ||
        value.get("version_major", 1).asInt() != 1 ||
        value.get("version_minor", 0).asInt() != 0) {
        PrintWarning(
                "PinholeCameraParameters read JSON failed: unsupported json "
                "format.\n");
        return false;
    }
    if (intrinsic_.ConvertFromJsonValue(value["intrinsic"]) == false) {
        return false;
    }
    if (EigenMatrix4dFromJsonArray(extrinsic_, value["extrinsic"]) == false) {
        return false;
    }
    return true;
}

Eigen::Vector3d PinholeCameraParameters::GetCameraCenter() const {
    Eigen::Matrix3d R = extrinsic_.block(0, 0, 3, 3);
    Eigen::Vector3d t = extrinsic_.block(0, 3, 3, 1);
    Eigen::Vector3d camera_center = -R.transpose() * t;
    return camera_center;
}

}  // namespace open3d
