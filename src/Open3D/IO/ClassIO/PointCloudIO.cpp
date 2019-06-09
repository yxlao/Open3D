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

#include "Open3D/IO/ClassIO/PointCloudIO.h"

#include <unordered_map>

#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/FileSystem.h"

namespace open3d {

namespace {
using namespace io;

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &, geometry::PointCloud &)>>
        file_extension_to_pointcloud_read_function{
                {"xyz", ReadPointCloudFromXYZ},
                {"xyzn", ReadPointCloudFromXYZN},
                {"xyzrgb", ReadPointCloudFromXYZRGB},
                {"ply", ReadPointCloudFromPLY},
                {"pcd", ReadPointCloudFromPCD},
                {"pts", ReadPointCloudFromPTS},
        };

static const std::unordered_map<std::string,
                                std::function<bool(const std::string &,
                                                   const geometry::PointCloud &,
                                                   const bool,
                                                   const bool)>>
        file_extension_to_pointcloud_write_function{
                {"xyz", WritePointCloudToXYZ},
                {"xyzn", WritePointCloudToXYZN},
                {"xyzrgb", WritePointCloudToXYZRGB},
                {"ply", WritePointCloudToPLY},
                {"pcd", WritePointCloudToPCD},
                {"pts", WritePointCloudToPTS},
        };
}  // unnamed namespace

namespace io {

std::shared_ptr<geometry::PointCloud> CreatePointCloudFromFile(
        const std::string &filename, const std::string &format) {
    auto pointcloud = std::make_shared<geometry::PointCloud>();
    ReadPointCloud(filename, *pointcloud, format);
    return pointcloud;
}

bool ReadPointCloud(const std::string &filename,
                    geometry::PointCloud &pointcloud,
                    const std::string &format) {
    std::string filename_ext;
    if (format == "auto") {
        filename_ext =
                utility::filesystem::GetFileExtensionInLowerCase(filename);
    } else {
        filename_ext = format;
    }
    if (filename_ext.empty()) {
        utility::PrintWarning(
                "Read geometry::PointCloud failed: unknown file extension.\n");
        return false;
    }
    auto map_itr =
            file_extension_to_pointcloud_read_function.find(filename_ext);
    if (map_itr == file_extension_to_pointcloud_read_function.end()) {
        utility::PrintWarning(
                "Read geometry::PointCloud failed: unknown file extension.\n");
        return false;
    }
    bool success = map_itr->second(filename, pointcloud);
    utility::PrintDebug("Read geometry::PointCloud: %d vertices.\n",
                        (int)pointcloud.points_.size());
    return success;
}

bool WritePointCloud(const std::string &filename,
                     const geometry::PointCloud &pointcloud,
                     bool write_ascii /* = false*/,
                     bool compressed /* = false*/) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::PrintWarning(
                "Write geometry::PointCloud failed: unknown file extension.\n");
        return false;
    }
    auto map_itr =
            file_extension_to_pointcloud_write_function.find(filename_ext);
    if (map_itr == file_extension_to_pointcloud_write_function.end()) {
        utility::PrintWarning(
                "Write geometry::PointCloud failed: unknown file extension.\n");
        return false;
    }
    bool success =
            map_itr->second(filename, pointcloud, write_ascii, compressed);
    utility::PrintDebug("Write geometry::PointCloud: %d vertices.\n",
                        (int)pointcloud.points_.size());
    return success;
}

}  // namespace io
}  // namespace open3d
