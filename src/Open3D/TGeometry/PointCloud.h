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

#pragma once

#include <Eigen/Core>
#include <string>
#include <unordered_map>

#include "Open3D/Core/Tensor.h"
#include "Open3D/Core/TensorList.h"
#include "Open3D/TGeometry/Geometry3D.h"

namespace open3d {
namespace tgeometry {

/// \class PointCloud
/// \brief A PointCloud contains point coordinates, and optionally point colors
/// and point normals.
class PointCloud : public Geometry3D {
public:
    PointCloud(Dtype dtype = Dtype::Float32, Device device = Device("CPU:0"))
        : Geometry3D(Geometry::GeometryType::PointCloud),
          dtype_(dtype),
          device_(device) {
        point_dict_["points"] = TensorList({3}, dtype_, device_);
    }

    ~PointCloud() override {}

    PointCloud &Clear() override;

    bool IsEmpty() const override;

    Tensor GetMinBound() const override;

public:
    bool HasPoints() const {
        return point_dict_.find("points") != point_dict_.end() &&
               point_dict_.at("points").GetSize() > 0;
    }

    bool HasColors() const {
        return point_dict_.find("colors") != point_dict_.end() &&
               point_dict_.at("colors").GetSize() > 0;
    }

    bool HasNormals() const {
        return point_dict_.find("normals") != point_dict_.end() &&
               point_dict_.at("normals").GetSize() > 0;
    }

public:
    std::unordered_map<std::string, TensorList> point_dict_;

protected:
    Dtype dtype_ = Dtype::Float32;
    Device device_ = Device("CPU:0");
};

}  // namespace tgeometry
}  // namespace open3d
