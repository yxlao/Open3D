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
#include <Eigen/Geometry>

#include "Open3D/TGeometry/Geometry.h"
#include "Open3D/Utility/Eigen.h"

namespace open3d {
namespace tgeometry {

/// \class Geometry3D
///
/// \brief The base geometry class for 3D geometries.
///
/// Main class for 3D geometries.
class Geometry3D : public Geometry {
public:
    ~Geometry3D() override {}

protected:
    Geometry3D(GeometryType type) : Geometry(type, 3) {}

public:
    Geometry3D& Clear() override = 0;

    bool IsEmpty() const override = 0;

    /// Returns min bounds for geometry coordinates.
    virtual Eigen::Vector3d GetMinBound() const = 0;
};

}  // namespace tgeometry
}  // namespace open3d
