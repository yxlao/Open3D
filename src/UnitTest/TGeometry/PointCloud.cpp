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

#include "Open3D/TGeometry/PointCloud.h"
#include "Open3D/Core/TensorList.h"
#include "TestUtility/UnitTest.h"

namespace open3d {
namespace tgeometry {

TEST(TPointCloud, DefaultConstructor) {
    tgeometry::PointCloud pc;

    // Inherited from Geometry3D.
    EXPECT_EQ(pc.GetGeometryType(), Geometry::GeometryType::PointCloud);
    EXPECT_EQ(pc.Dimension(), 3);

    // Public members.
    EXPECT_TRUE(pc.IsEmpty());
    EXPECT_FALSE(pc.HasPoints());
}

TEST(TPointCloud, GetMinBound) {
    tgeometry::PointCloud pc(Dtype::Float32, Device("CPU:0"));

    TensorList& points = pc.point_dict_["points"];
    points.PushBack(Tensor(std::vector<float>{1, 2, 3}, {3}, Dtype::Float32));
    points.PushBack(Tensor(std::vector<float>{4, 5, 6}, {3}, Dtype::Float32));

    EXPECT_FALSE(pc.IsEmpty());
    EXPECT_TRUE(pc.HasPoints());
    EXPECT_EQ(pc.GetMinBound().ToFlatVector<float>(),
              std::vector<float>({1, 2, 3}));
}

}  // namespace tgeometry
}  // namespace open3d
