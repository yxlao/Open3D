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

#include "TestUtility/UnitTest.h"

#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Geometry/VoxelGrid.h"
#include "Open3D/Visualization/Utility/DrawGeometry.h"

using namespace Eigen;
using namespace open3d;
using namespace std;
using namespace unit_test;

TEST(Octree, ToVoxelGrid) {
    std::shared_ptr<geometry::VoxelGrid> voxel_grid =
            std::make_shared<geometry::VoxelGrid>();
    voxel_grid->origin_ = Eigen::Vector3d(0, 0, 0);
    voxel_grid->voxel_size_ = 5;
    voxel_grid->voxels_ = {Eigen::Vector3i(0, 0, 0), Eigen::Vector3i(0, 1, 0)};
    voxel_grid->colors_ = {Eigen::Vector3d(0.9, 0, 0),
                           Eigen::Vector3d(0.9, 0.9, 0)};
    visualization::DrawGeometries({voxel_grid});
}

TEST(Octree, ToMesh) {
    std::shared_ptr<geometry::TriangleMesh> mesh =
            std::make_shared<geometry::TriangleMesh>();
    std::vector<Eigen::Vector3d> vertices{
            Eigen::Vector3d(-1, 2, 0), Eigen::Vector3d(1, 2, 0),
            Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(2, 0, 0)};
    std::vector<Eigen::Vector3i> triangles{Eigen::Vector3i(0, 2, 1),
                                           Eigen::Vector3i(1, 2, 3)};
    mesh->vertices_ = vertices;
    mesh->triangles_ = triangles;
    visualization::DrawGeometries({mesh});
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(FeatureIO, DISABLED_ReadFeature) { unit_test::NotImplemented(); }

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(FeatureIO, DISABLED_WriteFeature) { unit_test::NotImplemented(); }

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(FeatureIO, DISABLED_ReadFeatureFromBIN) { unit_test::NotImplemented(); }

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(FeatureIO, DISABLED_WriteFeatureToBIN) { unit_test::NotImplemented(); }
