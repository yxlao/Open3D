//
// Created by wei on 11/2/18.
//

#include <gtest/gtest.h>
#include <Open3D/Open3D.h>
#include <Cuda/Geometry/TriangleMeshCuda.h>

using namespace open3d;
using namespace open3d::geometry;
using namespace open3d::io;
using namespace open3d::cuda;

TEST(TriangleMeshCuda, GetMinMaxBound) {
    using namespace open3d;

    TriangleMesh mesh;
    ReadTriangleMesh("../../../examples/TestData/bathtub_0154.ply", mesh);
    Eigen::Vector3d min_bound = mesh.GetMinBound();
    Eigen::Vector3d max_bound = mesh.GetMaxBound();

    TriangleMeshCuda mesh_cuda(VertexWithNormal, 900000, 1800000);
    mesh_cuda.Upload(mesh);

    Eigen::Vector3d min_bound_cuda = mesh_cuda.GetMinBound();
    Eigen::Vector3d max_bound_cuda = mesh_cuda.GetMaxBound();

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(min_bound_cuda(i), min_bound(i), 1e-2);
        EXPECT_NEAR(max_bound_cuda(i), max_bound(i), 1e-2);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}