//
// Created by wei on 11/2/18.
//

#include <UnitTest/UnitTest.h>
#include <IO/IO.h>
#include <Cuda/Geometry/TriangleMeshCuda.h>

TEST(TriangleMeshCuda, GetMinMaxBound) {
    using namespace open3d;

    TriangleMesh mesh;
    ReadTriangleMesh("apt.ply", mesh);
    Eigen::Vector3d min_bound = mesh.GetMinBound();
    Eigen::Vector3d max_bound = mesh.GetMaxBound();

    TriangleMeshCuda mesh_cuda(VertexWithNormal, 40000, 80000);
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