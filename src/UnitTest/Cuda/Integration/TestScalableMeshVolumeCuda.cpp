//
// Created by wei on 10/24/18.
//

#include <Cuda/Integration/ScalableTSDFVolumeCuda.h>
#include <Cuda/Integration/ScalableMeshVolumeCuda.h>
#include <IO/IO.h>
#include <Core/Core.h>

#include "UnitTest.h"

TEST(ScalableMeshVolumeCuda, VertexAllocation) {
    using namespace open3d;
    cv::Mat im = cv::imread("../../examples/TestData/RGBD/depth/apt-022640.png",
                            cv::IMREAD_UNCHANGED);
    ImageCuda<Vector1s> imcuda;
    imcuda.Upload(im);
    auto imcudaf = imcuda.ToFloat(0.001f);

    MonoPinholeCameraCuda intrinsics;
    intrinsics.SetUp();

    float voxel_length = 0.01f;
    TransformCuda extrinsics = TransformCuda::Identity();
    ScalableTSDFVolumeCuda<8> tsdf_volume(10000, 200000,
                                          voxel_length, 3 * voxel_length,
                                          extrinsics);
    Timer timer;
    timer.Start();
    for (int i = 0; i < 10; ++i) {
        tsdf_volume.Integrate(imcudaf, intrinsics, extrinsics);
    }
    timer.Stop();
    PrintInfo("Integration takes: %f milliseconds\n", timer.GetDuration() / 10);

    ScalableMeshVolumeCuda<8> mesher(10000, VertexWithNormal, 100000, 200000);
    mesher.active_subvolumes_ = tsdf_volume.active_subvolume_entry_array().size();

    PrintInfo("Active subvolumes: %d\n", mesher.active_subvolumes_);

    timer.Start();
    int iter = 100;
    for (int i = 0; i < iter; ++i) {
        mesher.MarchingCubes(tsdf_volume);
    }
    timer.Stop();
    PrintInfo("MarchingCubes takes: %f milliseconds\n", timer.GetDuration() / iter);

    std::shared_ptr<TriangleMesh> mesh = mesher.mesh().Download();
    PrintInfo("triangle.size(): %d, vertices.size(): %d, normals.size(): %d\n",
              mesh->triangles_.size(),
              mesh->vertices_.size(),
              mesh->vertex_normals_.size());
    WriteTriangleMeshToPLY("test3.ply", *mesh, true);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}