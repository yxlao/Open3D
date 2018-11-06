//
// Created by wei on 10/24/18.
//

#include <Cuda/Integration/ScalableTSDFVolumeCuda.h>
#include <Cuda/Integration/ScalableMeshVolumeCuda.h>
#include <IO/IO.h>
#include <Core/Core.h>

#include "UnitTest.h"

TEST(ScalableMeshVolumeCuda, MarchingCubes) {
    using namespace open3d;

    cv::Mat depth = cv::imread(
        "../../examples/TestData/RGBD/depth/00000.png",
        cv::IMREAD_UNCHANGED);
    cv::Mat color = cv::imread(
        "../../examples/TestData/RGBD/color/00000.jpg");
    cv::cvtColor(color, color, cv::COLOR_BGR2RGB);

    RGBDImageCuda rgbd(0.1f, 3.5f, 1000.0f);
    rgbd.Upload(depth, color);

    MonoPinholeCameraCuda intrinsics;
    intrinsics.SetUp();

    float voxel_length = 0.01f;
    TransformCuda extrinsics = TransformCuda::Identity();
    extrinsics(0, 3) = 10.0f;
    extrinsics(1, 3) = -10.0f;
    extrinsics(2, 3) = 1.0f;
    ScalableTSDFVolumeCuda<8> tsdf_volume(10000, 200000,
                                          voxel_length, 3 * voxel_length,
                                          extrinsics);
    Timer timer;
    timer.Start();
    for (int i = 0; i < 10; ++i) {
        tsdf_volume.Integrate(rgbd, intrinsics, extrinsics);
    }
    timer.Stop();
    PrintInfo("Integration takes: %f milliseconds\n", timer.GetDuration() / 10);

    tsdf_volume.GetAllSubvolumes();
    ScalableMeshVolumeCuda<8> mesher(10000, VertexWithNormalAndColor, 100000, 200000);
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
    WriteTriangleMeshToPLY("test_scalable.ply", *mesh, true);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}