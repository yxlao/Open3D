//
// Created by wei on 10/10/20.
//

#include <Cuda/Integration/UniformTSDFVolumeCuda.h>
#include <Cuda/Integration/UniformMeshVolumeCuda.h>
#include <Cuda/Geometry/RGBDImageCuda.h>
#include <Cuda/Geometry/VectorCuda.h>
#include <Core/Core.h>
#include <Eigen/Eigen>
#include <IO/IO.h>

#include "UnitTest.h"

#include <opencv2/opencv.hpp>
#include <vector>

TEST(UniformMeshVolumeCuda, MarchingCubes) {
    using namespace open3d;
    cv::Mat depth = cv::imread("../../examples/TestData/RGBD/depth/00000.png",
                            cv::IMREAD_UNCHANGED);
    cv::Mat color = cv::imread("../../examples/TestData/RGBD/color/00000.jpg");
    cv::cvtColor(color, color, cv::COLOR_BGR2RGB);

    RGBDImageCuda rgbd(0.1f, 3.0f, 1000.0f);
    rgbd.Upload(depth, color);

    PinholeCameraIntrinsicCuda intrinsics(
        PinholeCameraIntrinsicParameters::PrimeSenseDefault);

    TransformCuda transform = TransformCuda::Identity();

    const float voxel_length = 0.01f;
    transform.SetTranslation(Vector3f(-voxel_length * 256));
    UniformTSDFVolumeCuda<512> volume(voxel_length, voxel_length * 3, transform);

    TransformCuda extrinsics = TransformCuda::Identity();
    volume.Integrate(rgbd, intrinsics, extrinsics);

    UniformMeshVolumeCuda<512> mesher(VertexWithNormalAndColor, 100000, 100000);

    Timer timer;
    timer.Start();
    for (int i = 0; i < 1; ++i) {
        mesher.MarchingCubes(volume);
    }
    timer.Stop();
    PrintInfo("MarchingCubes time: %f milliseconds\n", timer.GetDuration() / 10);

    std::shared_ptr<TriangleMesh> mesh = mesher.mesh().Download();
    WriteTriangleMeshToPLY("test_uniform.ply", *mesh, true);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}