//
// Created by wei on 10/10/18.
//

#include <Cuda/Integration/UniformTSDFVolumeCuda.h>
#include <Cuda/Integration/UniformMeshVolumeCuda.h>
#include <Cuda/Geometry/VectorCuda.h>
#include <Core/Core.h>
#include <Eigen/Eigen>
#include <IO/IO.h>

#include "UnitTest.h"

#include <opencv2/opencv.hpp>
#include <vector>

TEST(UniformTSDFVolumeCuda, UploadAndDownload) {
    //const size_t N = 512;
    const size_t N = 16;
    const size_t NNN = N * N * N;
    std::vector<float> tsdf;
    std::vector<uchar> weight;
    std::vector<open3d::Vector3b> color;
    tsdf.resize(NNN);
    weight.resize(NNN);
    color.resize(NNN);

    open3d::UniformTSDFVolumeCuda<N> volume;
    volume.Create();

    int cnt = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                tsdf[cnt] = i * j * k;
                weight[cnt] = uchar(fminf(i, 255));
                color[cnt] = open3d::Vector3b(i, j, k);
                ++cnt;
            }
        }
    }

    volume.UploadVolume(tsdf, weight, color);
    tsdf.clear();
    weight.clear();
    color.clear();

    auto downloaded = volume.DownloadVolume();
    tsdf = std::get<0>(downloaded);
    weight = std::get<1>(downloaded);
    color = std::get<2>(downloaded);

    cnt = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                EXPECT_NEAR(tsdf[cnt], i * j * k, NNN * 1e-6);
                EXPECT_EQ(weight[cnt], uchar(fminf(i, 255)));
                EXPECT_EQ(color[cnt], open3d::Vector3b(i, j, k));
                ++cnt;
            }
        }
    }
}

TEST(UniformTSDFVolumeCuda, RayCasting) {
    using namespace open3d;
    cv::Mat depth = cv::imread("../../examples/TestData/RGBD/depth/00000.png",
                            cv::IMREAD_UNCHANGED);
    cv::Mat color = cv::imread("../../examples/TestData/RGBD/color/00000.jpg");

    RGBDImageCuda rgbd;
    rgbd.Upload(depth, color);

    PinholeCameraIntrinsicCuda intrinsics(
        PinholeCameraIntrinsicParameters::PrimeSenseDefault);

    TransformCuda transform = TransformCuda::Identity();

    const float voxel_length = 0.01f;
    transform.SetTranslation(Vector3f(-voxel_length * 256));
    UniformTSDFVolumeCuda<512> volume(voxel_length, voxel_length * 3, transform);

    TransformCuda extrinsics = TransformCuda::Identity();
    for (int i = 0; i < 10; ++i) {
        volume.Integrate(rgbd, intrinsics, extrinsics);
    }

    ImageCuda<Vector3f> raycaster;
    raycaster.Create(depth.cols, depth.rows);

    Timer timer;
    int iteration_times = 100;
    timer.Start();
    for (int i = 0; i < iteration_times; ++i) {
        volume.RayCasting(raycaster, intrinsics, extrinsics);
    }
    timer.Stop();
    PrintInfo("Average raycasting time: %f milliseconds\n",
              timer.GetDuration() / iteration_times);

    cv::Mat imraycaster = raycaster.Download();
    cv::imshow("im", imraycaster);
    cv::waitKey(-1);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}