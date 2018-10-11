//
// Created by wei on 10/10/18.
//

#include <Cuda/Integration/UniformTSDFVolumeCuda.h>
#include <Cuda/Geometry/VectorCuda.h>
#include <Core/Core.h>

#include "UnitTest.h"

#include <opencv2/opencv.hpp>
#include <vector>

TEST(UniformTSDFVolumeCuda, UploadAndDownload) {
    //const size_t N = 512;
    const size_t N = 16;
    const size_t NNN = N * N * N;
    std::vector<float> tsdf, weight;
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
                weight[cnt] = i;
                color[cnt] = open3d::Vector3b(i, j, k);
                ++cnt;
            }
        }
    }

    volume.Upload(tsdf, weight, color);
    tsdf.clear();
    weight.clear();
    color.clear();

    auto downloaded = volume.Download();
    tsdf = std::get<0>(downloaded);
    weight = std::get<1>(downloaded);
    color = std::get<2>(downloaded);

    cnt = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                EXPECT_NEAR(tsdf[cnt], i * j * k, NNN * 1e-6);
                EXPECT_NEAR(weight[cnt], i, N * 1e-6);
                EXPECT_EQ(color[cnt], open3d::Vector3b(i, j, k));
                ++cnt;
            }
        }
    }
}

TEST(UniformTSDFVolumeCuda, Integrate) {
    cv::Mat im = cv::imread("../../examples/TestData/RGBD/depth/00000.png",
                            cv::IMREAD_UNCHANGED);
    open3d::ImageCuda<open3d::Vector1s> imcuda;
    imcuda.Upload(im);
    auto imcudaf = imcuda.ToFloat(0.001f);

    open3d::MonoPinholeCameraCuda default_camera;
    default_camera.SetUp();
    open3d::TransformCuda transform = open3d::TransformCuda::Identity();
    transform.SetTranslation(open3d::Vector3f(-0.04f * 256, -0.04f * 256, -0.04f * 256));

    open3d::UniformTSDFVolumeCuda<512> volume(0.04f, 0.12f, transform);

    open3d::TransformCuda extrinsics = open3d::TransformCuda::Identity();
    volume.Integrate(imcudaf, default_camera, extrinsics);

    open3d::ImageCuda<open3d::Vector3f> raycaster;
    raycaster.Create(imcuda.width(), imcuda.height());

    volume.RayCasting(raycaster, default_camera, extrinsics);

    cv::Mat imraycaster = raycaster.Download();
    cv::imshow("im", imraycaster);
    cv::waitKey(-1);

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}