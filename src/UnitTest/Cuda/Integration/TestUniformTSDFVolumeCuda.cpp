//
// Created by wei on 10/10/18.
//

#include <Cuda/Integration/UniformTSDFVolumeCuda.h>
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
                EXPECT_NEAR(weight[cnt], i, N * 1e-6);
                EXPECT_EQ(color[cnt], open3d::Vector3b(i, j, k));
                ++cnt;
            }
        }
    }
}

TEST(UniformTSDFVolumeCuda, RayCasting) {
    using namespace open3d;
    cv::Mat im = cv::imread("../../examples/TestData/RGBD/depth/00000.png",
                            cv::IMREAD_UNCHANGED);
    ImageCuda<open3d::Vector1s> imcuda;
    imcuda.Upload(im);
    auto imcudaf = imcuda.ToFloat(0.001f);

    MonoPinholeCameraCuda default_camera;
    default_camera.SetUp();
    TransformCuda transform = TransformCuda::Identity();

    const float voxel_length = 0.01f;
    transform.SetTranslation(Vector3f(-voxel_length * 256));
    UniformTSDFVolumeCuda<512> volume(voxel_length, voxel_length * 3, transform);

    TransformCuda extrinsics = TransformCuda::Identity();
    volume.Integrate(imcudaf, default_camera, extrinsics);

    ImageCuda<Vector3f> raycaster;
    raycaster.Create(imcuda.width(), imcuda.height());

    Timer timer;
    int iteration_times = 100;
    timer.Start();
    for (int i = 0; i < iteration_times; ++i) {
        volume.RayCasting(raycaster, default_camera, extrinsics);
    }
    timer.Stop();
    cv::Mat imraycaster = raycaster.Download();
    cv::imshow("im", imraycaster);
    cv::waitKey(-1);

    PrintInfo("Average raycasting time: %f milliseconds\n",
        timer.GetDuration() / iteration_times);
}

TEST(UniformTSDFVolumeCuda, MarchingCubes) {
    using namespace open3d;
    cv::Mat im = cv::imread("../../examples/TestData/RGBD/depth/00000.png",
                            cv::IMREAD_UNCHANGED);
    ImageCuda<open3d::Vector1s> imcuda;
    imcuda.Upload(im);
    auto imcudaf = imcuda.ToFloat(0.001f);

    MonoPinholeCameraCuda default_camera;
    default_camera.SetUp();
    TransformCuda transform = TransformCuda::Identity();

    const float voxel_length = 0.01f;
    transform.SetTranslation(Vector3f(-voxel_length * 256));
    UniformTSDFVolumeCuda<512> volume(voxel_length, voxel_length * 3, transform);

    TransformCuda extrinsics = TransformCuda::Identity();
    volume.Integrate(imcudaf, default_camera, extrinsics);

    Timer timer;
    timer.Start();
    volume.MarchingCubes();
    timer.Stop();
    PrintInfo("MarchingCubes time: %f milliseconds\n", timer.GetDuration());

    std::shared_ptr<TriangleMesh> mesh = volume.mesh().Download();
    PrintInfo("triangle.size(): %d\n, vertices.size(): %d\n",
        mesh->triangles_.size(), mesh->vertices_.size());
//    for (auto &triangle : mesh->triangles_) {
//        std::cout << triangle(0) << " " << triangle(1) << " " << triangle(2)
//        << std::endl;
//    }
    WriteTriangleMeshToPLY("test.ply", *mesh, true);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}