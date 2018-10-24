//
// Created by wei on 10/20/18.
//

#include <Cuda/Integration/ScalableTSDFVolumeCuda.h>
#include <Cuda/Geometry/ImageCuda.h>
#include <Core/Core.h>
#include <opencv2/opencv.hpp>
#include "UnitTest.h"

TEST(ScalableTSDFVolumeCuda, Create) {
    using namespace open3d;

    ScalableTSDFVolumeCuda<8> volume;
    volume.Create(10000, 200000);

    ScalableTSDFVolumeCuda<8> volume_copy;
    volume_copy = volume;

    auto downloaded_volume = volume_copy.DownloadVolumes();
    auto& keys = downloaded_volume.first;
    auto& volumes = downloaded_volume.second;
    for (auto &key : keys) {
        PrintInfo("(%d %d %d)\n", keys[0], keys[1], keys[2]);
    }
}

TEST(ScalableTSDFVolumeCuda, TouchSubvolumes) {
    using namespace open3d;
    cv::Mat im = cv::imread("../../examples/TestData/RGBD/depth/00000.png",
                            cv::IMREAD_UNCHANGED);
    ImageCuda<Vector1s> imcuda;
    imcuda.Upload(im);
    auto imcudaf = imcuda.ToFloat(0.001f);

    MonoPinholeCameraCuda default_camera;
    default_camera.SetUp();

    float voxel_length = 0.01f;
    TransformCuda extrinsics = TransformCuda::Identity();
    ScalableTSDFVolumeCuda<8> volume(10000,
                                     200000,
                                     voxel_length,
                                     3 * voxel_length,
                                     extrinsics);

    Timer timer;
    timer.Start();
    volume.TouchBlocks(imcudaf, default_camera, extrinsics);
    timer.Stop();
    PrintInfo("ToucBlocks takes %f milliseconds.\n", timer.GetDuration());

    auto entry_vector = volume.target_subvolume_entry_array().Download();
    for (auto &entry : entry_vector) {
        PrintInfo("%d %d %d %d\n", entry.key(0), entry.key(1), entry.key(2),
            entry.internal_addr);
    }
}

TEST(ScalableTSDFVolumeCuda, Integration) {
    using namespace open3d;
    cv::Mat im = cv::imread("../../examples/TestData/RGBD/depth/00000.png",
                            cv::IMREAD_UNCHANGED);
    ImageCuda<Vector1s> imcuda;
    imcuda.Upload(im);
    auto imcudaf = imcuda.ToFloat(0.001f);

    MonoPinholeCameraCuda default_camera;
    default_camera.SetUp();

    float voxel_length = 0.01f;
    TransformCuda extrinsics = TransformCuda::Identity();
    ScalableTSDFVolumeCuda<8> volume(10000,
                                     200000,
                                     voxel_length,
                                     3 * voxel_length,
                                     extrinsics);

    Timer timer;
    timer.Start();
    volume.TouchBlocks(imcudaf, default_camera, extrinsics);
    timer.Stop();
    PrintInfo("TouchBlocks takes %f milliseconds.\n", timer.GetDuration());
    auto entry_vector = volume.target_subvolume_entry_array().Download();

    timer.Start();
    volume.Integrate(imcudaf, default_camera, extrinsics);
    timer.Stop();
    PrintInfo("Integrate takes %f milliseconds.\n", timer.GetDuration());

    PrintInfo("Downloading volumes: \n");
    auto result = volume.DownloadVolumes();
    auto &keys = std::get<0>(result);
    auto &volumes = std::get<1>(result);
    for (int i = 0; i < keys.size(); ++i) {
        float sum_tsdf = 0;
        auto &volume = volumes[i];
        auto &tsdf = std::get<0>(volume);
        for (int k = 0; k < 512; ++k) {
            sum_tsdf += fabsf(tsdf[k]);
        }
        PrintInfo("%d %d %d %f\n", keys[i](0), keys[i](1), keys[i](2), sum_tsdf);
    }
}

TEST(ScalableTSDFVolumeCuda, RayCasting) {
    using namespace open3d;
    cv::Mat im = cv::imread("../../examples/TestData/RGBD/depth/00000.png",
                            cv::IMREAD_UNCHANGED);
    ImageCuda<Vector1s> imcuda;
    imcuda.Upload(im);
    auto imcudaf = imcuda.ToFloat(0.001f);

    MonoPinholeCameraCuda default_camera;
    default_camera.SetUp();

    float voxel_length = 0.01f;
    TransformCuda extrinsics = TransformCuda::Identity();
    ScalableTSDFVolumeCuda<8> volume(10000,
                                     200000,
                                     voxel_length,
                                     3 * voxel_length,
                                     extrinsics);

    Timer timer;
    timer.Start();
    volume.TouchBlocks(imcudaf, default_camera, extrinsics);
    timer.Stop();
    PrintInfo("TouchBlocks takes %f milliseconds.\n", timer.GetDuration());
    auto entry_vector = volume.target_subvolume_entry_array().Download();

    timer.Start();
    volume.Integrate(imcudaf, default_camera, extrinsics);
    timer.Stop();
    PrintInfo("Integrate takes %f milliseconds.\n", timer.GetDuration());

    ImageCuda<Vector3f> raycaster;
    raycaster.Create(imcuda.width(), imcuda.height());
    volume.RayCasting(raycaster, default_camera, extrinsics);
    cv::imshow("Raycaster", raycaster.Download());
    cv::waitKey(-1);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}