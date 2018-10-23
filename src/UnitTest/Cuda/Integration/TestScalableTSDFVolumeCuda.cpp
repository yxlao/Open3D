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
            entry.value_ptr);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}