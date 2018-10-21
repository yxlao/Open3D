//
// Created by wei on 10/20/18.
//

#include <Cuda/Integration/ScalableTSDFVolumeCuda.h>
#include <Core/Core.h>
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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}