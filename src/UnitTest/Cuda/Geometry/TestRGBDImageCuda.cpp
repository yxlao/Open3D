//
// Created by wei on 11/5/18.
//

#include "UnitTest.h"
#include <Core/Core.h>
#include <Cuda/Geometry/RGBDImageCuda.h>
#include <opencv2/opencv.hpp>

TEST(RGBDImageCuda, Reuse) {
    using namespace open3d;
    const std::string kDepthPath = "../../examples/TestData/RGBD/other_formats/TUM_depth.png";
    const std::string kColorPath = "../../examples/TestData/RGBD/other_formats/TUM_color.png";

    RGBDImageCuda rgbd_image;

    Timer timer;
    int iters = 1000;
    float time_reading = 0;
    float time_uploading = 0;
    for (int i = 0; i < iters; ++i) {
        timer.Start();
        cv::Mat depth = cv::imread(kDepthPath, cv::IMREAD_UNCHANGED);
        cv::Mat color = cv::imread(kColorPath, cv::IMREAD_UNCHANGED);
        timer.Stop();
        time_reading += timer.GetDuration();

        timer.Start();
        rgbd_image.Upload(depth, color);
        timer.Stop();
        time_uploading += timer.GetDuration();
    }
    PrintInfo("Average reading time: %.4f ms\n", time_reading / iters);
    PrintInfo("Average uploading time: %.4f ms\n", time_uploading / iters);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}