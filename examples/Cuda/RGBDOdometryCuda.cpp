//
// Created by wei on 10/6/18.
//

#include <Cuda/Odometry/RGBDOdometryCuda.h>
#include <string>
#include <vector>
#include <Core/Core.h>
#include <IO/IO.h>

#include <opencv2/opencv.hpp>

int main(int argc, char**argv) {
    using namespace open3d;

    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    std::string base_path = "../../../examples/TestData/RGBD/";

    Image source_color, source_depth, target_color, target_depth;
    ReadImage(base_path + "color/00000.jpg", source_color);
    ReadImage(base_path + "depth/00000.png", source_depth);

    ReadImage(base_path + "color/00001.jpg", target_color);
    ReadImage(base_path + "depth/00001.png", target_depth);

    RGBDImageCuda source, target;
    source.Upload(source_depth, source_color);
    target.Upload(target_depth, target_color);

    RGBDOdometryCuda<3> odometry;
    odometry.SetIntrinsics(PinholeCameraIntrinsic(
        PinholeCameraIntrinsicParameters::PrimeSenseDefault));

    odometry.SetParameters(0.2f, 0.1f, 4.0f, 0.07f);

    Timer timer;
    const int num_iters = 100;

    timer.Start();
    odometry.PrepareData(source, target);
    for (int i = 0; i < num_iters; ++i) {
        odometry.transform_source_to_target_ = Eigen::Matrix4d::Identity();
        odometry.Apply();
    }
    timer.Stop();
    PrintInfo("Average odometry time: %f milliseconds.\n",
              timer.GetDuration() / num_iters);
}