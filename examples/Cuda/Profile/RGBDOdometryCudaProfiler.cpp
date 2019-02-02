//
// Created by wei on 11/14/18.
//

#include <string>
#include <vector>
#include <Core/Core.h>
#include <IO/IO.h>

#include <Cuda/Odometry/RGBDOdometryCuda.h>
#include <Cuda/Geometry/PointCloudCuda.h>
#include <Visualization/Visualization.h>

#include "../Utils.h"

using namespace open3d;

int TwoFrameRGBDOdometry(
    const std::string &source_depth_path,
    const std::string &source_color_path,
    const std::string &target_depth_path,
    const std::string &target_color_path) {
    using namespace open3d;

    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    /** Load data **/
    Image source_color, source_depth, target_color, target_depth;
    ReadImage(source_color_path, source_color);
    ReadImage(source_depth_path, source_depth);
    ReadImage(target_color_path, target_color);
    ReadImage(target_depth_path, target_depth);

    cuda::RGBDImageCuda source, target;
    source.Upload(source_depth, source_color);
    target.Upload(target_depth, target_color);

    /** Prepare odometry class **/
    cuda::RGBDOdometryCuda<3> odometry;
    odometry.SetIntrinsics(PinholeCameraIntrinsic(
        PinholeCameraIntrinsicParameters::PrimeSenseDefault));
    odometry.SetParameters(OdometryOption({0, 0, 60}, 0.07), 0.5f);
    odometry.Initialize(source, target);

    Timer timer;
    timer.Start();
    const int cases = 100;
    for (int i = 0; i < cases; ++i) {
        odometry.transform_source_to_target_ = Eigen::Matrix4d::Identity();
        odometry.ComputeMultiScale();
    }
    timer.Stop();
    PrintInfo("time: %f\n", timer.GetDuration() / cases);

    return 0;
}

int main(int argc, char **argv) {
    std::string base_path = "/home/wei/Work/data/stanford/lounge";
    auto rgbd_filenames = ReadDataAssociation(
        base_path + "/data_association.txt");

    int i = 0;
    TwoFrameRGBDOdometry(
        base_path + "/" + rgbd_filenames[i + 5].first,
        base_path + "/" + rgbd_filenames[i + 5].second,
        base_path + "/" + rgbd_filenames[i].first,
        base_path + "/" + rgbd_filenames[i].second);

    return 0;
}