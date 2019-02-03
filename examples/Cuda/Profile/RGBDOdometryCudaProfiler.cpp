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
    odometry.SetParameters(OdometryOption({20, 10, 5}, 0.03), 0.5f);
    odometry.Initialize(source, target);

    Timer timer;
    timer.Start();
    const int cases = 100;
    for (int i = 0; i < cases; ++i) {
        odometry.transform_source_to_target_ = Eigen::Matrix4d::Identity();
        odometry.ComputeMultiScale();
    }
    auto information = odometry.ComputeInformationMatrix();
    std::cout << information << std::endl;
    timer.Stop();
    PrintInfo("time: %f\n", timer.GetDuration() / cases);

    RGBDImage source_cpu, target_cpu;
    double depth_scale = 1000.0, depth_trunc = 3.0;
    bool convert_rgb_to_intensity = true;

    std::shared_ptr<RGBDImage> rgbd_source =
        CreateRGBDImageFromColorAndDepth(
            source_color, source_depth,
            depth_scale, depth_trunc,
            convert_rgb_to_intensity);
    std::shared_ptr<RGBDImage> rgbd_target =
        CreateRGBDImageFromColorAndDepth(
            target_color, target_depth,
            depth_scale, depth_trunc,
            convert_rgb_to_intensity);

    bool is_success;
    Eigen::Matrix4d transformation;
    std::tie(is_success, transformation, information) =
        ComputeRGBDOdometry(*rgbd_source, *rgbd_target,
        PinholeCameraIntrinsic(
            PinholeCameraIntrinsicParameters::PrimeSenseDefault));
    std::cout << information << std::endl;

    return 0;
}

int main(int argc, char **argv) {
    std::string base_path = "/home/wei/Work/data/stanford/lounge";
    auto rgbd_filenames = ReadDataAssociation(
        base_path + "/data_association.txt");

    int i = 0;
    TwoFrameRGBDOdometry(
        base_path + "/" + rgbd_filenames[i + 1].first,
        base_path + "/" + rgbd_filenames[i + 1].second,
        base_path + "/" + rgbd_filenames[i].first,
        base_path + "/" + rgbd_filenames[i].second);

    return 0;
}