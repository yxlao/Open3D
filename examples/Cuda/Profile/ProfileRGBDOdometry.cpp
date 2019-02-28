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

#include "../ReconstructionSystem/DatasetConfig.h"
#include "Analyzer.h"

double ProfileOdometry(
    const std::string &source_depth_path,
    const std::string &source_color_path,
    const std::string &target_depth_path,
    const std::string &target_color_path,
    DatasetConfig &config,
    bool use_cuda) {
    using namespace open3d;

    /** Load data **/
    Image source_color, source_depth, target_color, target_depth;
    ReadImage(source_color_path, source_color);
    ReadImage(source_depth_path, source_depth);
    ReadImage(target_color_path, target_color);
    ReadImage(target_depth_path, target_depth);

    OdometryOption option({20, 10, 5},
                          config.max_depth_diff_,
                          config.min_depth_,
                          config.max_depth_);

    if (use_cuda) {
        Timer timer;
        timer.Start();

        /** Prepare odometry class **/
        cuda::RGBDOdometryCuda<3> odometry;
        odometry.SetIntrinsics(config.intrinsic_);
        odometry.SetParameters(option, 0.5f);

        cuda::RGBDImageCuda rgbd_source((float) config.max_depth_,
                                        (float) config.depth_factor_);
        cuda::RGBDImageCuda rgbd_target((float) config.max_depth_,
                                        (float) config.depth_factor_);
        rgbd_source.Upload(source_depth, source_color);
        rgbd_target.Upload(target_depth, target_color);

        odometry.Initialize(rgbd_source, rgbd_target);

        odometry.transform_source_to_target_ = Eigen::Matrix4d::Identity();
        odometry.ComputeMultiScale();
        auto information = odometry.ComputeInformationMatrix();
        timer.Stop();

        return timer.GetDuration();
    } else {
        Timer timer;
        timer.Start();
        RGBDImage source_cpu, target_cpu;
        double depth_scale = config.depth_factor_,
            depth_trunc = config.max_depth_;
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
        Eigen::Matrix6d information;
        std::tie(is_success, transformation, information) =
            ComputeRGBDOdometry(*rgbd_source, *rgbd_target,
                                config.intrinsic_,
                                Eigen::Matrix4d::Identity(),
                                RGBDOdometryJacobianFromHybridTerm(),
                                option);
        timer.Stop();

        return timer.GetDuration();
    }

    return -1;
}

int main(int argc, char **argv) {
    DatasetConfig config;

    std::string config_path = argc > 1 ? argv[1] :
                              "/home/wei/Work/projects/dense_mapping/Open3D/examples/Cuda"
                              "/ReconstructionSystem/config/copyroom.json";

    bool is_success = ReadIJsonConvertible(config_path, config);
    if (!is_success) return 1;

    int num_odometries = config.color_files_.size() - 1;
    std::vector<double> times;
    times.resize(num_odometries);
    double mean, std;
    for (int i = 0; i < num_odometries; ++i) {
        double time = ProfileOdometry(
            config.depth_files_[i], config.color_files_[i],
            config.depth_files_[i + 1], config.color_files_[i + 1],
            config,
            true);
        times[i] = time;
        PrintInfo("Frame %d / %d takes %f ms\n", i, num_odometries, time);
    }
    std::tie(mean, std) = ComputeStatistics(times);
        PrintInfo("gpu time: avg = %f, std = %f\n", mean, std);

//    for (int i = 0; i < num_odometries; ++i) {
//        double time = ProfileOdometry(
//            config.depth_files_[i], config.color_files_[i],
//            config.depth_files_[i + 1], config.color_files_[i + 1],
//            config,
//            false);
//        times[i] = time;
//        PrintInfo("Frame %d / %d takes %f ms\n", i, num_odometries, time);
//    }
//    std::tie(mean, std) = ComputeStatistics(times);
//    PrintInfo("cpu time: avg = %f, std = %f\n", mean, std);

    return 0;
}