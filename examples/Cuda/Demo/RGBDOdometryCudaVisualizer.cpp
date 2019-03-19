//
// Created by wei on 11/14/18.
//

#include <string>
#include <vector>
#include <Open3D/Open3D.h>

#include <Cuda/Odometry/RGBDOdometryCuda.h>
#include <Cuda/Geometry/PointCloudCuda.h>
#include "examples/Cuda/DatasetConfig.h"
#include "examples/Cuda/Utils.h"

using namespace open3d;

int TwoFrameRGBDOdometry(
    const std::string &source_depth_path,
    const std::string &source_color_path,
    const std::string &target_depth_path,
    const std::string &target_color_path) {
    using namespace open3d;

    SetVerbosityLevel(utility::VerbosityLevel::VerboseDebug);

    /** Load data **/
    geometry::Image source_color, source_depth, target_color, target_depth;
    io::ReadImage(source_color_path, source_color);
    io::ReadImage(source_depth_path, source_depth);
    io::ReadImage(target_color_path, target_color);
    io::ReadImage(target_depth_path, target_depth);

    cuda::RGBDImageCuda
        source(640, 480, 2.5, 5000), target(640, 480, 2.5, 5000);
    source.Upload(source_depth, source_color);
    target.Upload(target_depth, target_color);

    /** Prepare odometry class **/
    cuda::RGBDOdometryCuda<3> odometry;
    odometry.SetIntrinsics(camera::PinholeCameraIntrinsic(
        camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault));
    odometry.SetParameters(odometry::OdometryOption({20, 10, 5}, 0.07), 0.5);
    odometry.Initialize(source, target);
    odometry.transform_source_to_target_ = Eigen::Matrix4d::Identity();

    /** Prepare visualizer **/
    visualization::VisualizerWithKeyCallback visualizer;
    if (!visualizer.CreateVisualizerWindow("RGBDOdometry", 640, 480, 0, 0)) {
        utility::PrintWarning("Failed creating OpenGL window.\n");
        return -1;
    }
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();

    /** Prepare point cloud (original) **/
    std::shared_ptr<cuda::PointCloudCuda>
        pcl_source = std::make_shared<cuda::PointCloudCuda>(
        cuda::VertexWithColor, 300000),
        pcl_target = std::make_shared<cuda::PointCloudCuda>(
        cuda::VertexWithColor, 300000);
    pcl_source->Build(source, odometry.device_->intrinsics_[0]);
    pcl_target->Build(target, odometry.device_->intrinsics_[0]);
    visualizer.AddGeometry(pcl_source);
    visualizer.AddGeometry(pcl_target);

    /** Correspondence visualizer **/
    std::shared_ptr<geometry::LineSet> lines = std::make_shared<geometry::LineSet>();
    visualizer.AddGeometry(lines);

    const int kIterations[3] = {40, 20, 10};
    bool finished = false;
    int level = 2;
    int iter = kIterations[level];
    Eigen::Matrix4d prev_transform = Eigen::Matrix4d::Identity();

    bool is_success;
    Eigen::Matrix4d delta;
    float loss;
    visualizer.RegisterKeyCallback(GLFW_KEY_SPACE, [&](visualization::Visualizer *vis) {
        if (finished) return false;

        /* Odometry (1 iteration) */
        std::tie(is_success, delta, loss) =
            odometry.DoSingleIteration(level, iter);

        /* Update pose */
        odometry.transform_source_to_target_ = delta *
            odometry.transform_source_to_target_;
        prev_transform = odometry.transform_source_to_target_;

        /* Update point cloud */
        pcl_source->Build(odometry.source_depth_[level],
                          odometry.device_->intrinsics_[level]);
        pcl_target->Build(odometry.target_depth_[level],
                          odometry.device_->intrinsics_[level]);
        pcl_source->Transform(odometry.transform_source_to_target_);

        /* Update correspondences */
        lines->points_.clear();
        lines->lines_.clear();
        lines->colors_.clear();
        auto &intrinsic = odometry.device_->intrinsics_[level];
        auto correspondences = odometry.correspondences_.Download();
        auto src_depth = odometry.source_depth_[level].DownloadImage();
        auto tgt_depth = odometry.target_depth_[level].DownloadImage();
        for (int i = 0; i < correspondences.size(); ++i) {
            auto &c = correspondences[i];

            auto p_src = cuda::Vector2i(c(0), c(1));
            auto X_src = intrinsic.InverseProjectPixel(
                p_src, *geometry::PointerAt<float>(*src_depth, c(0), c(1)));
            Eigen::Vector4d X_src_h = odometry.transform_source_to_target_ *
                Eigen::Vector4d(X_src(0), X_src(1), X_src(2), 1.0);
            lines->points_.emplace_back(X_src_h.hnormalized());

            auto p_tgt = cuda::Vector2i(c(2), c(3));
            auto X_tgt = intrinsic.InverseProjectPixel(
                p_tgt, *geometry::PointerAt<float>(*tgt_depth, c(2), c(3)));
            lines->points_.emplace_back(X_tgt(0), X_tgt(1), X_tgt(2));

            lines->colors_.emplace_back(0, 0, 1);
            lines->lines_.emplace_back(Eigen::Vector2i(2 * i + 1, 2 * i));
        }

        /* Re-bind geometry */
        vis->UpdateGeometry();

        /* Update masks */
        --iter;
        if (iter == 0) {
            --level;
            if (level < 0) {
                finished = true;
            } else {
                iter = kIterations[level];
            }
        }
        return !finished;
    });

    bool should_close = false;
    while (!should_close) {
        should_close = !visualizer.PollEvents();
    }
    visualizer.DestroyVisualizerWindow();

    return 0;
}

int main(int argc, char **argv) {
    DatasetConfig config;

    std::string config_path = argc > 1 ? argv[1]
                                       : kDefaultDatasetConfigDir
                                  + "/tum/fr3_household.json";
    bool is_success = io::ReadIJsonConvertible(config_path, config);
    if (!is_success) return 1;

    // 1800 -> 1805
    // 1900 -> 1905
    for (int i = 1900; i < config.depth_files_.size(); i += 100) {
        utility::PrintInfo("%d -> %d\n", i, i + 5);
        TwoFrameRGBDOdometry(config.depth_files_[i],
                             config.color_files_[i],
                             config.depth_files_[i + 5],
                             config.color_files_[i + 5]);
    }

    return 0;
}