//
// Created by wei on 11/14/18.
//

#include <string>
#include <vector>
#include <Open3D/Open3D.h>
#include <Cuda/Open3DCuda.h>

#include "Utils.h"

using namespace open3d;

using namespace open3d::utility;
using namespace open3d::io;
using namespace open3d::geometry;
using namespace open3d::camera;
using namespace open3d::visualization;

int RGBDOdometry(
    const std::string &source_color_path,
    const std::string &source_depth_path,
    const std::string &target_color_path,
    const std::string &target_depth_path) {

    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    /** Load data **/
    Image source_color, source_depth, target_color, target_depth;
    ReadImage(source_color_path, source_color);
    ReadImage(source_depth_path, source_depth);
    ReadImage(target_color_path, target_color);
    ReadImage(target_depth_path, target_depth);

    float depth_trunc = 3.0;
    float depth_scale = 1000.0;
    cuda::RGBDImageCuda
        source(source_color.width_, source_color.height_, depth_trunc, depth_scale),
        target(target_color.width_, target_color.height_, depth_trunc, depth_scale);
    source.Upload(source_depth, source_color);
    target.Upload(target_depth, target_color);

    /** Prepare odometry class **/
    cuda::RGBDOdometryCuda<3> odometry;
    odometry.SetIntrinsics(PinholeCameraIntrinsic(
        PinholeCameraIntrinsicParameters::PrimeSenseDefault));
    odometry.SetParameters(odometry::OdometryOption({20, 10, 5}, 0.07), 0.5);
    odometry.Initialize(source, target);
    odometry.transform_source_to_target_ = Eigen::Matrix4d::Identity();

    /** Prepare visualizer **/
    VisualizerWithCudaModule visualizer;
    if (!visualizer.CreateVisualizerWindow("RGBDOdometry", 640, 480, 0, 0)) {
        utility::PrintWarning("Failed creating OpenGL window.\n");
        return -1;
    }
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();

    /** Prepare point cloud (original) **/
    std::shared_ptr<cuda::PointCloudCuda>
        pcl_source = std::make_shared<cuda::PointCloudCuda>(
        cuda::VertexWithColor, 640 * 480),
        pcl_target = std::make_shared<cuda::PointCloudCuda>(
        cuda::VertexWithColor, 640 * 480);

    int level = 2;
    pcl_source->Build(odometry.source_depth_[level],
                      odometry.source_intensity_[level],
                      odometry.device_->intrinsics_[level]);
    pcl_source->Build(odometry.target_depth_[level],
                      odometry.target_intensity_[level],
                      odometry.device_->intrinsics_[level]);
    visualizer.AddGeometry(pcl_source);
    visualizer.AddGeometry(pcl_target);

    /** Correspondence visualizer **/
    std::shared_ptr<LineSet> lines = std::make_shared<LineSet>();
    visualizer.AddGeometry(lines);

    const int kIterations[3] = {60, 60, 60};
    bool finished = false;
    int iter = kIterations[level];

    bool is_success;
    Eigen::Matrix4d delta;
    float loss;
    visualizer.RegisterKeyCallback(GLFW_KEY_SPACE, [&](Visualizer *vis) {
        if (finished) return false;

        /* Odometry (1 iteration) */
        std::tie(is_success, delta, loss)
            = odometry.DoSingleIteration(level, iter);
        odometry.transform_source_to_target_ = delta *
            odometry.transform_source_to_target_;

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

        /* Update point cloud */
        pcl_source->Transform(delta);

        /* Re-bind geometry */
        vis->UpdateGeometry();

        /* Update masks */
        --iter;
        if (iter == 0) {
            --level;
            if (level < 0) {
                finished = true;
            } else {
                pcl_source->Build(odometry.source_depth_[level],
                                  odometry.source_intensity_[level],
                                  odometry.device_->intrinsics_[level]);
                pcl_source->Build(odometry.target_depth_[level],
                                  odometry.target_intensity_[level],
                                  odometry.device_->intrinsics_[level]);
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
    std::string source_color_path, source_depth_path,
        target_color_path, target_depth_path;
    if (argc > 4) {
        source_color_path = argv[1];
        source_depth_path = argv[2];
        target_color_path = argv[3];
        target_depth_path = argv[4];
    } else {
        std::string test_data_path = "../../../examples/TestData/RGBD";
        source_color_path = test_data_path + "/color/00000.jpg";
        source_depth_path = test_data_path + "/depth/00000.png";
        target_color_path = test_data_path + "/color/00002.jpg";
        target_depth_path = test_data_path + "/depth/00002.png";
    }

    return RGBDOdometry(source_color_path, source_depth_path,
                        target_color_path, target_depth_path);
}