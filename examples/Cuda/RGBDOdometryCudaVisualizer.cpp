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

#include "Utils.h"

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
    odometry.SetParameters(OdometryOption({20, 10, 5}, 0.07), 0.5f);
    odometry.Initialize(source, target);
    odometry.transform_source_to_target_ = Eigen::Matrix4d::Identity();

    /** Prepare visualizer **/
    VisualizerWithKeyCallback visualizer;
    if (!visualizer.CreateVisualizerWindow("RGBDOdometry", 1280, 960, 0, 0)) {
        PrintWarning("Failed creating OpenGL window.\n");
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
    pcl_source->Build(odometry.source()[0],
                      odometry.server()->intrinsics_[0]);
    pcl_target->Build(odometry.target()[0],
                      odometry.server()->intrinsics_[0]);
    visualizer.AddGeometry(pcl_source);
    visualizer.AddGeometry(pcl_target);

    /** Correspondence visualizer **/
    std::shared_ptr<LineSet> lines = std::make_shared<LineSet>();
    visualizer.AddGeometry(lines);

    const int kIterations[3] = {60, 60, 60};
    bool finished = false;
    int level = 2;
    int iter = kIterations[level];
    Eigen::Matrix4d prev_transform = Eigen::Matrix4d::Identity();

    bool is_success;
    Eigen::Matrix4d delta;
    float loss;
    visualizer.RegisterKeyCallback(GLFW_KEY_SPACE, [&](Visualizer *vis) {
        if (finished) return false;

        /* Odometry (1 iteration) */
        std::tie(is_success, delta, loss) =
            odometry.DoSingleIteration(level, iter);

        /* Update pose */
        odometry.transform_source_to_target_ = delta *
            odometry.transform_source_to_target_;
        prev_transform = odometry.transform_source_to_target_;

        /* Update point cloud */
        pcl_source->Build(odometry.source()[level],
                          odometry.server()->intrinsics_[level]);
        pcl_target->Build(odometry.target()[level],
                          odometry.server()->intrinsics_[level]);
        pcl_source->Transform(odometry.transform_source_to_target_);

        /* Update correspondences */
        lines->points_.clear();
        lines->lines_.clear();
        lines->colors_.clear();
        auto &intrinsic = odometry.server()->intrinsics_[level];
        auto correspondences = odometry.correspondences_.Download();
        auto src_depth = odometry.source()[level].depthf().DownloadImage();
        auto tgt_depth = odometry.target()[level].depthf().DownloadImage();
        for (int i = 0; i < correspondences.size(); ++i) {
            auto &c = correspondences[i];

            auto p_src = cuda::Vector2i(c(0), c(1));
            auto X_src = intrinsic.InverseProjectPixel(
                p_src, *PointerAt<float>(*src_depth, c(0), c(1)));
            Eigen::Vector4d X_src_h = odometry.transform_source_to_target_ *
                Eigen::Vector4d(X_src(0), X_src(1), X_src(2), 1.0);
            lines->points_.emplace_back(X_src_h.hnormalized());

            auto p_tgt = cuda::Vector2i(c(2), c(3));
            auto X_tgt = intrinsic.InverseProjectPixel(
                p_tgt, *PointerAt<float>(*tgt_depth, c(2), c(3)));
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
    std::string base_path = "/home/wei/Work/data/stanford/lounge";
    auto rgbd_filenames = ReadDataAssociation(
        base_path + "/data_association.txt");

    for (int i = 0; i < rgbd_filenames.size(); i += 100) {
        std::cout << i << std::endl;
        TwoFrameRGBDOdometry(
            base_path + "/" + rgbd_filenames[i + 5].first,
            base_path + "/" + rgbd_filenames[i + 5].second,
            base_path + "/" + rgbd_filenames[i].first,
            base_path + "/" + rgbd_filenames[i].second);
    }

    return 0;
}