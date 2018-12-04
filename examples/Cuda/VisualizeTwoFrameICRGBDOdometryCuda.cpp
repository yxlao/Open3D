//
// Created by wei on 11/14/18.
//

//
// Created by wei on 10/6/18.
//

#include <string>
#include <vector>
#include <Core/Core.h>
#include <IO/IO.h>
#include <Cuda/Odometry/ICRGBDOdometryCuda.h>
#include <Cuda/Odometry/RGBDOdometryCuda.h>

#include <Cuda/Integration/ScalableTSDFVolumeCuda.h>
#include <Cuda/Integration/ScalableMeshVolumeCuda.h>
#include <Cuda/Geometry/PointCloudCuda.h>
#include <Visualization/Visualization.h>

#include <opencv2/opencv.hpp>
#include <thread>

#include "ReadDataAssociation.h"

using namespace open3d;

void PrintHelp() {
    PrintOpen3DVersion();
    PrintInfo("Usage :\n");
    PrintInfo("    > VisualizeTwoFrameICRGBDOdometryCuda [dataset_path]\n");
}

int TwoFrameRGBDOdometry(
    std::string source_depth_path,
    std::string source_color_path,
    std::string target_depth_path,
    std::string target_color_path) {
    using namespace open3d;

    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    /** Load data **/
    Image source_color, source_depth, target_color, target_depth;
    ReadImage(source_color_path, source_color);
    ReadImage(source_depth_path, source_depth);
    ReadImage(target_color_path, target_color);
    ReadImage(target_depth_path, target_depth);

    cuda::RGBDImageCuda source(0.1f, 4.0f, 5000.0f), target(0.1f, 4.0f, 5000.0f);
    source.Upload(source_depth, source_color);
    target.Upload(target_depth, target_color);

    /** Prepare odometry **/
    cuda::ICRGBDOdometryCuda<3> odometry;
//    odometry.SetIntrinsics(PinholeCameraIntrinsic(
//        PinholeCameraIntrinsicParameters::PrimeSenseDefault));
    odometry.SetIntrinsics(PinholeCameraIntrinsic(
        640, 480, 535.4, 539.2, 320.1, 247.6));
    odometry.SetParameters(OdometryOption({60, 60, 60}, 0.07, 0.01), 0.5f);
    odometry.Initialize(source, target);
    odometry.transform_source_to_target_ = Eigen::Matrix4d::Identity();

    /** Prepare point cloud **/
    std::shared_ptr<cuda::PointCloudCuda>
        pcl_source = std::make_shared<cuda::PointCloudCuda>(
            cuda::VertexWithColor, 300000),
        pcl_target = std::make_shared<cuda::PointCloudCuda>(
            cuda::VertexWithColor, 300000);

    /** Prepare visualizer **/
    VisualizerWithKeyCallback visualizer;
    if (!visualizer.CreateVisualizerWindow("IC RGBD Odometry", 1280, 960, 0,0)) {
        PrintWarning("Failed creating OpenGL window.\n");
        return -1;
    }
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();
    pcl_source->Build(odometry.source()[0],
                      odometry.server()->intrinsics_[0]);
//    pcl_source->colors().Fill(Vector3f(0, 1, 0));
    pcl_target->Build(odometry.target()[0],
                      odometry.server()->intrinsics_[0]);
//    pcl_target->colors().Fill(Vector3f(1, 0, 0));

    visualizer.AddGeometry(pcl_source);
    visualizer.AddGeometry(pcl_target);

    std::shared_ptr<LineSet> lines = std::make_shared<LineSet>();
    visualizer.AddGeometry(lines);

    std::vector<float> losses[3];
    const int kIterations[3] = {20, 20, 20};
    bool finished = false;
    int level = 2;
    int iter = kIterations[level];
    Eigen::Matrix4d prev_transform = Eigen::Matrix4d::Identity();


    bool is_success;
    Eigen::Matrix4d delta;
    float loss;
    visualizer.RegisterKeyCallback(GLFW_KEY_SPACE, [&](Visualizer *vis) {
        if (!finished) {
            Timer timer;
            timer.Start();
            std::tie(is_success, delta, loss) =
                odometry.DoSingleIteration(level, iter);
            odometry.transform_source_to_target_ = delta.inverse() *
                odometry.transform_source_to_target_;
            timer.Stop();
            //PrintInfo("Per iteration: %.4f ms\n", timer.GetDuration());

            lines->points_.clear();
            lines->lines_.clear();
            lines->colors_.clear();
            std::vector<cuda::Vector4i>
                correspondences = odometry.correspondences_.Download();

            cuda::PinholeCameraIntrinsicCuda intrinsic = odometry.server()
                ->intrinsics_[level];

            pcl_source->Build(odometry.source()[level],
                              odometry.server()->intrinsics_[level]);
//            pcl_source->colors().Fill(Vector3f(0, 1, 0));
            pcl_target->Build(odometry.target()[level],
                              odometry.server()->intrinsics_[level]);
//            pcl_target->colors().Fill(Vector3f(1, 0, 0));

            std::shared_ptr<Image> src_depth = odometry.source()[level].depthf()
                .DownloadImage();
            std::shared_ptr<Image> tgt_depth = odometry.target()[level].depthf()
                .DownloadImage();

            for (int i = 0; i < correspondences.size(); ++i) {
                auto &c = correspondences[i];
                cuda::Vector2i p_src = cuda::Vector2i(c(0), c(1));
                cuda::Vector3f X_src = intrinsic.InverseProjectPixel(
                    p_src, *PointerAt<float>(*src_depth, c(0), c(1)));
                Eigen::Vector4d X_src_h = odometry.transform_source_to_target_ *
                    Eigen::Vector4d(X_src(0), X_src(1), X_src(2), 1.0);
                lines->points_.emplace_back(X_src_h.hnormalized());

                cuda::Vector2i p_tgt = cuda::Vector2i(c(2), c(3));
                cuda::Vector3f X_tgt = intrinsic.InverseProjectPixel(
                    p_tgt, *PointerAt<float>(*tgt_depth, c(2), c(3)));
                lines->points_.emplace_back(X_tgt(0), X_tgt(1), X_tgt(2));

                lines->colors_.emplace_back(0, 1, 0);
                lines->lines_.emplace_back(Eigen::Vector2i(2 * i + 1, 2 * i));
            }

            pcl_source->Transform(odometry.transform_source_to_target_);
            prev_transform = odometry.transform_source_to_target_;
            vis->UpdateGeometry();
        }

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
    if (argc != 2 || ProgramOptionExists(argc, argv, "--help")) {
        PrintHelp();
        return 1;
    }

    std::string base_path = argv[1];
    auto rgbd_filenames = ReadDataAssociation(
        base_path + "/data_association.txt");

    for (int i = 37; i < 38; i += 1) {
        std::cout << rgbd_filenames[i].first << std::endl;
        TwoFrameRGBDOdometry(
            base_path + "/" + rgbd_filenames[i + 1].first,
            base_path + "/" + rgbd_filenames[i + 1].second,
            base_path + "/" + rgbd_filenames[i].first,
            base_path + "/" + rgbd_filenames[i].second);
    }

    return 0;
}