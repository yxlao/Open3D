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
    PrintInfo("    > VisualizeTwoFrameRGBDOdometryCuda [dataset_path]\n");
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

    RGBDImageCuda source, target;
    source.Upload(source_depth, source_color);
    target.Upload(target_depth, target_color);

    /** Prepare odometry **/
    RGBDOdometryCuda<3> odometry;
    odometry.SetIntrinsics(PinholeCameraIntrinsic(
        PinholeCameraIntrinsicParameters::PrimeSenseDefault));
    odometry.SetParameters(OdometryOption(), 1.0f);
    odometry.PrepareData(source, target);
    odometry.transform_source_to_target_ = Eigen::Matrix4d::Identity();

    /** Prepare point cloud **/
    std::shared_ptr<PointCloudCuda>
        pcl_source = std::make_shared<PointCloudCuda>(VertexWithColor, 300000),
        pcl_target = std::make_shared<PointCloudCuda>(VertexWithColor, 300000);

    /** Prepare visualizer **/
    VisualizerWithKeyCallback visualizer;
    if (!visualizer.CreateVisualizerWindow("ScalableFusion", 640, 480, 0, 0)) {
        PrintWarning("Failed creating OpenGL window.\n");
        return -1;
    }
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();
    pcl_source->Build(odometry.source()[0],
                      odometry.server()->intrinsics_[0]);
    pcl_source->colors().Fill(Vector3f(0, 1, 0));
    pcl_target->Build(odometry.target()[0],
                      odometry.server()->intrinsics_[0]);
    pcl_target->colors().Fill(Vector3f(1, 0, 0));

    visualizer.AddGeometry(pcl_source);
    visualizer.AddGeometry(pcl_target);

    std::shared_ptr<LineSet> lines = std::make_shared<LineSet>();
    visualizer.AddGeometry(lines);

    std::vector<float> losses[3];
    const int kIterations[3] = {60, 60, 60};
    bool finished = false;
    int level = 2;
    int iter = kIterations[level];
    Eigen::Matrix4d prev_transform = Eigen::Matrix4d::Identity();


    bool is_success;
    Eigen::Matrix4d delta;
    float loss;
    visualizer.RegisterKeyCallback(GLFW_KEY_SPACE, [&](Visualizer *vis) {
        if (!finished) {
            std::tie(is_success, delta, loss) =
                odometry.DoSingleIteration(level, iter);
            odometry.transform_source_to_target_ = delta *
                odometry.transform_source_to_target_;

            lines->points_.clear();
            lines->lines_.clear();
            lines->colors_.clear();
            std::vector<Vector4i>
                correspondences = odometry.correspondences_.Download();

            PinholeCameraIntrinsicCuda intrinsic = odometry.server()
                ->intrinsics_[level];

            pcl_source->Build(odometry.source()[level],
                              odometry.server()->intrinsics_[level]);
            pcl_source->colors().Fill(Vector3f(0, 1, 0));
            pcl_target->Build(odometry.target()[level],
                              odometry.server()->intrinsics_[level]);
            pcl_target->colors().Fill(Vector3f(1, 0, 0));

            std::shared_ptr<Image> src_depth = odometry.source()[level].depthf()
                .DownloadImage();
            std::shared_ptr<Image> tgt_depth = odometry.target()[level].depthf()
                .DownloadImage();

            for (int i = 0; i < correspondences.size(); ++i) {
                auto &c = correspondences[i];
                Vector2i p_src = Vector2i(c(0), c(1));
                Vector3f X_src = intrinsic.InverseProjectPixel(
                    p_src, *PointerAt<float>(*src_depth, c(0), c(1)));
                Eigen::Vector4d X_src_h = odometry.transform_source_to_target_ *
                    Eigen::Vector4d(X_src(0), X_src(1), X_src(2), 1.0);
                lines->points_.emplace_back(X_src_h.hnormalized());

                Vector2i p_tgt = Vector2i(c(2), c(3));
                Vector3f X_tgt = intrinsic.InverseProjectPixel(
                    p_tgt, *PointerAt<float>(*tgt_depth, c(2), c(3)));
                lines->points_.emplace_back(X_tgt(0), X_tgt(1), X_tgt(2));

                lines->colors_.emplace_back(0, 0, 1);
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

void ProcessGaussianImage(std::string depth_path) {
    Image depth;

    ReadImage(depth_path, depth);
    ImageCuda<Vector1s> depths;
    ImageCuda<Vector1f> depthf, gaussian;

    depths.Upload(depth);
    depths.ConvertToFloat(depthf, 1.0f / 1000.0f);

    depthf.Gaussian(gaussian, Gaussian3x3, true);

    cv::Mat matf = gaussian.DownloadMat();
    cv::Mat mats = cv::Mat(matf.rows, matf.cols, CV_16UC1);
    for (int i = 0; i < mats.rows; ++i) {
        for (int j = 0; j < mats.cols; ++j) {
            mats.at<unsigned short>(i, j) = matf.at<float>(i, j) * 1000.0f;
        }
    }
    cv::imwrite("gaussian_normal_filter.png", mats);
}

int main(int argc, char **argv) {
    if (argc != 2 || ProgramOptionExists(argc, argv, "--help")) {
        PrintHelp();
        return 1;
    }

    std::string base_path = argv[1];
    auto rgbd_filenames = ReadDataAssociation(
        base_path + "/data_association.txt");

    for (int i = 2; i < rgbd_filenames.size(); ++i) {
        TwoFrameRGBDOdometry(
            base_path + "/" + rgbd_filenames[i + 1].first,
            base_path + "/" + rgbd_filenames[i + 1].second,
            base_path + "/" + rgbd_filenames[i].first,
            base_path + "/" + rgbd_filenames[i].second);
    }

    return 0;
}