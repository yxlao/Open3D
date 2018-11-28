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

using namespace open3d;

int TestCudaRGBDOdometry(
    std::string source_color_path,
    std::string source_depth_path,
    std::string target_color_path,
    std::string target_depth_path) {
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
    odometry.SetParameters(0.5f, 0.001f, 3.0f, 0.03f);
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

    visualizer.RegisterKeyCallback(GLFW_KEY_SPACE, [&](Visualizer *vis) {
        if (!finished) {
            float loss = odometry.ApplyOneIterationOnLevel(level, iter);

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

            losses[level].push_back(loss);

            pcl_source->Transform(
                odometry.transform_source_to_target_);
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

void TestDifferentGaussianKernel(std::string depth_path_refined,
                                 std::string depth_path_unrefined) {
    Image depth;
    ReadImage(depth_path_refined, depth);
    auto pcl_refined = CreatePointCloudFromDepthImage(
        depth,
        PinholeCameraIntrinsic(
            PinholeCameraIntrinsic(
                PinholeCameraIntrinsicParameters::PrimeSenseDefault)));

    ReadImage(depth_path_unrefined, depth);
    auto pcl_unrefined = CreatePointCloudFromDepthImage(
        depth,
        PinholeCameraIntrinsic(
            PinholeCameraIntrinsic(
                PinholeCameraIntrinsicParameters::PrimeSenseDefault)));

    VisualizerWithKeyCallback visualizer;
    if (!visualizer.CreateVisualizerWindow("Gaussian", 640, 480, 0, 0)) {
        PrintWarning("Failed creating OpenGL window.\n");
        return;
    }
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();

    auto pcl = CreatePointCloudFromDepthImage(
        depth,
        PinholeCameraIntrinsic(
            PinholeCameraIntrinsic(
                PinholeCameraIntrinsicParameters::PrimeSenseDefault)));
    visualizer.AddGeometry(pcl);

    bool refined = false;
    visualizer.RegisterKeyCallback(GLFW_KEY_SPACE, [&](Visualizer *vis) {
        refined = !refined;
        if (refined) {
            pcl->points_ = pcl_refined->points_;
            pcl->colors_ = pcl_refined->colors_;
        } else {
            pcl->points_ = pcl_unrefined->points_;
            pcl->colors_ = pcl_unrefined->colors_;
        }
        vis->UpdateGeometry();
        return true;
    });

    bool should_close = false;
    while (!should_close) {
        should_close = !visualizer.PollEvents();
    }
    visualizer.DestroyVisualizerWindow();
}

int main(int argc, char **argv) {
    std::string base_path = "/home/wei/Work/data/stanford/lounge/";
//    ProcessGaussianImage(base_path + "depth/000004.png");

    TestDifferentGaussianKernel("gaussian_normal_filter.png",
                                "gaussian_refined_filter.png");

//    TestCudaRGBDOdometry(base_path + "color/000004.png",
//                         base_path + "depth/000004.png",
//                         base_path + "color/000001.png",
//                         base_path + "depth/000001.png");
//    TestCudaRGBDOdometry(base_path + "color/000366.png",
//                         base_path + "depth/000366.png",
//                         base_path + "color/000365.png",
//                         base_path + "depth/000365.png");
//    for (int i = 360; i < 3000; ++i) {
//        std::stringstream ss;
//        ss.str("");
//        ss << base_path << "color/"
//           << std::setw(6) << std::setfill('0') << i << ".png";
//        std::string target_color_path = ss.str();
//
//        ss.str("");
//        ss << base_path << "depth/"
//           << std::setw(6) << std::setfill('0') << i << ".png";
//        std::string target_depth_path = ss.str();
//
//        ss.str("");
//        ss << base_path << "color/"
//           << std::setw(6) << std::setfill('0') << i + 1 << ".png";
//        std::string source_color_path = ss.str();
//
//        ss.str("");
//        ss << base_path << "depth/"
//           << std::setw(6) << std::setfill('0') << i + 1 << ".png";
//        std::string source_depth_path = ss.str();
//
//        std::cout << target_color_path << std::endl;
//        TestCudaRGBDOdometry(source_color_path, source_depth_path,
//            target_color_path, target_depth_path);
//    }
}