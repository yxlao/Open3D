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
#include <Cuda/Odometry/SequentialRGBDOdometryCuda.h>
#include <Cuda/Integration/ScalableTSDFVolumeCuda.h>
#include <Cuda/Integration/ScalableMeshVolumeCuda.h>
#include <Cuda/Geometry/PointCloudCuda.h>
#include <Visualization/Visualization.h>

#include <opencv2/opencv.hpp>
#include <thread>

using namespace open3d;

std::shared_ptr<RGBDImage> ReadRGBDImage(
    const char *color_filename, const char *depth_filename,
    const PinholeCameraIntrinsic &intrinsic,
    bool visualize) {
    Image color, depth;
    ReadImage(color_filename, color);
    ReadImage(depth_filename, depth);
    PrintDebug("Reading RGBD image : \n");
    PrintDebug("     Color : %d x %d x %d (%d bits per channel)\n",
               color.width_, color.height_,
               color.num_of_channels_, color.bytes_per_channel_ * 8);
    PrintDebug("     Depth : %d x %d x %d (%d bits per channel)\n",
               depth.width_, depth.height_,
               depth.num_of_channels_, depth.bytes_per_channel_ * 8);
    double depth_scale = 1000.0, depth_trunc = 4.0;
    bool convert_rgb_to_intensity = true;
    std::shared_ptr<RGBDImage> rgbd_image =
        CreateRGBDImageFromColorAndDepth(color,
                                         depth,
                                         depth_scale,
                                         depth_trunc,
                                         convert_rgb_to_intensity);
    if (visualize) {
        auto pcd = CreatePointCloudFromRGBDImage(*rgbd_image, intrinsic);
        DrawGeometries({pcd});
    }
    return rgbd_image;
}

int TestNativeRGBDOdometry() {
    using namespace open3d;
    PinholeCameraIntrinsic intrinsic = PinholeCameraIntrinsic(
        PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    bool visualize = true;
    std::string base_path = "/home/wei/Work/data/stanford/lounge/";
    Image source_color, source_depth, target_color, target_depth;

    auto source = ReadRGBDImage((base_path + "color/000026.png").c_str(),
                                (base_path + "depth/000026.png").c_str(),
                                intrinsic, visualize);
    auto target = ReadRGBDImage((base_path + "color/000025.png").c_str(),
                                (base_path + "depth/000025.png").c_str(),
                                intrinsic, visualize);

    Eigen::Matrix4d odo_init = Eigen::Matrix4d::Identity();
    std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d> rgbd_odo =
        ComputeRGBDOdometry(*source, *target, intrinsic, odo_init,
                            RGBDOdometryJacobianFromHybridTerm(),
                            OdometryOption({60, 60, 60}, 0.03, 0.0, 3.0));
    std::cout << "RGBD Odometry" << std::endl;
    std::cout << std::get<1>(rgbd_odo) << std::endl;

    auto pcl_source = CreatePointCloudFromRGBDImage(*source, intrinsic);
    auto pcl_target = CreatePointCloudFromRGBDImage(*target, intrinsic);
    pcl_source->Transform(std::get<1>(rgbd_odo));
    DrawGeometries({pcl_source, pcl_target});
}

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
    odometry.SetParameters(1.0f, 0.01f, 3.0f, 0.03f);
    odometry.PrepareData(source, target);
    odometry.transform_source_to_target_ = Eigen::Matrix4d::Identity();

    /** Prepare point cloud **/
    std::shared_ptr<PointCloudCuda>
        pcl_source = std::make_shared<PointCloudCuda>(VertexWithColor, 300000),
        pcl_target = std::make_shared<PointCloudCuda>(VertexWithColor, 300000);

    pcl_source->Build(odometry.source()[2], odometry.server()->intrinsics_[2]);
    pcl_source->colors().Fill(Vector3f(0, 0, 1));
    pcl_target->Build(odometry.target()[2], odometry.server()->intrinsics_[2]);
    pcl_target->colors().Fill(Vector3f(1, 1, 0));

    /** Prepare visualizer **/
    VisualizerWithKeyCallback visualizer;
    if (!visualizer.CreateVisualizerWindow("ScalableFusion", 640, 480, 0, 0)) {
        PrintWarning("Failed creating OpenGL window.\n");
        return -1;
    }
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();
    visualizer.AddGeometry(pcl_source);
    visualizer.AddGeometry(pcl_target);

    std::vector<float> losses[3];
    const int kIterations[3] = {60, 60, 60};
    bool finished = false;
    int level = 2;
    int iter = kIterations[level];
    Eigen::Matrix4d prev_transform = Eigen::Matrix4d::Identity();
    visualizer.RegisterKeyCallback(GLFW_KEY_SPACE, [&](Visualizer *vis) {

        if (!finished) {
            float loss = odometry.ApplyOneIterationOnLevel(level, iter);
            losses[level].push_back(loss);

            pcl_source->Transform(
                odometry.transform_source_to_target_
                    * prev_transform.inverse());
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
    std::string base_path = "/home/wei/Work/data/stanford/lounge/";
    TestCudaRGBDOdometry(base_path + "color/000026.png",
                         base_path + "depth/000026.png",
                         base_path + "color/000025.png",
                         base_path + "depth/000025.png");
//    for (int i = 1; i < 3000; ++i) {
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
    TestNativeRGBDOdometry();
}