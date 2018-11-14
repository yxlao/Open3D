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


int main(int argc, char**argv) {
    using namespace open3d;

    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    /** Load data **/
    std::string base_path = "/home/wei/Work/data/lounge/";
    Image source_color, source_depth, target_color, target_depth;
    ReadImage(base_path + "image/000003.png", source_color);
    ReadImage(base_path + "depth/000003.png", source_depth);
    ReadImage(base_path + "image/000001.png", target_color);
    ReadImage(base_path + "depth/000001.png", target_depth);

    RGBDImageCuda source, target;
    source.Upload(source_depth, source_color);
    target.Upload(target_depth, target_color);

    /** Prepare odometry **/
    RGBDOdometryCuda<3> odometry;
    odometry.SetIntrinsics(PinholeCameraIntrinsic(
        PinholeCameraIntrinsicParameters::PrimeSenseDefault));
    odometry.SetParameters(0.0f, 0.1f, 4.0f, 0.07f);
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
    VisualizerWithCustomAnimation visualizer;
    if (!visualizer.CreateVisualizerWindow("ScalableFusion", 640, 480, 0, 0)) {
        PrintWarning("Failed creating OpenGL window.\n");
        return -1;
    }
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();
    visualizer.AddGeometry(pcl_source);
    visualizer.AddGeometry(pcl_target);

    visualizer.RegisterKeyCallback(GLFW_KEY_SPACE, [&](Visualizer* vis) {
        static const int kIterations[] = {3, 5, 10};
        static bool finished = false;
        static int level = 2;
        static int iter = kIterations[level];
        static Eigen::Matrix4d prev_transform = Eigen::Matrix4d::Identity();

        if (! finished) {
            odometry.ApplyOneIterationOnLevel(level, iter);
            pcl_source->Transform(
                odometry.transform_source_to_target_ * prev_transform.inverse());
            prev_transform = odometry.transform_source_to_target_;
            vis->UpdateGeometry();
        }

        -- iter;
        if (iter == 0) {
            -- level;
            if (level < 0) {
                finished = true;
            } else {
                iter = kIterations[level];
            }
        }

        return true;
    });

    bool should_close = false;
    while (! should_close) {
        should_close = !visualizer.PollEvents();
    }
    visualizer.DestroyVisualizerWindow();

    return 0;
}