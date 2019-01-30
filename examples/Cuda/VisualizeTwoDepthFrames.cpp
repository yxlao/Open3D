//
// Created by wei on 11/29/18.
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

#include "Utils.h"

using namespace open3d;

void PrintHelp() {
    PrintOpen3DVersion();
    PrintInfo("Usage :\n");
    PrintInfo("    > VisualizeTwoDepthFrames [depth_path1] [depth_path2]\n");
}

void VisualizeTwoDepthFrames(std::string depth_path1,
                             std::string depth_path2) {
    Image depth;
    ReadImage(depth_path1, depth);
    auto pcl_refined = CreatePointCloudFromDepthImage(
        depth,
        PinholeCameraIntrinsic(
            PinholeCameraIntrinsic(
                PinholeCameraIntrinsicParameters::PrimeSenseDefault)));

    ReadImage(depth_path2, depth);
    auto pcl_unrefined = CreatePointCloudFromDepthImage(
        depth,
        PinholeCameraIntrinsic(
            PinholeCameraIntrinsic(
                PinholeCameraIntrinsicParameters::PrimeSenseDefault)));

    VisualizerWithKeyCallback visualizer;
    if (!visualizer.CreateVisualizerWindow("VisualizeTwoDepthFrames",
        640, 480, 0, 0)) {
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
    if (argc != 3 || ProgramOptionExists(argc, argv, "--help")) {
        PrintHelp();
        return 1;
    }

    VisualizeTwoDepthFrames(argv[1], argv[2]);

    return 0;
}