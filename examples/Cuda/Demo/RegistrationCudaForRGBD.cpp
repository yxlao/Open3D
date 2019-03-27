//
// Created by wei on 3/19/19.
//

#include <string>
#include <vector>

#include <Open3D/Open3D.h>
#include <Cuda/Open3DCuda.h>

#include "Utils.h"

using namespace open3d;
using namespace open3d::camera;
using namespace open3d::registration;
using namespace open3d::utility;
using namespace open3d::io;
using namespace open3d::visualization;

int RegistrationForRGBDFrames(
    const std::string &source_color,
    const std::string &source_depth,
    const std::string &target_color,
    const std::string &target_depth) {

    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    PinholeCameraIntrinsic intrinsic = PinholeCameraIntrinsic(
        PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    const float depth_scale = 1000;

    auto rgbd_source = ReadRGBDImage(source_color, source_depth, intrinsic, depth_scale);
    auto rgbd_target = ReadRGBDImage(target_color, target_depth, intrinsic, depth_scale);

    auto source_origin = CreatePointCloudFromRGBDImage(*rgbd_source, intrinsic);
    auto target_origin = CreatePointCloudFromRGBDImage(*rgbd_target, intrinsic);
    EstimateNormals(*source_origin);
    EstimateNormals(*target_origin);

    auto source_down = VoxelDownSample(*source_origin, 0.02);
    auto target_down = VoxelDownSample(*target_origin, 0.02);

    /** Load data **/
    cuda::RegistrationCuda registration(TransformationEstimationType::ColoredICP);
    registration.Initialize(*source_down, *target_down, 0.05f);

    /** Prepare visualizer **/
    VisualizerWithCudaModule visualizer;
    if (!visualizer.CreateVisualizerWindow("ColoredICP", 640, 480, 0, 0)) {
        PrintWarning("Failed creating OpenGL window.\n");
        return -1;
    }
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();
    visualizer.AddGeometry(source_down);
    visualizer.AddGeometry(target_down);

    bool finished = false;
    int iter = 0, max_iter = 50;
    visualizer.RegisterKeyCallback(GLFW_KEY_SPACE, [&](Visualizer *vis) {
        if (finished) return false;

        /* Registration (1 iteration) */
        auto delta = registration.DoSingleIteration(iter++);

        /* Updated source */
        source_down->Transform(delta.transformation_);
        vis->UpdateGeometry();

        /* Update flags */
        if (iter >= max_iter)
            finished = true;
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

    return RegistrationForRGBDFrames(source_color_path, source_depth_path,
        target_color_path, target_depth_path);
}