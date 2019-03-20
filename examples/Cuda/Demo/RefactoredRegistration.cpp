//
// Created by wei on 3/19/19.
//

#include <string>
#include <vector>
#include <Open3D/Open3D.h>

#include <Cuda/Odometry/RGBDOdometryCuda.h>
#include <Cuda/Geometry/PointCloudCuda.h>
#include <Cuda/Registration/RegistrationCuda.h>

#include "examples/Cuda/DatasetConfig.h"
#include "examples/Cuda/Utils.h"

using namespace open3d;

int TwoFragmentRegistration(
    const std::string &source_ply_path,
    const std::string &target_ply_path) {

    SetVerbosityLevel(utility::VerbosityLevel::VerboseDebug);

    auto source = io::CreatePointCloudFromFile(source_ply_path);
    auto target = io::CreatePointCloudFromFile(target_ply_path);

    /** Load data **/
    cuda::RegistrationCuda registration(
        registration::TransformationEstimationType::PointToPlane);
    registration.Initialize(*source, *target, 0.07f);

    /** Prepare visualizer **/
    visualization::VisualizerWithKeyCallback visualizer;
    if (!visualizer.CreateVisualizerWindow("ColoredICP", 640, 480, 0, 0)) {
        utility::PrintWarning("Failed creating OpenGL window.\n");
        return -1;
    }
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();
    visualizer.AddGeometry(source);
    visualizer.AddGeometry(target);

    bool finished = false;
    int iter = 0, max_iter = 50;
    visualizer.RegisterKeyCallback(GLFW_KEY_SPACE, [&](visualization::Visualizer *vis) {
        if (finished) return false;

        /* Odometry (1 iteration) */
        registration.DoSingleIteration(iter++);
        *source = registration.source_cpu_;

        /* Re-bind geometry */
        vis->UpdateGeometry();

        /* Update masks */
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
    DatasetConfig config;

    std::string config_path = argc > 1 ? argv[1]
                                       : kDefaultDatasetConfigDir
                                  + "/redwood_simulated/livingroom2.json";
    bool is_success = io::ReadIJsonConvertible(config_path, config);
    if (!is_success) return 1;

    is_success = config.GetThumbnailFragmentFiles();
    if (!is_success) return 1;

    for (int i = 2; i < config.thumbnail_fragment_files_.size() - 1; ++i) {
        utility::PrintInfo("%d -> %d\n", i, i + 1);
        TwoFragmentRegistration(
            config.thumbnail_fragment_files_[i],
            config.thumbnail_fragment_files_[i + 1]);
    }

    return 0;
}