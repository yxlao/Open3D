//
// Created by wei on 3/1/19.
//

#include <string>
#include <vector>
#include <Core/Core.h>
#include <IO/IO.h>

#include <Cuda/Odometry/RGBDOdometryCuda.h>
#include <Cuda/Geometry/PointCloudCuda.h>
#include <Cuda/Registration/RegistrationCuda.h>

#include <Visualization/Visualization.h>

#include "examples/Cuda/DatasetConfig.h"
#include "examples/Cuda/Utils.h"

using namespace open3d;

int TwoFragmentRegistration(
    const std::string &source_ply_path,
    const std::string &target_ply_path) {

    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    auto source = CreatePointCloudFromFile(source_ply_path);
    auto target = CreatePointCloudFromFile(target_ply_path);

    /** Load data **/
    cuda::RegistrationCuda registration(
        TransformationEstimationType::ColoredICP);
    registration.Initialize(*source, *target, 0.07f);

    /** Prepare visualizer **/
    VisualizerWithKeyCallback visualizer;
    if (!visualizer.CreateVisualizerWindow("ColoredICP", 640, 480, 0, 0)) {
        PrintWarning("Failed creating OpenGL window.\n");
        return -1;
    }
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();
    visualizer.AddGeometry(source);
    visualizer.AddGeometry(target);

    bool finished = false;
    int iter = 0, max_iter = 50;
    visualizer.RegisterKeyCallback(GLFW_KEY_SPACE, [&](Visualizer *vis) {
        if (finished) return false;

        /* Odometry (1 iteration) */
        registration.DoSingleIteration(iter++);
        *source = registration.estimator_->source_cpu_;

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
    bool is_success = ReadIJsonConvertible(config_path, config);
    if (!is_success) return 1;

    is_success = config.GetThumbnailFragmentFiles();
    if (!is_success) return 1;

    for (int i = 2; i < config.thumbnail_fragment_files_.size() - 1; ++i) {
        PrintInfo("%d -> %d\n", i, i + 1);
        TwoFragmentRegistration(
            config.thumbnail_fragment_files_[i],
            config.thumbnail_fragment_files_[i + 1]);
    }

    return 0;
}