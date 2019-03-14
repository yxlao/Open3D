//
// Created by wei on 3/1/19.
//

#include <string>
#include <vector>
#include <Core/Core.h>
#include <IO/IO.h>

#include <Cuda/Odometry/RGBDOdometryCuda.h>
#include <Cuda/Geometry/PointCloudCuda.h>
#include <Cuda/Registration/FastGlobalRegistrationCuda.h>

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

    auto source0 = *source;
    auto target0 = *target;

    /** Load data **/
    cuda::FastGlobalRegistrationCuda fgr;
    fgr.Initialize(*source, *target);

    /** Prepare visualizer **/
    VisualizerWithKeyCallback visualizer;
    if (!visualizer.CreateVisualizerWindow("Fast Global Registration", 640, 480,0, 0)) {
        PrintWarning("Failed creating OpenGL window.\n");
        return -1;
    }
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();
    visualizer.AddGeometry(source);
    visualizer.AddGeometry(target);

    bool finished = false;
    int iter = 0, max_iter = 64;
    visualizer.RegisterKeyCallback(GLFW_KEY_SPACE, [&](Visualizer *vis) {
        if (finished) return false;

        /* Odometry (1 iteration) */
        fgr.DoSingleIteration(iter++);
        *source = *fgr.source_.Download();
        source->colors_ = source0.colors_;

        *target = *fgr.target_.Download();
        target->colors_ = target0.colors_;

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
                                  + "/tum/fr3_household.json";
    bool is_success = ReadIJsonConvertible(config_path, config);
    if (!is_success) return 1;

    is_success = config.GetThumbnailFragmentFiles();
    if (!is_success) return 1;

    for (int i = 13; i < config.thumbnail_fragment_files_.size() - 5; ++i) {
        PrintInfo("%d -> %d\n", i, i + 3);
        TwoFragmentRegistration(
            config.thumbnail_fragment_files_[i],
            config.thumbnail_fragment_files_[i + 3]);
    }

    return 0;
}