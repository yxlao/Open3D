//
// Created by wei on 3/19/19.
//

#include <string>
#include <vector>

#include <Open3D/Open3D.h>
#include <Cuda/Open3DCuda.h>

#include "Utils.h"

using namespace open3d;
using namespace open3d::registration;
using namespace open3d::utility;
using namespace open3d::io;
using namespace open3d::visualization;

int RegistrationForPointClouds(
    const std::string &source_ply_path,
    const std::string &target_ply_path,
    const TransformationEstimationType &type) {

    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    auto source_origin = CreatePointCloudFromFile(source_ply_path);
    auto target_origin = CreatePointCloudFromFile(target_ply_path);

    if (type == TransformationEstimationType::ColoredICP
    && (! source_origin->HasColors() || ! target_origin->HasColors())) {
        PrintError("Point cloud does not have color, abort.\n");
        return -1;
    }
    auto source_down = VoxelDownSample(*source_origin, 0.03);
    auto target_down = VoxelDownSample(*target_origin, 0.03);

    /** Load data **/
    cuda::RegistrationCuda registration(type);
    registration.Initialize(*source_origin, *target_origin, 0.07f);

    /** Prepare visualizer **/
    VisualizerWithCudaModule visualizer;
    if (!visualizer.CreateVisualizerWindow("ColoredICP", 640, 480, 0, 0)) {
        PrintWarning("Failed creating OpenGL window.\n");
        return -1;
    }
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();
    visualizer.AddGeometry(source_origin);
    visualizer.AddGeometry(target_origin);

    bool finished = false;
    int iter = 0, max_iter = 50;
    visualizer.RegisterKeyCallback(GLFW_KEY_SPACE, [&](Visualizer *vis) {
        if (finished) return false;

        /* Registration (1 iteration) */
        auto delta = registration.DoSingleIteration(iter++);
        source_origin->Transform(delta.transformation_);

        /* Updated source */
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
    std::string source_path, target_path;
    TransformationEstimationType type;
    if (argc > 1) {
        std::string str_type = std::string(argv[1]);
        if (str_type == "PointToPlane") {
            type = TransformationEstimationType::PointToPlane;
        } else if (str_type == "PointToPoint") {
            type = TransformationEstimationType::PointToPoint;
        } else if (str_type == "ColoredICP") { /* Default */
            type = TransformationEstimationType ::ColoredICP;
        } else {
            PrintError("Unknown type, abort.\n");
            return -1;
        }
        PrintInfo("Using registration type: %s.\n", str_type.c_str());
    } else {
        PrintInfo("Using default registration type: ColoredICP.\n");
        type = TransformationEstimationType ::ColoredICP;
    }

    if (argc > 3) {
        source_path = argv[2];
        target_path = argv[3];
    } else {
        std::string test_data_path = "../../../examples/TestData/ColoredICP";
        source_path = test_data_path + "/frag_115.ply";
        target_path = test_data_path + "/frag_116.ply";
    }

    return RegistrationForPointClouds(source_path, target_path, type);
}