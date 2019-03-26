//
// Created by wei on 3/26/19.
//

#include "DrawGeometryWithCudaModule.h"

#include <Cuda/Visualization/Visualizer/VisualizerWithCudaModule.h>

namespace open3d {
namespace visualization {

bool DrawGeometriesWithCudaModule(
    const std::vector<std::shared_ptr<const geometry::Geometry>>
    &geometry_ptrs,
    const std::string &window_name /* = "Open3D"*/,
    int width /* = 640*/,
    int height /* = 480*/,
    int left /* = 50*/,
    int top /* = 50*/) {
    VisualizerWithCudaModule visualizer;
    if (visualizer.CreateVisualizerWindow(window_name,
        width, height, left, top) == false) {
        utility::PrintWarning(
            "[DrawGeometries] Failed creating OpenGL window.\n");
        return false;
    }

    for (const auto &geometry_ptr : geometry_ptrs) {
        if (visualizer.AddGeometry(geometry_ptr) == false) {
            utility::PrintWarning("[DrawGeometries] Failed adding geometry.\n");
            utility::PrintWarning(
                "[DrawGeometries] Possibly due to bad geometry or wrong "
                "geometry type.\n");
            return false;
        }
    }
    visualizer.Run();
    visualizer.DestroyVisualizerWindow();
    return true;
}
}
}