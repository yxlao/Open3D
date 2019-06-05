//
// Created by wei on 4/15/19.
//

#include <AdvancedRendering/Visualization/Visualizer/VisualizerUV.h>
#include "DrawGeometryPBR.h"

namespace open3d {
namespace visualization {
bool DrawGeometriesUV(
    const std::vector<std::shared_ptr<const geometry::Geometry>> &geometry_ptrs,
    const bool forward /* = true */,
    const std::shared_ptr<geometry::Image> &target /* = nullptr */,
    const std::string &window_name /* = "Open3D"*/,
    int width /* = 640*/,
    int height /* = 480*/,
    int left /* = 50*/,
    int top /* = 50*/) {

    VisualizerUV visualizer;

    if (! visualizer.CreateVisualizerWindow(
        window_name, width, height, left, top)) {
        utility::PrintWarning(
            "[DrawGeometriesUV] Failed creating OpenGL window.\n");
        return false;
    }

    for (auto &geometry_ptr : geometry_ptrs) {
        if (! visualizer.AddGeometry(geometry_ptr)) {
            utility::PrintWarning(
                "[DrawGeometriesUV] Failed adding geometry.\n");
            utility::PrintWarning(
                "[DrawGeometriesUV] Possibly due to bad geometry or wrong "
                "geometry type.\n");
            return false;
        }
    }

    visualizer.Setup(forward, target);

    visualizer.Run();
    visualizer.DestroyVisualizerWindow();
    return true;
}
}
}