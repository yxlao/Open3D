//
// Created by wei on 4/15/19.
//

#pragma once

#include <AdvancedRendering/Visualization/Visualizer/VisualizerPBR.h>

namespace open3d {
namespace visualization {
bool DrawGeometriesPBR(
    const std::vector<std::shared_ptr<const geometry::Geometry>> &geometry_ptrs,
    const std::shared_ptr<const geometry::Lighting> &lighting,
    const std::string &window_name = "Open3D",
    int width = 640,
    int height = 480,
    int left = 50,
    int top = 50);
}
}