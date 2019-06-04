//
// Created by wei on 4/15/19.
//

#pragma once

#include <AdvancedRendering/Visualization/Visualizer/VisualizerPBR.h>

namespace open3d {
namespace visualization {
bool DrawGeometriesUV(
    const std::vector<std::shared_ptr<const geometry::Geometry>> &geometry_ptrs,
    const bool forward = true,
    const std::shared_ptr<geometry::Image> &target = nullptr,
    const std::string &window_name = "Open3D",
    int width = 1024,
    int height = 1024,
    int left = 50,
    int top = 50);
}
}