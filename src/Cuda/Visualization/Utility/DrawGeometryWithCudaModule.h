//
// Created by wei on 3/26/19.
//

#pragma once

#include <Open3D/Open3D.h>

namespace open3d {
namespace visualization {

bool DrawGeometriesWithCudaModule(
    const std::vector<std::shared_ptr<const geometry::Geometry>>
    &geometry_ptrs,
    const std::string &window_name = "Open3D",
    int width = 640,
    int height = 480,
    int left = 50,
    int top = 50);

}
}


