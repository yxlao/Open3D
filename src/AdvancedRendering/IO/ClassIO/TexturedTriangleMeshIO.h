//
// Created by Wei Dong on 2019-06-03.
//

#pragma once

#include <AdvancedRendering/Geometry/TexturedTriangleMesh.h>

namespace open3d {
namespace io {
/** To be separated in another class **/
bool ReadTexturedTriangleMeshFromOBJ(
    const std::string &filename,
    geometry::TexturedTriangleMesh &mesh,
    int default_tex_width = 512,
    int default_tex_height = 512);
}
}