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
    geometry::TexturedTriangleMesh &mesh);

/** Unsupported now: tinyobjload does not support writing **/
bool WriteTexturedTriangleMeshToOBJ(
    const std::string &filename,
    /* size = 1: diffuse;
     * size = 5: diffuse, normal, roughness, metallic, ambient */
    const std::vector<std::string> &textures,
    const geometry::TexturedTriangleMesh &mesh);
}
}