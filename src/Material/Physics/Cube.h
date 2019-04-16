//
// Created by wei on 4/16/19.
//

#pragma once

namespace open3d {
namespace physics {

const std::vector<float> kCubeVertices = {
    // back face
    -1.0f, -1.0f, -1.0f,
    1.0f,  1.0f, -1.0f,
    1.0f, -1.0f, -1.0f,
    1.0f,  1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f,
    -1.0f,  1.0f, -1.0f,
    // front face
    -1.0f, -1.0f,  1.0f,
    1.0f, -1.0f,  1.0f,
    1.0f,  1.0f,  1.0f,
    1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,
    // left face
    -1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f,  1.0f,
    // right face
    1.0f,  1.0f,  1.0f,
    1.0f, -1.0f, -1.0f,
    1.0f,  1.0f, -1.0f,
    1.0f, -1.0f, -1.0f,
    1.0f,  1.0f,  1.0f,
    1.0f, -1.0f,  1.0f,
    // bottom face
    -1.0f, -1.0f, -1.0f,
    1.0f, -1.0f, -1.0f,
    1.0f, -1.0f,  1.0f,
    1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f, -1.0f,
    // top face
    -1.0f,  1.0f, -1.0f,
    1.0f,  1.0f , 1.0f,
    1.0f,  1.0f, -1.0f,
    1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f
};
}
}