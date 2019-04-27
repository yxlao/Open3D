//
// Created by wei on 4/16/19.
//

#pragma once

#include <Eigen/Core>

namespace open3d {
namespace geometry {

///** Face inside **/
const std::vector<Eigen::Vector3f> kCubeVertices = {
    // back face
    Eigen::Vector3f(-1.0f, -1.0f, -1.0f),
    Eigen::Vector3f(1.0f, -1.0f, -1.0f),
    Eigen::Vector3f(1.0f, 1.0f, -1.0f),
    Eigen::Vector3f(-1.0f, 1.0f, -1.0f),
    // front face
    Eigen::Vector3f(-1.0f, -1.0f, 1.0f),
    Eigen::Vector3f(1.0f, 1.0f, 1.0f),
    Eigen::Vector3f(1.0f, -1.0f, 1.0f),
    Eigen::Vector3f(-1.0f, 1.0f, 1.0f),
};

const std::vector<Eigen::Vector3i> kCubeTriangles = {
    Eigen::Vector3i(0, 1, 2),
    Eigen::Vector3i(2, 3, 0),
    Eigen::Vector3i(4, 5, 6),
    Eigen::Vector3i(5, 4, 7),
    Eigen::Vector3i(7, 0, 3),
    Eigen::Vector3i(0, 7, 4),
    Eigen::Vector3i(5, 2, 1),
    Eigen::Vector3i(1, 6, 5),
    Eigen::Vector3i(0, 6, 1),
    Eigen::Vector3i(6, 0, 4),
    Eigen::Vector3i(3, 2, 5),
    Eigen::Vector3i(5, 7, 3),
};

const std::vector<Eigen::Vector3f> kQuadVertices = {
    Eigen::Vector3f(-1.0f, 1.0f, 0.0f),
    Eigen::Vector3f(-1.0f, -1.0f, 0.0f),
    Eigen::Vector3f(1.0f, 1.0f, 0.0f),
    Eigen::Vector3f(1.0f, -1.0f, 0.0f)
};

const std::vector<Eigen::Vector2f> kQuadUVs = {
    Eigen::Vector2f(0.0f, 1.0f),
    Eigen::Vector2f(0.0f, 0.0f),
    Eigen::Vector2f(1.0f, 1.0f),
    Eigen::Vector2f(1.0f, 0.0f),
};

}
}