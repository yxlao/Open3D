//
// Created by wei on 4/13/19.
//

#pragma once

#include <Open3D/Open3D.h>
namespace open3d {
namespace geometry {

class TriangleMeshPhysics : public TriangleMesh {
public:
    std::vector<Eigen::Vector2f> vertex_uvs_;

public:
    bool HasUVs() const {
        return vertex_uvs_.size() != 0
            && vertex_uvs_.size() == vertices_.size();
    }

};
}
}