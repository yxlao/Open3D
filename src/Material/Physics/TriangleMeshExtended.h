//
// Created by wei on 4/13/19.
//

#pragma once

#include <Open3D/Open3D.h>
namespace open3d {
namespace geometry {

class TriangleMeshExtended : public TriangleMesh {
public:
    /** texture coordinate **/
    std::vector<Eigen::Vector2d> vertex_uvs_;

    /** color and normal: pre-stored **/
    std::vector<float> roughness_;
    std::vector<float> metallic_;
    std::vector<float> ao_;

public:
    bool HasUVs() const {
        return vertex_uvs_.size() != 0
            && vertex_uvs_.size() == vertices_.size();
    }

    bool HasRoughness() const {
        return !roughness_.empty() && roughness_.size() == vertices_.size();
    }

    bool HasMetallic() const {
        return !metallic_.empty() && metallic_.size() == vertices_.size();
    }

    bool HasAo() const {
        return !ao_.empty() && ao_.size() == vertices_.size();
    }

};
}
}