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

    /** @color and @normal: already-stored.
     *  @material holds: roughness, metallic, ao. **/
    std::vector<Eigen::Vector3d> vertex_materials_;

public:
    bool HasUVs() const {
        return !vertex_uvs_.empty()
            && vertex_uvs_.size() == vertices_.size();
    }

    bool HasMaterials() const {
        return !vertex_materials_.empty()
            && vertex_materials_.size() == vertices_.size();
    }
};
}
}