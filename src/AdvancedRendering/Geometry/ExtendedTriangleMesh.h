//
// Created by wei on 4/13/19.
//

#pragma once

#include <Open3D/Open3D.h>
#include "ImageExt.h"
namespace open3d {
namespace geometry {

/** Triangle mesh with vertex-wise customized material properties **/
class ExtendedTriangleMesh : public TriangleMesh {
public:
    ExtendedTriangleMesh() : TriangleMesh(
        Geometry::GeometryType::ExtendedTriangleMesh) {}
    ~ExtendedTriangleMesh() override {}

public:
    /** @color and @normal: already-stored.
     *  @material holds: roughness, metallic, ao. **/
    std::vector<Eigen::Vector3d> vertex_textures_;

    /** Legacy: for reference **/
    std::vector<Eigen::Vector2d> vertex_uvs_;

public:
    bool HasVertexTextures() const {
        return !vertex_textures_.empty()
            && vertex_textures_.size() == vertices_.size();
    }

    bool HasVertexUVs() const {
        return !vertex_uvs_.empty()
            && vertex_uvs_.size() == vertices_.size();
    }
};
}
}