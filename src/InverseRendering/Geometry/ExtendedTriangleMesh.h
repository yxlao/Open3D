//
// Created by wei on 4/13/19.
//

#pragma once

#include <Open3D/Open3D.h>
namespace open3d {
namespace geometry {

/** This kind of mesh either have
 * - uvs & textures, or
 * - per vertex properties
 * TODO: split them in two classes (necessary?)
 */
class ExtendedTriangleMesh : public TriangleMesh {
public:
    ExtendedTriangleMesh() : TriangleMesh(
        Geometry::GeometryType::ExtendedTriangleMesh) {}
    ~ExtendedTriangleMesh() override {}

    /** texture coordinate **/
    std::vector<Eigen::Vector2d> vertex_uvs_;
    std::vector<geometry::Image> image_textures_;

    /** @color and @normal: already-stored.
     *  @material holds: roughness, metallic, ao. **/
    std::vector<Eigen::Vector3d> vertex_textures_;

public:
    bool HasUVs() const {
        return !vertex_uvs_.empty()
            && vertex_uvs_.size() == vertices_.size();
    }
    bool HasTexturesMaps() const {
        return !image_textures_.empty();
    }

    bool HasMaterials() const {
        return !vertex_textures_.empty()
            && vertex_textures_.size() == vertices_.size();
    }
};
}
}