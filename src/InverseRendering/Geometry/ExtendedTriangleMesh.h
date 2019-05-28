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

    /** We need to have uv coordinates **/
    bool LoadImageTextures(std::vector<std::string> &filenames) {
        std::vector<std::shared_ptr<geometry::Image>> images;
        for (auto &filename : filenames) {
            auto image_ptr = io::CreateImageFromFile(filename);
            if (!image_ptr) {
                utility::PrintError(
                    "Invalid input texture image %s abort\n.",
                    filename.c_str());
                return false;
            }
        }
        LoadImageTextures(images);
        return true;
    }

    void LoadImageTextures(std::vector<std::shared_ptr<geometry::Image>> &images) {
        for (auto &image : images) {
            // TODO: check if we need to flip here.
            image_textures_.emplace_back(*image);
        }
    }

public:
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
    bool HasImageTextures() const {
        return !image_textures_.empty();
    }

    bool HasVertexTextures() const {
        return !vertex_textures_.empty()
            && vertex_textures_.size() == vertices_.size();
    }
};
}
}