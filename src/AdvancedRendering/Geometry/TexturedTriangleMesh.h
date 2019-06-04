//
// Created by wei on 4/13/19.
//

#pragma once

#include <Open3D/Open3D.h>
#include <AdvancedRendering/Geometry/ExtendedTriangleMesh.h>

#include "ImageExt.h"

namespace open3d {
namespace geometry {

/** Triangle mesh with vertex-wise uv coordinate and affliating textures **/
class TexturedTriangleMesh : public TriangleMesh {
public:
    TexturedTriangleMesh() : TriangleMesh(
        Geometry::GeometryType::TexturedTriangleMesh) {}
    TexturedTriangleMesh(const ExtendedTriangleMesh &other) {
        vertices_ = other.vertices_;
        vertex_colors_ = other.vertex_colors_;
        vertex_normals_ = other.vertex_normals_;
        vertex_uvs_ = other.vertex_uvs_;
    }

    ~TexturedTriangleMesh() override {}

    /** Manually load textures **/
    bool LoadImageTextures(const std::vector<std::string> &filenames) {
        std::vector<std::shared_ptr<geometry::Image>> images;
        for (auto &filename : filenames) {
            auto image_ptr = io::CreateImageFromFile(filename);
            if (!image_ptr) {
                utility::PrintError("Invalid input texture image %s abort\n.",
                                    filename.c_str());
                return false;
            }
            images.emplace_back(image_ptr);
        }
        LoadImageTextures(images);
        return true;
    }

    void LoadImageTextures(
        const std::vector<std::shared_ptr<geometry::Image>> &images) {
        for (auto &image : images) {
            image_textures_.emplace_back(*FlipImageExt(*image));
        }
    }

public:
    /** texture coordinate **/
    std::vector<Eigen::Vector2d> vertex_uvs_;
    std::vector<geometry::Image> image_textures_;

public:
    bool HasUVs() const {
        return !vertex_uvs_.empty()
            && vertex_uvs_.size() == vertices_.size();
    }
    bool HasImageTextures(int num_textures) const {
        return image_textures_.size() == num_textures;
    }
};
}
}