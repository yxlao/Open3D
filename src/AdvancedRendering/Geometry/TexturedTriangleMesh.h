//
// Created by wei on 4/13/19.
//

#pragma once

#include <AdvancedRendering/Geometry/ExtendedTriangleMesh.h>
#include <Open3D/Open3D.h>

#include "ImageExt.h"

namespace open3d {
namespace geometry {

/** Triangle mesh with vertex-wise uv coordinate and affliating textures **/
class TexturedTriangleMesh : public TriangleMesh {
public:
    TexturedTriangleMesh()
        : TriangleMesh(Geometry::GeometryType::TexturedTriangleMesh) {}
    TexturedTriangleMesh(const ExtendedTriangleMesh &other) {
        vertices_ = other.vertices_;
        vertex_colors_ = other.vertex_colors_;
        vertex_normals_ = other.vertex_normals_;
        vertex_uvs_ = other.vertex_uvs_;
    }

    ~TexturedTriangleMesh() override {}

    /** Manually load textures: nullptr allowed, fallback to other modes
     * e.g. phong **/
    bool LoadImageTextures(const std::vector<std::string> &filenames,
                           int default_tex_width = 512,
                           int default_tex_height = 512) {
        std::vector<std::shared_ptr<geometry::Image>> images;
        for (auto &filename : filenames) {
            auto image_ptr = io::CreateImageFromFile(filename);
            if (image_ptr->IsEmpty()) {
                utility::PrintWarning(
                        "Invalid input texture image %s, use default blank "
                        "image (%d %d) instead.\n",
                        filename.c_str(), default_tex_width,
                        default_tex_height);
                image_ptr->PrepareImage(default_tex_width, default_tex_height,
                                        3, 1);
            }
            images.emplace_back(image_ptr);
        }
        LoadImageTextures(images);
        return true;
    }

    void LoadImageTextures(
            const std::vector<std::shared_ptr<geometry::Image>> &images) {
        for (auto &image : images) {
            texture_images_.emplace_back(*FlipImageExt(*image));
        }
    }

public:
    /** texture coordinate **/
    std::vector<Eigen::Vector2d> vertex_uvs_;
    std::vector<geometry::Image> texture_images_;

public:
    bool HasUVs() const {
        return !vertex_uvs_.empty() && vertex_uvs_.size() == vertices_.size();
    }
    bool HasTextureImages(int num_textures) const {
        return texture_images_.size() == num_textures;
    }
};
}  // namespace geometry
}