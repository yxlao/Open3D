//
// Created by wei on 4/12/19.
//

#include <Eigen/Dense>
#include <InverseRendering/Geometry/ImageExt.h>
#include "DifferentiableRenderer.h"

namespace open3d {
namespace visualization {

namespace glsl {
bool DifferentiableRenderer::Render(const RenderOption &option,
                                    const ViewControl &view) {
    if (!is_visible_ || mutable_geometry_ptr_->IsEmpty()) return true;

    if (mutable_geometry_ptr_->GetGeometryType()
        != geometry::Geometry::GeometryType::TriangleMesh) {
        utility::PrintWarning("[DifferentiableRenderer] "
                              "Geometry type is not TriangleMesh\n");
        return false;
    }
    auto &mesh =
        (const geometry::TriangleMeshExtended &) (*mutable_geometry_ptr_);
    if (!mesh.HasMaterials()) {
        utility::PrintWarning("[DifferentiableRenderer] "
                              "Mesh does not include material\n");
        return false;
    }
    if (lighting_ptr_->GetLightingType()
        != geometry::Lighting::LightingType::IBL) {
        utility::PrintWarning("[DifferentiableRenderer] "
                              "Lighting type is not IBL\n");
        return false;
    }
    auto &ibl = (geometry::IBLLighting &) (*lighting_ptr_);

    bool success = true;
    if (!ibl.is_preprocessed_) {
        success &= PreprocessLights(ibl, option, view);
    }

    /** Major differential rendering steps **/
    success &= differential_shader_.Render(mesh, textures_, ibl, option, view);
    success &= index_shader_.Render(mesh, textures_, ibl, option, view);

    /** Visualize object changes **/
    success &=
        ibx_vertex_map_shader_.Render(mesh, textures_, ibl, option, view);

    /** Visualize background **/
    success &= background_shader_.Render(mesh, textures_, ibl, option, view);

    return success;
}

namespace {
inline Eigen::Vector3d GetVector3d(geometry::Image &im, int u, int v) {
    auto *ptr = geometry::PointerAt<float>(im, u, v, 0);
    return Eigen::Vector3d(ptr[0], ptr[1], ptr[2]);
}

inline bool HasNan(Eigen::Vector3d &in) {
    return (std::isnan(in(0)) || std::isnan(in(1)) || std::isnan(in(2)));
}

inline void Clamp(Eigen::Vector3d &in, double min_val, double max_val) {
    in(0) = std::max(std::min(in(0), max_val), min_val);
    in(1) = std::max(std::min(in(1), max_val), min_val);
    in(2) = std::max(std::min(in(2), max_val), min_val);
}

inline Eigen::Matrix3d Rotation(const Eigen::Vector3d &in) {
    return (Eigen::AngleAxisd(in(2), Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(in(1), Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(in(0), Eigen::Vector3d::UnitX()))
        .matrix();
}

Eigen::Vector2d NormalToAngle(Eigen::Vector3d normal) { // inclination (theta), azimuth (phi)
    return Eigen::Vector2d(std::acos(normal(2)),
                           std::atan2(normal(1), normal(0)));
}

Eigen::Vector3d AngleToNormal(Eigen::Vector2d angle) { // inclination, azimuth
    return Eigen::Vector3d(sin(angle(0)) * cos(angle(1)),
                           sin(angle(0)) * sin(angle(1)),
                           cos(angle(0)));
}
}

bool DifferentiableRenderer::CaptureBuffer(const std::string &filename,
                                           int index) {
    auto buffer_map = differential_shader_.fbo_outputs_[index];
    auto output_image = std::make_shared<geometry::Image>();
    output_image->PrepareImage(buffer_map->width_,
                               buffer_map->height_,
                               3, 1);
    for (int v = 0; v < buffer_map->height_; ++v) {
        for (int u = 0; u < buffer_map->width_; ++u) {
            auto colorf = GetVector3d(*buffer_map, u, v);
            auto coloru_ptr = geometry::PointerAt<uint8_t>(
                *output_image, u, buffer_map->height_ - 1 - v, 0);
            coloru_ptr[0] = uint8_t(std::min(colorf(0) * 255, 255.0));
            coloru_ptr[1] = uint8_t(std::min(colorf(1) * 255, 255.0));
            coloru_ptr[2] = uint8_t(std::min(colorf(2) * 255, 255.0));
        }
    }
    io::WriteImageToHDR(filename + ".hdr", *buffer_map);
    io::WriteImage(filename, *output_image);
}

float DifferentiableRenderer::SGD(
    float lambda,
    bool update_albedo, bool update_material, bool update_normal) {

    auto index_map = index_shader_.fbo_outputs_[0];

    auto out_render_map = differential_shader_.fbo_outputs_[0];
    auto residual_map = differential_shader_.fbo_outputs_[1];
    auto grad_albedo_map = differential_shader_.fbo_outputs_[2];
    auto grad_material_map = differential_shader_.fbo_outputs_[3];
    auto grad_normal_map = differential_shader_.fbo_outputs_[4];

    auto tmp_map = differential_shader_.fbo_outputs_[5];

    auto &mesh = (geometry::TriangleMeshExtended &) *mutable_geometry_ptr_;

    float total_residual = 0;
    int count = 0;
    for (int v = 0; v < index_map->height_; ++v) {
        for (int u = 0; u < index_map->width_; ++u) {
            int *idx = geometry::PointerAt<int>(*index_map, u, v);
            if (*idx > 0) {
                auto residual = GetVector3d(*residual_map, u, v);
                auto rendered = GetVector3d(*out_render_map, u, v);
                auto temp = GetVector3d(*tmp_map, u, v);

                if (update_albedo) {
                    auto &color = mesh.vertex_colors_[*idx];

                    if (HasNan(residual) || HasNan(rendered)) {
                        std::cout << "residual: " << residual.transpose()
                                  << "\n"
                                  << "rendered: " << rendered.transpose()
                                  << "\n"
                                  << "albedo: " << rendered.transpose() << "\n"
                                  << "color: " << color.transpose() << "\n";
                    }

                    auto grad_albedo = GetVector3d(*grad_albedo_map, u, v);
                    color -= lambda * grad_albedo;
                    Clamp(color, 0, 1);
                }

                if (update_material) {
                    auto &material = mesh.vertex_materials_[*idx];
                    auto grad_material = GetVector3d(*grad_material_map, u, v);
                    material -= lambda * grad_material;
                    Clamp(material, 0, 1);
                }

                if (update_normal) {
                    auto &normal = mesh.vertex_normals_[*idx];
                    auto grad_normal = GetVector3d(*grad_normal_map, u, v);

                    auto angle = NormalToAngle(normal);
                    angle -= lambda * Eigen::Vector2d(grad_normal(0), grad_normal(1));
                    normal = AngleToNormal(angle);
                }

//                std::cout << residual.transpose() << "\n";
                total_residual += residual.dot(residual);
                count++;
            }
        }
    }

    RebindGeometry(RenderOption(),
                   update_albedo, update_material, update_normal);

    return total_residual / count;
}

bool DifferentiableRenderer::AddMutableGeometry(
    std::shared_ptr<open3d::geometry::Geometry> &geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::TriangleMesh) {
        return false;
    }
    mutable_geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool DifferentiableRenderer::UpdateGeometry() {
    ibx_vertex_map_shader_.InvalidateGeometry();
    differential_shader_.InvalidateGeometry();
    index_shader_.InvalidateGeometry();
    background_shader_.InvalidateGeometry();
    return true;
}

bool DifferentiableRenderer::RebindTexture(const geometry::Image &image) {
    differential_shader_.RebindTexture(image);
}

bool DifferentiableRenderer::RebindGeometry(
    const RenderOption &option,
    bool rebind_color,
    bool rebind_material,
    bool rebind_normal) {
    ibx_vertex_map_shader_.RebindGeometry(
        *mutable_geometry_ptr_,
        option, rebind_color, rebind_material, rebind_normal);
    differential_shader_.RebindGeometry(
        *mutable_geometry_ptr_,
        option, rebind_color, rebind_material, rebind_normal);
}
}
}
}
