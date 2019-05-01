//
// Created by wei on 4/12/19.
//

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
    success &= SGD(0.1f);
    RebindGeometry(option, true, true, false);
    return success;
}

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

bool DifferentiableRenderer::SGD(float lambda) {
    std::cout << "SGD\n";
    auto index_map = index_shader_.fbo_outputs_[0];

    auto out_render_map = differential_shader_.fbo_outputs_[0];
    auto residual_map = differential_shader_.fbo_outputs_[1];
    auto grad_albedo_map = differential_shader_.fbo_outputs_[2];
    auto grad_material_map = differential_shader_.fbo_outputs_[3];

    auto &mesh = (geometry::TriangleMeshExtended &) *mutable_geometry_ptr_;
    for (int v = 0; v < index_map->height_; ++v) {
        for (int u = 0; u < index_map->width_; ++u) {
            int *idx = geometry::PointerAt<int>(*index_map, u, v);
            if (*idx > 0) {
                auto &color = mesh.vertex_colors_[*idx];
                auto grad_albedo = GetVector3d(*grad_albedo_map, u, v);
                color -= lambda * grad_albedo;
                Clamp(color, 0, 1);

                auto &material = mesh.vertex_materials_[*idx];
                auto grad_material = GetVector3d(*grad_material_map, u, v);
                material -= lambda * grad_material;
                Clamp(material, 0, 1);
            }
        }
    }
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
