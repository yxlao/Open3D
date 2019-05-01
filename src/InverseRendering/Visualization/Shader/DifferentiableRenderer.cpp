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
    success &= ibx_vertex_map_shader_.Render(mesh, textures_, ibl, option, view);

    /** Visualize background **/
    success &= background_shader_.Render(mesh, textures_, ibl, option, view);
    success &= SGD(0.1f);
    RebindGeometry(option, true, true, false);
    return success;
}

inline Eigen::Vector3d GetVector3d(geometry::Image &im, int u, int v) {
    auto * ptr = geometry::PointerAt<float>(im, u, v, 0);
    return Eigen::Vector3d(ptr[0], ptr[1], ptr[2]);
}

inline bool HasNan(Eigen::Vector3d &in) {
    return (std::isnan(in(0)) || std::isnan(in(1)) || std::isnan(in(2)));
}

bool DifferentiableRenderer::SGD(float lambda) {
    std::cout << "SGD\n";
    auto index_map = index_shader_.fbo_outputs_[0];

    auto diff_roughness_map = differential_shader_.fbo_outputs_[1];
    auto diff_albedo_map = differential_shader_.fbo_outputs_[2];
    auto residual_map = differential_shader_.fbo_outputs_[3];

    auto out_render_map = differential_shader_.fbo_outputs_[0];
    auto in_albedo_map = differential_shader_.fbo_outputs_[4];
    auto out_sample_map = differential_shader_.fbo_outputs_[5];
    auto out_diffuse_map = differential_shader_.fbo_outputs_[6];

    auto &mesh = (geometry::TriangleMeshExtended &) *mutable_geometry_ptr_;
    for (int v = 0; v < index_map->height_; ++v) {
        for (int u = 0; u < index_map->width_; ++u) {
            int *idx = geometry::PointerAt<int>(*index_map, u, v);
            if (*idx > 0) {
                auto &color = mesh.vertex_colors_[*idx];

                auto diff_albedo = GetVector3d(*diff_albedo_map, u, v);
                auto diff_roughness = GetVector3d(*diff_roughness_map, u, v);
                auto residual = GetVector3d(*residual_map, u, v);

                mesh.vertex_materials_[*idx](0) -= diff_roughness.dot(residual);
                mesh.vertex_materials_[*idx](0) = std::min(std::max(mesh.vertex_materials_[*idx](0), 0.0), 1.0);

                color -= lambda * diff_albedo.cwiseProduct(residual);
                color(0) = std::min(std::max(color(0), 0.0), 1.0);
                color(1) = std::min(std::max(color(1), 0.0), 1.0);
                color(2) = std::min(std::max(color(2), 0.0), 1.0);

                auto out_render = GetVector3d(*out_render_map, u, v);
                auto in_albedo = GetVector3d(*in_albedo_map, u, v);
                auto out_sample = GetVector3d(*out_sample_map, u, v);
                auto out_diffuse = GetVector3d(*out_diffuse_map, u, v);

                if (HasNan(in_albedo) || HasNan(out_render) || HasNan(out_sample)
                || HasNan(diff_albedo) || HasNan(residual)) {
                    std::cout << *idx << "\n";
                    std::cout << "diff_albedo: " << diff_albedo.transpose() << "\n";
                    std::cout << "residual: " << residual.transpose() << "\n";
                    std::cout << "in_albedo: " << in_albedo.transpose() << "\n";
                    std::cout << "out_render: " << out_render.transpose() << "\n";
                    std::cout << "out_diffuse: " << out_diffuse.transpose() << "\n";
                    std::cout << "out_sample: " << out_sample.transpose() << "\n";
                }
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
