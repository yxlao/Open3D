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
    auto &mesh = (const geometry::TriangleMeshExtended &) (*geometry_ptr_);
    if (! mesh.HasMaterials()) {
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
        success &= hdr_to_env_cubemap_shader_.Render(
            mesh, textures_, ibl, option, view);
        ibl.UpdateEnvBuffer(
            hdr_to_env_cubemap_shader_.GetGeneratedCubemapBuffer());

        success &= preconv_env_diffuse_shader_.Render(
            mesh, textures_, ibl, option, view);
        ibl.UpdateEnvDiffuseBuffer(
            preconv_env_diffuse_shader_.GetGeneratedDiffuseBuffer());

        success &= prefilter_env_specular_shader_.Render(
            mesh, textures_, ibl, option, view);
        ibl.UpdateEnvSpecularBuffer(
            prefilter_env_specular_shader_.GetGeneratedPrefilterEnvBuffer());

        success &= preintegrate_lut_specular_shader_.Render(
            mesh, textures_, ibl, option, view);
        ibl.UpdateLutSpecularBuffer(
            preintegrate_lut_specular_shader_.GetGeneratedLUTBuffer());

        ibl.is_preprocessed_ = true;
    }

    /** Major differential rendering steps **/
    success &= differential_shader_.Render(mesh, textures_, ibl, option, view);
    success &= index_shader_.Render(mesh, textures_, ibl, option, view);
//    fbo_outputs_.clear();
//    fbo_outputs_.insert(fbo_outputs_.begin(),
//                        differential_shader_.fbo_outputs_.begin(),
//                        differential_shader_.fbo_outputs_.end());
//    fbo_outputs_.emplace_back(index_shader_.index_map_);

    /** Visualize object changes **/
    success &= ibl_no_tex_shader_.Render(mesh, textures_, ibl, option, view);

    /** Visualize background **/
    success &= background_shader_.Render(mesh, textures_, ibl, option, view);

    return success;
}

bool DifferentiableRenderer::AddMutableGeometry(
    std::shared_ptr<open3d::geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::TriangleMesh) {
        return false;
    }
    mutable_geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool DifferentiableRenderer::AddGeometry(
    std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::TriangleMesh) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool DifferentiableRenderer::AddTextures(
    const std::vector<geometry::Image> &textures) {
    textures_ = textures;
    return true;
}

bool DifferentiableRenderer::AddLights(
    const std::shared_ptr<geometry::Lighting> &lighting_ptr) {
    lighting_ptr_ = lighting_ptr;
    return true;
}

bool DifferentiableRenderer::UpdateGeometry() {
    differential_shader_.InvalidateGeometry();
    index_shader_.InvalidateGeometry();
    ibl_no_tex_shader_.InvalidateGeometry();
    return true;
}

}
}
}
