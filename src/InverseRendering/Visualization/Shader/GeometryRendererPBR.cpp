//
// Created by wei on 4/12/19.
//

#include "GeometryRendererPBR.h"

namespace open3d {
namespace visualization {

namespace glsl {
bool TriangleMeshRendererPBR::Render(const RenderOption &option,
                                     const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->IsEmpty()) return true;

    if (geometry_ptr_->GetGeometryType()
        != geometry::Geometry::GeometryType::TriangleMesh) {
        utility::PrintWarning("[TriangleMeshRendererPBR] "
                              "Geometry type is not TriangleMesh\n");
        return false;
    }
    const auto &mesh = (const geometry::TriangleMeshExtended &)(*geometry_ptr_);

    bool success = true;

    /** ibl: a bit pre-processing required **/
    if (lighting_ptr_->GetLightingType()
        == geometry::Lighting::LightingType::IBL) {

        auto &ibl = (geometry::IBLLighting &) (*lighting_ptr_);
        if (! ibl.is_preprocessed_) {
            success &= PreprocessLights(ibl, option, view);
        }

        if (mesh.HasUVs()) {
            success &= ibl_shader_.Render(mesh, textures_, ibl, option, view);
        } else if (mesh.HasMaterials()) {
            success &= ibl_no_tex_shader_.Render(
                mesh, textures_, ibl, option, view);
        } else {
            success = false;
        }

        success &= background_shader_.Render(
            mesh, textures_, ibl, option, view);
    }

    /* no ibl: simple */
    else if (lighting_ptr_->GetLightingType()
        == geometry::Lighting::LightingType::Spot) {

        const auto &spot = (const geometry::SpotLighting &) (*lighting_ptr_);
        if (mesh.HasVertexNormals() && mesh.HasUVs()) {
            success &= spot_light_shader_.Render(
                mesh, textures_, spot, option, view);
        }
    }

    return success;
}

bool TriangleMeshRendererPBR::AddGeometry(
    std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::TriangleMesh) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool TriangleMeshRendererPBR::AddTextures(
    const std::vector<geometry::Image> &textures) {
    textures_ = textures;
    return true;
}

bool TriangleMeshRendererPBR::AddLights(
    const std::shared_ptr<geometry::Lighting> &lighting_ptr) {
    lighting_ptr_ = lighting_ptr;
    return true;
}

bool TriangleMeshRendererPBR::PreprocessLights(
    geometry::IBLLighting &ibl,
    const RenderOption &option,
    const ViewControl &view) {

    bool success = true;

    auto dummy = std::make_shared<geometry::TriangleMesh>();
    success &= hdr_to_env_cubemap_shader_.Render(
        *dummy, textures_, ibl, option, view);
    ibl.UpdateEnvBuffer(
        hdr_to_env_cubemap_shader_.GetGeneratedCubemapBuffer());

    success &= preconv_env_diffuse_shader_.Render(
        *dummy, textures_, ibl, option, view);
    ibl.UpdateEnvDiffuseBuffer(
        preconv_env_diffuse_shader_.GetGeneratedDiffuseBuffer());

    success &= prefilter_env_specular_shader_.Render(
        *dummy, textures_, ibl, option, view);
    ibl.UpdateEnvSpecularBuffer(
        prefilter_env_specular_shader_.GetGeneratedPrefilterEnvBuffer());

    success &= preintegrate_lut_specular_shader_.Render(
        *dummy, textures_, ibl, option, view);
    ibl.UpdateLutSpecularBuffer(
        preintegrate_lut_specular_shader_.GetGeneratedLUTBuffer());

    ibl.is_preprocessed_ = true;

    return success;
}

bool TriangleMeshRendererPBR::UnbindLights() {
    if (lighting_ptr_ != nullptr
        && lighting_ptr_->GetLightingType()
            == geometry::Lighting::LightingType::IBL) {
        auto &ibl = (geometry::IBLLighting &) (*lighting_ptr_);

        if (ibl.is_preprocessed_) {
            glDeleteBuffers(1, &ibl.tex_hdr_buffer_);
            glDeleteBuffers(1, &ibl.tex_env_buffer_);
            glDeleteBuffers(1, &ibl.tex_env_diffuse_buffer_);
            glDeleteBuffers(1, &ibl.tex_env_specular_buffer_);
            glDeleteBuffers(1, &ibl.tex_lut_specular_buffer_);

            ibl.is_preprocessed_ = false;
        };
    }
}

bool TriangleMeshRendererPBR::UpdateGeometry() {
    ibl_no_tex_shader_.InvalidateGeometry();
    spot_light_shader_.InvalidateGeometry();

    UnbindLights();

    return true;
}

}
}
}
