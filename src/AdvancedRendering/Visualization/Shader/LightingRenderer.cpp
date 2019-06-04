//
// Created by Wei Dong on 2019-05-28.
//

#include "LightingRenderer.h"
#include "../Utility/BufferHelper.h"

namespace open3d {
namespace visualization {
namespace glsl {

bool LightingRenderer::AddGeometry(
    std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::Lighting) {
        return false;
    }

    /** Clear memory anyway:
     * - IBL: re-generate textures,
     * - non-IBL: we dont need buffers **/
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

/** Handle the textures all in the renderer:
 *  separating textures in the shaders will be hard to manage.
 **/
bool LightingRenderer::UpdateGeometry() {
    if (is_tex_allocated_) {
        glDeleteBuffers(1, &tex_hdr_buffer_);
        glDeleteBuffers(1, &tex_env_buffer_);
        glDeleteBuffers(1, &tex_env_diffuse_buffer_);
        glDeleteBuffers(1, &tex_env_specular_buffer_);
        glDeleteBuffers(1, &tex_lut_specular_buffer_);
        is_tex_allocated_ = false;
    }
    return true;
}

bool LightingRenderer::Render(const RenderOption &option,
                              const ViewControl &view) {
    auto pbr_option = (const RenderOptionWithLighting &) option;
    if (pbr_option.type_ == geometry::Lighting::LightingType::IBL) {
        return background_shader_.Render(*geometry_ptr_, option, view);
    } else {
        return true;
    }
}

/** Call this function ONLY AFTER @AddGeometry(lighting)
 * Here we use non-constant option, in contrast to default @Render
 **/
bool LightingRenderer::RenderToOption(RenderOptionWithLighting &option,
                                      const ViewControl &view) {

    auto lighting_ptr = (std::shared_ptr<geometry::Lighting> &) geometry_ptr_;

    if (lighting_ptr->GetLightingType()
        == geometry::Lighting::LightingType::Spot) {
        option.type_ = geometry::Lighting::LightingType::Spot;

        auto &spot_lighting_ptr =
            (std::shared_ptr<geometry::SpotLighting> &) lighting_ptr;
        option.spot_light_positions_ = spot_lighting_ptr->light_positions_;
        option.spot_light_colors_ = spot_lighting_ptr->light_colors_;

        return true;
    } else if (lighting_ptr->GetLightingType()
        == geometry::Lighting::LightingType::IBL) {
        option.type_ = geometry::Lighting::LightingType::IBL;

        bool success = true;

        auto &ibl = (std::shared_ptr<geometry::IBLLighting> &) lighting_ptr;
        option.tex_hdr_buffer_
            = tex_hdr_buffer_
            = BindTexture2D(*(ibl->hdr_), option);

        success &= hdr_to_env_cubemap_shader_.Render(
            *geometry_ptr_, option, view);
        option.tex_env_buffer_
            = tex_env_buffer_
            = hdr_to_env_cubemap_shader_.GetGeneratedCubemapBuffer();

        success &= preconv_env_diffuse_shader_.Render(
            *geometry_ptr_, option, view);
        option.tex_env_diffuse_buffer_
            = tex_env_diffuse_buffer_
            = preconv_env_diffuse_shader_.GetGeneratedDiffuseBuffer();

        success &= prefilter_env_specular_shader_.Render(
            *geometry_ptr_, option, view);
        option.tex_env_specular_buffer_
            = tex_env_specular_buffer_
            = prefilter_env_specular_shader_.GetGeneratedPrefilterEnvBuffer();

        success &= preintegrate_lut_specular_shader_.Render(
            *geometry_ptr_, option, view);
        option.tex_lut_specular_buffer_
            = tex_lut_specular_buffer_
            = preintegrate_lut_specular_shader_.GetGeneratedLUTBuffer();

        is_tex_allocated_ = success;
        return success;
    }

    return false;
}
} // namespace glsl
} // namespace visualization
} // namespace open3d