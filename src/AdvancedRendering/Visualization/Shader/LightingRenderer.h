//
// Created by Wei Dong on 2019-05-28.
//

#pragma once

#include <Open3D/Open3D.h>
#include <AdvancedRendering/Visualization/Visualizer/RenderOptionWithLighting.h>
#include <AdvancedRendering/Geometry/Lighting.h>
#include <Open3D/Geometry/Geometry.h>

#include "LightingShader/HDRToEnvCubemapShader.h"
#include "LightingShader/BackgroundShader.h"
#include "LightingShader/PreConvEnvDiffuseShader.h"
#include "LightingShader/PreFilterEnvSpecularShader.h"
#include "LightingShader/PreIntegrateLUTSpecularShader.h"

namespace open3d {
namespace visualization {
namespace glsl {

class LightingRenderer : public GeometryRenderer {
public:
    ~LightingRenderer() override = default;

public:
    bool AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr)
    override {
        /** Change nothing **/
        if (geometry_ptr->GetGeometryType() !=
            geometry::Geometry::GeometryType::Lighting) {
            utility::PrintError(
                "Fail to bind geometry to LightingPreprocessRenderer.\n");
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
    bool UpdateGeometry() override {
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

    /** Render Nothing (TODO: or only the skybox)? **/
    bool Render(const RenderOption &option, const ViewControl &view)
    override { return true; }

    /** Call this function ONLY AFTER @AddGeometry(lighting)
     * Here we use non-constant option, in contrast to default @Render
     **/
    bool RenderToOption(RenderOptionWithLighting &option,
                        const ViewControl &view) {
        bool success = true;

        auto lighting_ptr = (std::shared_ptr<geometry::Lighting> &)
            geometry_ptr_;

        if (lighting_ptr->GetLightingType()
            == geometry::Lighting::LightingType::Spot) {
            utility::PrintDebug("Pass non-IBL lighting.\n");
            option.type_ = geometry::Lighting::LightingType::Spot;

            auto &spot_lighting_ptr =
                (std::shared_ptr<geometry::SpotLighting> &) lighting_ptr;
            option.spot_light_positions_ = spot_lighting_ptr->light_positions_;
            option.spot_light_colors_ = spot_lighting_ptr->light_colors_;
            return true;
        }

        option.type_ = geometry::Lighting::LightingType::IBL;
        auto &ibl = (std::shared_ptr<geometry::IBLLighting> &) lighting_ptr;
        BindHDRTexture2D(ibl);
        option.tex_hdr_buffer_ = tex_hdr_buffer_;

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

    bool BindHDRTexture2D(std::shared_ptr<geometry::IBLLighting> &ibl_ptr) {
        glGenTextures(1, &tex_hdr_buffer_);

        utility::PrintInfo("Binding HDR Texture\n");
        glBindTexture(GL_TEXTURE_2D, tex_hdr_buffer_);
        glTexImage2D(GL_TEXTURE_2D,
                     0, GL_RGB16F,
                     ibl_ptr->hdr_->width_,
                     ibl_ptr->hdr_->height_,
                     0, GL_RGB, GL_FLOAT,
                     ibl_ptr->hdr_->data_.data());

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        return true;
    }

public:
    bool is_tex_allocated_ = false;
    GLuint tex_hdr_buffer_;
    GLuint tex_env_buffer_;
    GLuint tex_env_diffuse_buffer_;
    GLuint tex_env_specular_buffer_; /* roughness, R = 2<V, N>N - V */
    GLuint tex_lut_specular_buffer_; /* roughness, <V, N>) */

    /** IBL preprocessors **/
    HDRToEnvCubemapShader hdr_to_env_cubemap_shader_;
    PreConvEnvDiffuseShader preconv_env_diffuse_shader_;
    PreFilterEnvSpecularShader prefilter_env_specular_shader_;
    PreIntegrateLUTSpecularShader preintegrate_lut_specular_shader_;
};

}
}
}
