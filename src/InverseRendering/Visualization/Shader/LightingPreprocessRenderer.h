//
// Created by Wei Dong on 2019-05-28.
//

#pragma once

#include <Open3D/Open3D.h>
#include <InverseRendering/Visualization/Visualizer/RenderOptionWithLighting.h>
#include <InverseRendering/Geometry/Lighting.h>

#include "LightingShader/HDRToEnvCubemapShader.h"
#include "LightingShader/BackgroundShader.h"
#include "LightingShader/PreConvEnvDiffuseShader.h"
#include "LightingShader/PreFilterEnvSpecularShader.h"
#include "LightingShader/PreIntegrateLUTSpecularShader.h"

namespace open3d {
namespace visualization {
namespace glsl {

class LightingPreprocessRenderer : public GeometryRenderer {
public:
    ~LightingPreprocessRenderer() override = default;

public:
    bool AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr)
    override { return true; }
    bool UpdateGeometry() { return true; }
    bool Render(const RenderOption &option, const ViewControl &view)
    override { return true; }

    bool UnbindTextures() {
        if (is_tex_allocated_) {
            glDeleteBuffers(1, &tex_hdr_buffer_);
            glDeleteBuffers(1, &tex_env_buffer_);
            glDeleteBuffers(1, &tex_env_diffuse_buffer_);
            glDeleteBuffers(1, &tex_env_specular_buffer_);
            glDeleteBuffers(1, &tex_lut_specular_buffer_);
            is_tex_allocated_ = false;
        }
    }

    /** Here we use non-constant option **/
    bool RenderToOption(RenderOption &option, const ViewControl &view) {
        bool success = true;
        auto dummy = std::make_shared<geometry::TriangleMesh>();

        UnbindTextures();

        auto &option_lighting = (RenderOptionWithLighting &) option;
        if (option_lighting.lighting_ptr_->GetLightingType()
            != geometry::Lighting::LightingType::IBL) {
            utility::PrintWarning("Invalid lighting, IBL expected.\n");
            return false;
        }

        auto &ibl = (std::shared_ptr<geometry::IBLLighting> &)
            option_lighting.lighting_ptr_;
        BindHDRTexture2D(ibl);
        option_lighting.tex_hdr_buffer_ = tex_hdr_buffer_;

        success &= hdr_to_env_cubemap_shader_.Render(
            *dummy, option, view);
        option_lighting.tex_env_buffer_
            = tex_env_buffer_
            = hdr_to_env_cubemap_shader_.GetGeneratedCubemapBuffer();

        success &= preconv_env_diffuse_shader_.Render(
            *dummy, option, view);
        option_lighting.tex_env_diffuse_buffer_
            = tex_env_diffuse_buffer_
            = preconv_env_diffuse_shader_.GetGeneratedDiffuseBuffer();

        success &= prefilter_env_specular_shader_.Render(
            *dummy, option, view);
        option_lighting.tex_env_specular_buffer_
            = tex_env_specular_buffer_
            = prefilter_env_specular_shader_.GetGeneratedPrefilterEnvBuffer();

        success &= preintegrate_lut_specular_shader_.Render(
            *dummy, option, view);
        option_lighting.tex_lut_specular_buffer_
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
    std::shared_ptr<geometry::Lighting> lighting_ptr_ = nullptr;

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
