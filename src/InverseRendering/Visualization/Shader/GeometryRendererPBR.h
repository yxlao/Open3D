//
// Created by wei on 4/12/19.
//

#pragma once

#include <Open3D/Open3D.h>
#include <InverseRendering/Visualization/Shader/PBRShader/IBLVertexMapShader.h>

#include "InverseRendering/Visualization/Shader/PBRShader/SpotLightShader.h"

#include "InverseRendering/Visualization/Shader/LightingShader/HDRToEnvCubemapShader.h"
#include "InverseRendering/Visualization/Shader/LightingShader/BackgroundShader.h"
#include "InverseRendering/Visualization/Shader/LightingShader/PreConvEnvDiffuseShader.h"
#include "InverseRendering/Visualization/Shader/LightingShader/PreFilterEnvSpecularShader.h"
#include "InverseRendering/Visualization/Shader/LightingShader/PreIntegrateLUTSpecularShader.h"
#include "InverseRendering/Visualization/Shader/PBRShader/IBLTexMapShader.h"
#include "InverseRendering/Visualization/Shader/DifferentiableShader/DifferentialShader.h"
#include "InverseRendering/Visualization/Shader/DifferentiableShader/IndexShader.h"
#include "InverseRendering/Visualization/Shader/PBRShader/DirectSamplingShader.h"

namespace open3d {
namespace visualization {
namespace glsl {

class GeometryRendererPBR : public GeometryRenderer {
public:
    ~GeometryRendererPBR() override = default;

public:
    bool AddGeometry(std::shared_ptr<const geometry::Geometry> geometry_ptr)
    override {
        if (geometry_ptr->GetGeometryType() !=
            geometry::Geometry::GeometryType::TriangleMesh) {
            return false;
        }
        geometry_ptr_ = geometry_ptr;
        return UpdateGeometry();
    }

    virtual bool AddTextures(const std::vector<geometry::Image> &textures) {
        textures_ = textures;
        return true;
    }

    virtual bool AddLights(const std::shared_ptr<geometry::Lighting> &lighting) {
        lighting_ptr_ = lighting;
        return true;
    }

    virtual bool UnbindLights() {
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
            }
        }
	return true;
    }

    virtual bool PreprocessLights(geometry::IBLLighting &ibl,
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

protected:
    std::vector<geometry::Image> textures_;
    std::shared_ptr<geometry::Lighting> lighting_ptr_ = nullptr;

    /** IBL preprocessors **/
    HDRToEnvCubemapShader hdr_to_env_cubemap_shader_;
    PreConvEnvDiffuseShader preconv_env_diffuse_shader_;
    PreFilterEnvSpecularShader prefilter_env_specular_shader_;
    PreIntegrateLUTSpecularShader preintegrate_lut_specular_shader_;
};


class TriangleMeshRendererPBR : public GeometryRendererPBR {
public:
    ~TriangleMeshRendererPBR() override = default;

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool UpdateGeometry() override;

protected:
    /** NoIBL: simple **/
    SpotLightShader spot_light_shader_;

    /** IBL: w/ and w/o texture maps **/
    IBLTexMapShader ibx_tex_map_shader_;
    IBLVertexMapShader ibl_vertex_map_shader_;
    BackgroundShader background_shader_;
};

} // glsl
} // visualization
} // open3d

