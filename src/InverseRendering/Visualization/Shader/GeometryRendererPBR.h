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
    virtual bool AddTextures(const std::vector<geometry::Image> &textures) = 0;
    virtual bool AddLights(const std::shared_ptr<geometry::Lighting> &lighting) = 0;

    virtual bool PreprocessLights(geometry::IBLLighting &ibl,
                                  const RenderOption &option,
                                  const ViewControl &view) = 0;
    virtual bool UnbindLights() = 0;

protected:
    std::vector<geometry::Image> textures_;
    std::shared_ptr<geometry::Lighting> lighting_ptr_ = nullptr;
};

class TriangleMeshRendererPBR : public GeometryRendererPBR {
public:
    ~TriangleMeshRendererPBR() override = default;

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;

    bool AddGeometry(std::shared_ptr<const geometry::Geometry> geometry_ptr) override;
    bool AddTextures(const std::vector<geometry::Image> &textures) override;
    bool AddLights(const std::shared_ptr<geometry::Lighting> &lighting_ptr) override;

    bool PreprocessLights(geometry::IBLLighting &ibl,
                          const RenderOption &option,
                          const ViewControl &view) override;
    bool UnbindLights() override;

    bool UpdateGeometry() override;

protected:
    /** NoIBL: simple **/
    SpotLightShader spot_light_shader_;

    /** IBL: Preprocessing **/
    HDRToEnvCubemapShader hdr_to_env_cubemap_shader_;
    PreConvEnvDiffuseShader preconv_env_diffuse_shader_;
    PreFilterEnvSpecularShader prefilter_env_specular_shader_;
    PreIntegrateLUTSpecularShader preintegrate_lut_specular_shader_;

    /** IBL: w/ and w/o texture maps **/
    IBLTexMapShader ibl_shader_;
    IBLVertexMapShader ibl_no_tex_shader_;

    BackgroundShader background_shader_;
};

} // glsl
} // visualization
} // open3d

