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
        std::shared_ptr<const geometry::Geometry> geometry_ptr) override;

    /** Handle the textures all in the renderer:
     *  separating textures in the shaders will be hard to manage.
     **/
    bool UpdateGeometry() override;

    bool Render(const RenderOption &option, const ViewControl &view) override;

    /** Call this function ONLY AFTER @AddGeometry(lighting)
     * Here we use non-constant option, in contrast to default @Render
     **/
    bool RenderToOption(RenderOptionWithLighting &option,
                        const ViewControl &view);

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

    /** Skybox renderer **/
    BackgroundShader background_shader_;
};

}
}
}
