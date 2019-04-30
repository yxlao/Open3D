//
// Created by wei on 4/12/19.
//

#pragma once

#include <Open3D/Open3D.h>
#include "GeometryRendererPBR.h"

namespace open3d {
namespace visualization {
namespace glsl {

class DifferentiableRenderer : public GeometryRendererPBR {
public:
    ~DifferentiableRenderer () override = default;

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;

    bool AddMutableGeometry(std::shared_ptr<geometry::Geometry> geometry_ptr);

    bool AddGeometry(std::shared_ptr<const geometry::Geometry> geometry_ptr) override;
    bool AddTextures(const std::vector<geometry::Image> &textures) override;
    bool AddLights(const std::shared_ptr<geometry::Lighting> &lighting_ptr) override;

    bool UpdateGeometry() override;

protected:
    /** Preprocess illumination **/
    HDRToEnvCubemapShader hdr_to_env_cubemap_shader_;
    PreConvEnvDiffuseShader preconv_env_diffuse_shader_;
    PreFilterEnvSpecularShader prefilter_env_specular_shader_;
    PreIntegrateLUTSpecularShader preintegrate_lut_specular_shader_;

    BackgroundShader background_shader_;

    IBLVertexMapShader ibl_no_tex_shader_;
    DifferentialShader differential_shader_;
    IndexShader index_shader_;

private:
    std::shared_ptr<geometry::Geometry> mutable_geometry_ptr_;
};

} // glsl
} // visualization
} // open3d

