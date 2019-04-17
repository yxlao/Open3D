//
// Created by wei on 4/12/19.
//

#pragma once

#include <Open3D/Open3D.h>

#include "NoIBLShader.h"

#include "HDRToCubemapShader.h"
#include "BackgroundShader.h"
#include "PreConvDiffuseShader.h"
#include "PreFilterEnvShader.h"
#include "PreIntegrateLUTShader.h"
#include "IBLShader.h"

namespace open3d {
namespace visualization {
namespace glsl {

class GeometryRendererPBR : public GeometryRenderer {
public:
    ~GeometryRendererPBR() override = default;

public:
    virtual bool AddTextures(const std::vector<geometry::Image> &textures) = 0;
    virtual bool AddLights(const std::shared_ptr<physics::Lighting> &lighting) = 0;

protected:
    std::vector<geometry::Image> textures_;
    std::shared_ptr<physics::Lighting> lighting_ptr_;
};

class TriangleMeshRendererPBR : public GeometryRendererPBR {
public:
    ~TriangleMeshRendererPBR() override = default;

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;

    bool AddGeometry(std::shared_ptr<const geometry::Geometry> geometry_ptr) override;
    bool AddTextures(const std::vector<geometry::Image> &textures) override;
    bool AddLights(const std::shared_ptr<physics::Lighting> &lighting_ptr) override;

    bool UpdateGeometry() override;

protected:
    /** NoIBL: simple **/
    NoIBLShader no_ibl_shader_;

    /** IBL **/
    HDRToCubemapShader hdr_to_cubemap_shader_;
    PreConvDiffuseShader pre_conv_diffuse_shader_;
    PreFilterEnvShader pre_filter_env_shader_;
    PreIntegrateLUTShader pre_integrate_lut_shader_;

    IBLShader ibl_shader_;
    BackgroundShader background_shader_;

};

} // glsl
} // visualization
} // open3d

