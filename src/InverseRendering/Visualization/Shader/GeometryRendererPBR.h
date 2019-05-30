//
// Created by wei on 4/12/19.
//

#pragma once

#include <Open3D/Open3D.h>

#include "PBRShader/IBLTexMapShader.h"
#include "PBRShader/IBLVertexMapShader.h"
#include "PBRShader/SpotLightShader.h"
#include "PBRShader/DirectSamplingShader.h"

#include "LightingShader/BackgroundShader.h"

namespace open3d {
namespace visualization {
namespace glsl {

class GeometryRendererPBR : public GeometryRenderer {
public:
    ~GeometryRendererPBR() override = default;

public:
    bool AddGeometry(std::shared_ptr<const geometry::Geometry> geometry_ptr)
    override;
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool UpdateGeometry() override;

protected:
    /** NoIBL: simple **/
    SpotLightShader spot_light_shader_;

    /** IBL: w/ and w/o texture maps **/
    IBLTexMapShader ibx_tex_map_shader_;
    IBLVertexMapShader ibl_vertex_map_shader_;

    /** IBL: display skybox **/
    BackgroundShader background_shader_;
};

} // glsl
} // visualization
} // open3d