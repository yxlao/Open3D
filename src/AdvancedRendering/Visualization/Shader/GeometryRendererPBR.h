//
// Created by wei on 4/12/19.
//

#pragma once

#include <Open3D/Open3D.h>

#include "AdvancedRendering/Visualization/Shader/UVShader/UVForwardShader.h"
#include "AdvancedRendering/Visualization/Shader/PBRShader/IBLTextureMapShader.h"
#include "PBRShader/IBLVertexMapShader.h"
#include "AdvancedRendering/Visualization/Shader/PBRShader/SpotTextureMapShader.h"
#include "AdvancedRendering/Visualization/Shader/PBRShader/IBLVertexMapMCShader.h"

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
    const int kNumTextures = 5;

    /** NoIBL: simple **/
    SpotTextureMapShader spot_light_shader_;

    /** IBL: w/ and w/o texture maps **/
    IBLTextureMapShader ibx_texure_map_shader_;
    IBLVertexMapShader ibl_vertex_map_shader_;
};

} // glsl
} // visualization
} // open3d