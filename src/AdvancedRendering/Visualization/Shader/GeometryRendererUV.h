//
// Created by wei on 4/12/19.
//

#pragma once

#include <Open3D/Open3D.h>
#include <AdvancedRendering/Visualization/Shader/UVShader/SimpleTextureShader.h>

#include "AdvancedRendering/Visualization/Shader/UVShader/UVForwardShader.h"
#include "AdvancedRendering/Visualization/Shader/UVShader/UVBackwardShader.h"

namespace open3d {
namespace visualization {
namespace glsl {

class GeometryRendererUV : public GeometryRenderer {
public:
    ~GeometryRendererUV() override = default;

public:
    bool AddGeometry(std::shared_ptr<const geometry::Geometry> geometry_ptr)
    override;
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool UpdateGeometry() override;

protected:
    UVForwardShader uv_forward_shader_;
    UVBackwardShader uv_backward_shader_;
    SimpleTextureShader simple_texture_shader_;
};

} // glsl
} // visualization
} // open3d