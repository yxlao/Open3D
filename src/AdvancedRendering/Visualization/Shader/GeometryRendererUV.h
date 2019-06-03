//
// Created by wei on 4/12/19.
//

#pragma once

#include <Open3D/Open3D.h>

#include "UVShader/UVTexMapShader.h"
#include "UVShader/UVTexAtlasShader.h"

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
    UVTexMapShader uv_tex_map_shader_;
    UVTexAtlasShader uv_tex_atlas_shader_;
};

} // glsl
} // visualization
} // open3d