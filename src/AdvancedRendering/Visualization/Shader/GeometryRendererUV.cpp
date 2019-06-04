//
// Created by wei on 4/12/19.
//

#include <AdvancedRendering/Visualization/Visualizer/RenderOptionWithLighting.h>
#include <AdvancedRendering/Geometry/TexturedTriangleMesh.h>
#include "GeometryRendererUV.h"

namespace open3d {
namespace visualization {

namespace glsl {
bool GeometryRendererUV::AddGeometry(
    std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::TexturedTriangleMesh) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool GeometryRendererUV::Render(const RenderOption &option,
                                const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->IsEmpty()) return true;

    if (geometry_ptr_->GetGeometryType()
        != geometry::Geometry::GeometryType::TexturedTriangleMesh) {
        utility::PrintWarning("[GeometryRendererUV] "
                              "Geometry type is not TexturedTriangleMesh\n");
        return false;
    }

    const auto &mesh = (const geometry::TexturedTriangleMesh &)
        (*geometry_ptr_);
    auto uv_option = (const RenderOptionWithTargetImage &) option;
    return uv_option.forward_ ?
           uv_forward_shader_.Render(mesh, option, view) :
           uv_backward_shader_.Render(mesh, option, view);
}

bool GeometryRendererUV::UpdateGeometry() {
    uv_forward_shader_.InvalidateGeometry();
    uv_backward_shader_.InvalidateGeometry();

    return true;
}

}
}
}
