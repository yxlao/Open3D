//
// Created by wei on 4/12/19.
//

#include <AdvancedRendering/Visualization/Visualizer/RenderOptionWithLighting.h>
#include "GeometryRendererUV.h"

namespace open3d {
namespace visualization {

namespace glsl {
bool GeometryRendererUV::AddGeometry(
    std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::ExtendedTriangleMesh) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool GeometryRendererUV::Render(const RenderOption &option,
                                 const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->IsEmpty()) return true;

    if (geometry_ptr_->GetGeometryType()
        != geometry::Geometry::GeometryType::ExtendedTriangleMesh) {
        utility::PrintWarning("[TriangleMeshRendererPBR] "
                              "Geometry type is not ExtendedTriangleMesh\n");
        return false;
    }
    const auto
        &mesh = (const geometry::ExtendedTriangleMesh &) (*geometry_ptr_);

    bool success = true;

    success &= uv_tex_atlas_shader_.Render(mesh, option, view);
    /** ibl: a bit pre-processing required **/

    return success;
}

bool GeometryRendererUV::UpdateGeometry() {
//    uv_tex_atlas_shader_.InvalidateGeometry();

    return true;
}

}
}
}
