//
// Created by wei on 4/12/19.
//

#include <InverseRendering/Visualization/Visualizer/RenderOptionWithLighting.h>
#include "GeometryRendererPBR.h"

namespace open3d {
namespace visualization {

namespace glsl {
bool GeometryRendererPBR::AddGeometry(
    std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::ExtendedTriangleMesh) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool GeometryRendererPBR::Render(const RenderOption &option,
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

    /** ibl: a bit pre-processing required **/
    auto &option_lighting = (const RenderOptionWithLighting &) option;
    if (option_lighting.type_ == geometry::Lighting::LightingType::IBL) {

        if (mesh.HasUVs() && mesh.HasImageTextures()) {
            success &= ibx_tex_map_shader_.Render(
                mesh, option, view);
        } else if (mesh.HasVertexTextures()) {
            success &= ibl_vertex_map_shader_.Render(
                mesh, option, view);
        } else {
            success = false;
        }

        success &= background_shader_.Render(
            mesh, option, view);
    } else if (option_lighting.type_
        == geometry::Lighting::LightingType::Spot) {
        if (mesh.HasVertexNormals()
            && mesh.HasUVs() && mesh.HasImageTextures()) {
            success &= spot_light_shader_.Render(
                mesh, option, view);
        }
    }

    return success;
}

bool GeometryRendererPBR::UpdateGeometry() {
    ibl_vertex_map_shader_.InvalidateGeometry();
    spot_light_shader_.InvalidateGeometry();

    return true;
}

}
}
}
