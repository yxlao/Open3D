//
// Created by wei on 4/12/19.
//

#include <AdvancedRendering/Visualization/Visualizer/RenderOptionAdvanced.h>
#include <AdvancedRendering/Geometry/TexturedTriangleMesh.h>
#include "GeometryRendererPBR.h"

namespace open3d {
namespace visualization {

namespace glsl {
bool GeometryRendererPBR::AddGeometry(
    std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::ExtendedTriangleMesh
        && geometry_ptr->GetGeometryType() !=
            geometry::Geometry::GeometryType::TexturedTriangleMesh) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool GeometryRendererPBR::Render(const RenderOption &option,
                                 const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->IsEmpty()) return true;
    auto &option_lighting = (const RenderOptionAdvanced &) option;

    /** Vertex material mapping **/
    if (geometry_ptr_->GetGeometryType()
        == geometry::Geometry::GeometryType::ExtendedTriangleMesh) {
        auto &mesh = (geometry::ExtendedTriangleMesh &) *geometry_ptr_;

        if (option_lighting.type_
            == geometry::Lighting::LightingType::IBL) {
            return mesh.HasVertexTextures() ?
                   ibl_vertex_map_shader_.Render(mesh, option, view) :
                   false;
        } // spotlight not yet supported
    }

    /** Texture material mapping **/
    else if (geometry_ptr_->GetGeometryType()
        == geometry::Geometry::GeometryType::TexturedTriangleMesh) {
        auto &mesh = (geometry::TexturedTriangleMesh &) *geometry_ptr_;

        if (option_lighting.type_
            == geometry::Lighting::LightingType::IBL) {
            return (mesh.HasUVs() && mesh.HasTextureImages(kNumTextures)) ?
                   ibx_texure_map_shader_.Render(mesh, option, view) :
                   false;
        } else if (option_lighting.type_
            == geometry::Lighting::LightingType::Spot) {
            return (mesh.HasUVs() && mesh.HasTextureImages(kNumTextures)) ?
                   spot_light_shader_.Render(mesh, option, view) :
                   false;
        }
    } else {
        utility::PrintWarning("[GeometryRendererPBR] "
                              "Geometry type is not supported\n");
        return false;
    }

    return false;
}

bool GeometryRendererPBR::UpdateGeometry() {
    ibl_vertex_map_shader_.InvalidateGeometry();
    spot_light_shader_.InvalidateGeometry();

    return true;
}

}
}
}
