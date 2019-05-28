//
// Created by wei on 4/12/19.
//

#include "GeometryRendererPBR.h"

namespace open3d {
namespace visualization {

namespace glsl {
bool TriangleMeshRendererPBR::Render(const RenderOption &option,
                                     const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->IsEmpty()) return true;

    if (geometry_ptr_->GetGeometryType()
        != geometry::Geometry::GeometryType::ExtendedTriangleMesh) {
        utility::PrintWarning("[TriangleMeshRendererPBR] "
                              "Geometry type is not ExtendedTriangleMesh\n");
        return false;
    }
    const auto &mesh = (const geometry::ExtendedTriangleMesh &)(*geometry_ptr_);

    bool success = true;

    /** ibl: a bit pre-processing required **/
    if (lighting_ptr_->GetLightingType()
        == geometry::Lighting::LightingType::IBL) {

        auto &ibl = (geometry::IBLLighting &) (*lighting_ptr_);
        if (!ibl.is_preprocessed_) {
            if (!ibl.BindHDRTexture2D()) {
                utility::PrintError("Binding failed when loading light.");
                return false;
            }
            success &= PreprocessLights(ibl, option, view);
        }

        if (mesh.HasUVs()) {
            success &= ibx_tex_map_shader_.Render(
                mesh, textures_, ibl, option, view);
        } else if (mesh.HasMaterials()) {
            success &= ibl_vertex_map_shader_.Render(
                mesh, textures_, ibl, option, view);
        } else {
            success = false;
        }

        success &= background_shader_.Render(
            mesh, textures_, ibl, option, view);
    }

    /* no ibl: simple */
    else if (lighting_ptr_->GetLightingType()
        == geometry::Lighting::LightingType::Spot) {

        const auto &spot = (const geometry::SpotLighting &) (*lighting_ptr_);
        if (mesh.HasVertexNormals() && mesh.HasUVs()) {
            success &= spot_light_shader_.Render(
                mesh, textures_, spot, option, view);
        }
    }

    return success;
}

bool TriangleMeshRendererPBR::UpdateGeometry() {
    ibl_vertex_map_shader_.InvalidateGeometry();
    spot_light_shader_.InvalidateGeometry();

    UnbindLights();

    return true;
}

}
}
}
