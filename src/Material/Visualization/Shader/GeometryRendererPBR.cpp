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
    const auto &mesh = (const geometry::TriangleMeshPhysics &)(*geometry_ptr_);
    const auto &spot = (const physics::SpotLighting &)(*lighting_ptr_);
    bool success = true;
    if (mesh.HasVertexNormals() && mesh.HasUVs()) {
        success &= pbr_no_ibl_shader_.Render(
            mesh, textures_, spot, option, view);
    }
    return success;
}

bool TriangleMeshRendererPBR::AddGeometry(
    std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::TriangleMesh) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool TriangleMeshRendererPBR::AddTextures(
    const std::vector<geometry::Image> &textures) {
    textures_ = textures;
    return true;
}

bool TriangleMeshRendererPBR::AddLights(
    const std::shared_ptr<physics::Lighting> &lighting_ptr) {
    lighting_ptr_ = lighting_ptr;
    return true;
}

bool TriangleMeshRendererPBR::UpdateGeometry() {
    pbr_no_ibl_shader_.InvalidateGeometry();
    return true;
}

}
}
}
