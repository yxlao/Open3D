//
// Created by wei on 10/31/18.
//

#include "GeometryRendererCuda.h"

namespace open3d {

namespace glsl {
bool TriangleMeshCudaRenderer::Render(const RenderOption &option,
                                      const ViewControl &view) {
    if (is_visible_ == false || geometry_ptr_->IsEmpty()) return true;
    const auto &mesh = (const TriangleMesh &) (*geometry_ptr_);
    bool success = true;
    success &= normal_mesh_shader_.Render(mesh, option, view);
    return success;
}

bool TriangleMeshCudaRenderer::AddGeometry(
    std::shared_ptr<const Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        Geometry::GeometryType::TriangleMesh) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool TriangleMeshCudaRenderer::UpdateGeometry() {
    normal_mesh_shader_.InvalidateGeometry();
    return true;
}
}
}