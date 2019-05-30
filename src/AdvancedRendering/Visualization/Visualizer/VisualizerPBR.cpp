//
// Created by wei on 4/15/19.
//

#include "VisualizerPBR.h"

#include <AdvancedRendering/Geometry/ImageExt.h>
#include <AdvancedRendering/Visualization/Shader/DifferentiableRenderer.h>
#include <AdvancedRendering/Visualization/Shader/GeometryRendererPBR.h>

namespace open3d {
namespace visualization {

bool VisualizerPBR::AddGeometry(
    std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() ==
        geometry::Geometry::GeometryType::ExtendedTriangleMesh) {
        auto renderer_ptr =
            std::make_shared<glsl::GeometryRendererPBR>();
        if (!renderer_ptr->AddGeometry(geometry_ptr)) {
            utility::PrintDebug("Failed to add geometry\n");
            return false;
        }
        geometry_renderer_ptrs_.emplace(renderer_ptr);
    }

    geometry_ptrs_.emplace(geometry_ptr);

    view_control_ptr_->FitInGeometry(*geometry_ptr);
    ResetViewPoint();
    utility::PrintDebug(
        "Add geometry and update bounding box to %s\n",
        view_control_ptr_->GetBoundingBox().GetPrintInfo().c_str());
    return UpdateGeometry();
}
}  // namespace visualization
}  // namespace open3d
