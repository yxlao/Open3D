//
// Created by wei on 3/25/19.
//

#include "VisualizerWithCudaModule.h"
#include <Cuda/Visualization/Shader/GeometryRendererCuda.h>

namespace open3d {
namespace visualization {
bool VisualizerWithCudaModule::AddGeometry(
    std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    bool result = Visualizer::AddGeometry(geometry_ptr);

    if (! result) {
        if (geometry_ptr->GetGeometryType() ==
            geometry::Geometry::GeometryType::TriangleMeshCuda) {
            auto renderer_ptr = std::make_shared<glsl::TriangleMeshCudaRenderer>();
            if (renderer_ptr->AddGeometry(geometry_ptr) == false) {
                return false;
            }
            geometry_renderer_ptrs_.push_back(renderer_ptr);
        } else if (geometry_ptr->GetGeometryType() ==
            geometry::Geometry::GeometryType::PointCloudCuda) {
            auto renderer_ptr = std::make_shared<glsl::PointCloudCudaRenderer>();
            if (renderer_ptr->AddGeometry(geometry_ptr) == false) {
                return false;
            }
            geometry_renderer_ptrs_.push_back(renderer_ptr);
        }
    }

    geometry_ptrs_.push_back(geometry_ptr);
    view_control_ptr_->FitInGeometry(*geometry_ptr);
    ResetViewPoint();
    utility::PrintDebug(
        "Add geometry and update bounding box to %s\n",
        view_control_ptr_->GetBoundingBox().GetPrintInfo().c_str());
    return UpdateGeometry();
}
}
}