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

void VisualizerPBR::Render() {
    glfwMakeContextCurrent(window_);

    view_control_ptr_->SetViewMatrices();

    glEnable(GL_MULTISAMPLE);
    glDisable(GL_BLEND);
    auto &background_color = render_option_ptr_->background_color_;
    glClearColor((GLclampf)background_color(0), (GLclampf)background_color(1),
                 (GLclampf)background_color(2), 1.0f);
    glClearDepth(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


    for (const auto &renderer_ptr : geometry_renderer_ptrs_) {
        renderer_ptr->Render(*render_option_ptr_, *view_control_ptr_);
    }
    for (const auto &renderer_ptr : utility_renderer_ptrs_) {
        renderer_ptr->Render(*render_option_ptr_, *view_control_ptr_);
    }
    light_renderer_ptr_->Render(*render_option_ptr_, *view_control_ptr_);

    glfwSwapBuffers(window_);
}
}  // namespace visualization
}  // namespace open3d
