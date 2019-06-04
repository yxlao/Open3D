//
// Created by wei on 4/15/19.
//

#include "VisualizerUV.h"

#include <AdvancedRendering/Geometry/ImageExt.h>
#include <AdvancedRendering/Visualization/Shader/DifferentiableRenderer.h>
#include <AdvancedRendering/Visualization/Shader/GeometryRendererUV.h>

namespace open3d {
namespace visualization {

bool VisualizerUV::AddGeometry(
    std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() ==
        geometry::Geometry::GeometryType::TexturedTriangleMesh) {
        auto renderer_ptr = std::make_shared<glsl::GeometryRendererUV>();
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

bool VisualizerUV::InitRenderOption() {
    render_option_ptr_ = std::unique_ptr<RenderOptionAdvanced>(
        new RenderOptionAdvanced);
    return true;
}

/** Call this function
 * - AFTER @CreateVisualizerWindow
 *   :to ensure OpenGL context has been created.
 * - BEFORE @Run (or whatever customized rendering task)
 *   :to ensure target image is ready.
 * Currently we only support one target image.
 *   It would remove the previous bound image.
 * **/
bool VisualizerUV::SetupMode(
    bool forward, const std::shared_ptr<geometry::Image> &image) {
    auto &render_option_with_target =
        (std::shared_ptr<RenderOptionAdvanced> &) render_option_ptr_;

    if (forward) {
        render_option_with_target->forward_ = true;
        return true;
    } else {
        render_option_with_target->forward_ = false;
        assert(image != nullptr);
        auto tex_image = geometry::FlipImageExt(*image);

        /** Single instance of the texture buffer **/
        if (!render_option_with_target->is_ref_tex_allocated_) {
            render_option_with_target->tex_ref_buffer_
                = glsl::BindTexture2D(*tex_image, *render_option_with_target);
            render_option_with_target->is_ref_tex_allocated_ = true;
        } else {
            glsl::BindTexture2D(render_option_with_target->tex_ref_buffer_,
                                *tex_image, *render_option_with_target);
        }
        return true;
    }

    return false;
}

}  // namespace visualization
}  // namespace open3d
