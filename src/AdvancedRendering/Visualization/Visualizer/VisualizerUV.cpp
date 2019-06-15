//
// Created by wei on 4/15/19.
//

#include "VisualizerUV.h"

#include <AdvancedRendering/Geometry/ImageExt.h>
#include <AdvancedRendering/Geometry/TexturedTriangleMesh.h>
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
    render_option_ptr_ =
            std::unique_ptr<RenderOptionAdvanced>(new RenderOptionAdvanced);
    return true;
}

/** Call this function
 * - AFTER @CreateVisualizerWindow and @AddGeometry
 *   :to ensure OpenGL context has been created.
 * - BEFORE @Run (or whatever customized rendering task)
 *   :to ensure target image is ready.
 * Currently we only support one target image.
 *   It would remove the previous bound image.
 * **/
bool VisualizerUV::EnableForwardMode() {
    auto &render_option_advanced =
            (std::shared_ptr<RenderOptionAdvanced> &)render_option_ptr_;
    render_option_advanced->forward_ = true;

    /* Should not be used, just in case of debugging. */
    render_option_advanced->tex_image_width_ =
            view_control_ptr_->GetWindowWidth();
    render_option_advanced->tex_image_height_ =
            view_control_ptr_->GetWindowHeight();
}

bool VisualizerUV::EnableBackwardMode(
        const std::shared_ptr<open3d::geometry::Image> &image) {
    auto &render_option_advanced =
            (std::shared_ptr<RenderOptionAdvanced> &)render_option_ptr_;
    render_option_advanced->forward_ = false;

    assert(!image->IsEmpty());
    auto tex_image = geometry::FlipImageExt(*image);
    if (!render_option_advanced->is_image_tex_allocated_) {
        render_option_advanced->tex_image_buffer_ =
                BindTexture2D(*tex_image, *render_option_advanced);
        render_option_advanced->is_image_tex_allocated_ = true;
    } else { /* Rebind */
        BindTexture2D(render_option_advanced->tex_image_buffer_, *tex_image,
                      *render_option_advanced);
    }

    /** Set texture size in the forward pass **/
    render_option_advanced->tex_image_width_ = image->width_;
    render_option_advanced->tex_image_height_ = image->height_;

    /** Set texture size in the backward pass **/
    assert(!geometry_ptrs_.empty());
    auto mesh_ptr =
            (std::shared_ptr<geometry::TexturedTriangleMesh> &)*geometry_ptrs_
                    .begin();
    assert(!mesh_ptr->texture_images_.empty());
    render_option_advanced->tex_uv_width_ = mesh_ptr->texture_images_[0].width_;
    render_option_advanced->tex_uv_height_ =
            mesh_ptr->texture_images_[0].height_;

    /** Single instance of fbo and its textures */
    if (!render_option_advanced->is_fbo_allocated_) {
        glGenFramebuffers(1, &render_option_advanced->fbo_forward_);
        glGenRenderbuffers(1, &render_option_advanced->rbo_forward_);
        glGenFramebuffers(1, &render_option_advanced->fbo_backward_);
        glGenRenderbuffers(1, &render_option_advanced->rbo_backward_);

        render_option_advanced->is_fbo_allocated_ = true;
    }

    if (!render_option_advanced->is_fbo_tex_allocated_) {
        /* color (forward, only for debugging) */
        render_option_advanced->tex_forward_image_buffer_ = CreateTexture2D(
                render_option_advanced->tex_image_width_,
                render_option_advanced->tex_image_height_, GL_RGB16F, GL_RGB,
                GL_FLOAT, false, *render_option_advanced);

        /* depth (forward, for occlusion test) */
        render_option_advanced->tex_forward_depth_buffer_ = CreateTexture2D(
                render_option_advanced->tex_image_width_,
                render_option_advanced->tex_image_height_, GL_DEPTH_COMPONENT24,
                GL_DEPTH_COMPONENT, GL_FLOAT, false, *render_option_advanced);

        /* color (atlas, read and write)  */
        render_option_advanced->tex_backward_uv_color_buffer_ = CreateTexture2D(
                render_option_advanced->tex_uv_width_,
                render_option_advanced->tex_uv_height_, GL_RGB16F, GL_RGB,
                GL_FLOAT, false, *render_option_advanced);

        /* weight (atlas, read and write.) */
        render_option_advanced->tex_backward_uv_weight_buffer_ =
                CreateTexture2D(render_option_advanced->tex_uv_width_,
                                render_option_advanced->tex_uv_height_,
                                GL_RGB16F, GL_RGB, GL_FLOAT, false,
                                *render_option_advanced);

        sum_color_ = std::make_shared<geometry::Image>();
        sum_color_->PrepareImage(render_option_advanced->tex_uv_width_,
                                 render_option_advanced->tex_uv_height_, 3, 4);

        sum_weight_ = std::make_shared<geometry::Image>();
        sum_weight_->PrepareImage(render_option_advanced->tex_uv_width_,
                                  render_option_advanced->tex_uv_height_, 3, 4);

        for (int v = 0; v < render_option_advanced->tex_uv_height_; ++v) {
            for (int u = 0; u < render_option_advanced->tex_uv_width_; ++u) {
                for (int c = 0; c < 3; ++c) {
                    *geometry::PointerAt<float>(*sum_color_, u, v, c) = 0;
                    *geometry::PointerAt<float>(*sum_weight_, u, v, c) = 0;
                }
            }
        }

        render_option_advanced->is_fbo_tex_allocated_ = true;
    }

    return true;
}

}  // namespace visualization
}  // namespace open3d
