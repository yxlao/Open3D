//
// Created by wei on 4/12/19.
//

#include <AdvancedRendering/Visualization/Visualizer/RenderOptionAdvanced.h>
#include <AdvancedRendering/Geometry/TexturedTriangleMesh.h>
#include "GeometryRendererUV.h"

namespace open3d {
namespace visualization {

namespace glsl {
bool GeometryRendererUV::AddGeometry(
    std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::TexturedTriangleMesh) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool GeometryRendererUV::Render(const RenderOption &option,
                                const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->IsEmpty()) return true;

    if (geometry_ptr_->GetGeometryType()
        != geometry::Geometry::GeometryType::TexturedTriangleMesh) {
        utility::PrintWarning("[GeometryRendererUV] "
                              "Geometry type is not TexturedTriangleMesh\n");
        return false;
    }

    const auto &mesh = (const geometry::TexturedTriangleMesh &)
        (*geometry_ptr_);

    auto &uv_option = (RenderOptionAdvanced &) option;

    if (uv_option.forward_) {
        uv_option.render_to_fbo_ = false;
        uv_forward_shader_.Render(mesh, uv_option, view);
    } else {
        if (!uv_option.is_fbo_texture_allocated_) {
            /*** Accumulation (initial input) ***/
            uv_option.tex_sum_color_buffer_ = CreateTexture2D(
                view.GetWindowWidth(), view.GetWindowHeight(),
                GL_RGB16F, GL_RGB, GL_FLOAT, false, option);

            uv_option.tex_sum_weight_buffer_ = CreateTexture2D(
                view.GetWindowWidth(), view.GetWindowHeight(),
                GL_RGB16F, GL_RGB, GL_FLOAT, false, option);

            const int kNumOutputTex = 4;
            uv_option.tex_output_buffer_.resize(kNumOutputTex);

            /*** Intermediate output textures ***/
            /* color (forward, only for debugging) */
            uv_option.tex_output_buffer_[0] = CreateTexture2D(
                view.GetWindowWidth(), view.GetWindowHeight(),
                GL_RGB16F, GL_RGB, GL_FLOAT, false, option);

            /* depth (forward, for occlusion test) */
            uv_option.tex_output_buffer_[1] = CreateTexture2D(
                view.GetWindowWidth(), view.GetWindowHeight(),
                GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT,
                false, option);

            /*** Final output textures ***/
            /* color (atlas, read and write)  */
            uv_option.tex_output_buffer_[2] = CreateTexture2D(
                    view.GetWindowWidth(), view.GetWindowHeight(),
                    GL_RGB16F, GL_RGB, GL_FLOAT, false, option);

            /* weight (atlas, read and write. Only one channel should be enough) */
            uv_option.tex_output_buffer_[3] = CreateTexture2D(
                view.GetWindowWidth(), view.GetWindowHeight(),
                    GL_RGB16F, GL_RGB, GL_FLOAT, false, option);

            uv_option.is_fbo_texture_allocated_ = true;
        }

        /** Render to depth buffer **/
        uv_option.render_to_fbo_ = true;
        uv_forward_shader_.Render(mesh, uv_option, view);

        /** Only for debugging
        uv_option.SetVisualizeBuffer(1);
        simple_texture_shader_.Render(mesh, uv_option, view);
         **/

        /** Render to texture atlas **/
        uv_option.SetDepthBuffer(1);
        uv_option.render_to_fbo_ = true;
        uv_backward_shader_.Render(mesh, uv_option, view);

        uv_option.SetVisualizeBuffer(2);
        simple_texture_shader_.Render(mesh, uv_option, view);
    }

    return true;
}

bool GeometryRendererUV::UpdateGeometry() {
    uv_forward_shader_.InvalidateGeometry();
    uv_backward_shader_.InvalidateGeometry();
    simple_texture_shader_.InvalidateGeometry();

    return true;
}

}
}
}
