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
        uv_option.render_to_fbo_ = true;
        if (!uv_option.is_fbo_texture_allocated_) {
            const int kNumOutputTex = 2;
            uv_option.tex_output_buffer_.resize(kNumOutputTex);

            /* color */
            uv_option.tex_output_buffer_[0] = CreateTexture2D(
                view.GetWindowWidth(), view.GetWindowHeight(),
                GL_RGB16F, GL_RGB, GL_FLOAT,
                false, option);

            /* depth */
            uv_option.tex_output_buffer_[1] = CreateTexture2D(
                view.GetWindowWidth(), view.GetWindowHeight(),
                GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT,
                false, option);

            uv_option.is_fbo_texture_allocated_ = true;
        }

        uv_forward_shader_.Render(mesh, uv_option, view);

        uv_option.SetVisualizeBuffer(1);
        simple_texture_shader_.Render(mesh, uv_option, view);

//        uv_backward_shader_.Render(mesh, uv_option, view);
    }

    return true;
}

bool GeometryRendererUV::UpdateGeometry() {
    uv_forward_shader_.InvalidateGeometry();
    uv_backward_shader_.InvalidateGeometry();

    return true;
}

}
}
}
