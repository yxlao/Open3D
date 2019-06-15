//
// Created by wei on 4/12/19.
//

#include "GeometryRendererUV.h"
#include <AdvancedRendering/Geometry/TexturedTriangleMesh.h>
#include <AdvancedRendering/Visualization/Visualizer/RenderOptionAdvanced.h>

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

    if (geometry_ptr_->GetGeometryType() !=
        geometry::Geometry::GeometryType::TexturedTriangleMesh) {
        utility::PrintWarning(
                "[GeometryRendererUV] "
                "Geometry type is not TexturedTriangleMesh\n");
        return false;
    }

    const auto &mesh = (const geometry::TexturedTriangleMesh &)(*geometry_ptr_);
    auto &uv_option = (RenderOptionAdvanced &)option;

    if (uv_option.forward_) {
        uv_option.render_to_fbo_ = false;
        uv_forward_shader_.Render(mesh, uv_option, view);
    } else {
        /** Render to depth buffer **/
        uv_option.render_to_fbo_ = true;
        glViewport(0, 0, uv_option.tex_image_width_,
                   uv_option.tex_image_height_);
        uv_forward_shader_.Render(mesh, uv_option, view);

        /** Only for debugging
        uv_option.SetVisualizeBuffer(uv_option.tex_forward_image_buffer_);
        simple_texture_shader_.Render(mesh, uv_option, view);
         **/

        /** Render to texture atlas **/
        uv_option.render_to_fbo_ = true;
        glViewport(0, 0, uv_option.tex_uv_width_, uv_option.tex_uv_height_);
        uv_backward_shader_.Render(mesh, uv_option, view);

        uv_option.render_to_fbo_ = false;
        glViewport(0, 0, uv_option.tex_image_width_,
                   uv_option.tex_image_height_);
        uv_option.SetVisualizeBuffer(uv_option.tex_backward_uv_color_buffer_);
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

}  // namespace glsl
}  // namespace visualization
}  // namespace open3d
