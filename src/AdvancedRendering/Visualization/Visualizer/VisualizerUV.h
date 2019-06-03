//
// Created by wei on 4/15/19.
//

#pragma once

#include <Open3D/Open3D.h>
#include <AdvancedRendering/Geometry/Lighting.h>
#include <AdvancedRendering/Geometry/ImageExt.h>
#include <AdvancedRendering/Visualization/Shader/LightingRenderer.h>
#include <AdvancedRendering/Visualization/Utility/BindWrapper.h>

#include "RenderOptionWithLighting.h"

namespace open3d {
namespace visualization {

/** Visualizer for rendering with uv mapping **/
class VisualizerUV : public VisualizerWithKeyCallback {
public:
    /** This geometry object is supposed to include textures **/
    virtual bool AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) override;

    virtual bool InitRenderOption() override {
        render_option_ptr_ = std::unique_ptr<RenderOptionWithTargetImage>(
            new RenderOptionWithTargetImage);
        return true;
    }

    /** Call this function
     * - AFTER @CreateVisualizerWindow
     *   :to ensure OpenGL context has been created.
     * - BEFORE @Run (or whatever customized rendering task)
     *   :to ensure target image is ready.
     * Currently we only support one lighting.
     *   It would remove the previous bound lighting.
     * **/
    bool UpdateTargetImage(
        const std::shared_ptr<geometry::Image> &image) {
        auto tex_image = geometry::FlipImageExt(*image);

        /** Single instance of the texture buffer **/
        auto &render_option_with_target =
            (std::shared_ptr<RenderOptionWithTargetImage> &) render_option_ptr_;
        if (!render_option_with_target->is_tex_allocated_) {
            render_option_with_target->tex_image_buffer_
                = glsl::BindTexture2D(*tex_image, *render_option_with_target);
            render_option_with_target->is_tex_allocated_ = true;
        } else {
            glsl::BindTexture2D(render_option_with_target->tex_image_buffer_,
                *tex_image, *render_option_with_target);
        }

        return true;
    }
};
}  // namespace visualization
}  // namespace open3d
