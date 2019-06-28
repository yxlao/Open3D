//
// Created by wei on 4/15/19.
//

#pragma once

#include <AdvancedRendering/Geometry/ImageExt.h>
#include <AdvancedRendering/Geometry/Lighting.h>
#include <AdvancedRendering/Visualization/Shader/LightingRenderer.h>
#include <AdvancedRendering/Visualization/Utility/BufferHelper.h>
#include <Open3D/Open3D.h>

#include "../Utility/BufferHelper.h"
#include "RenderOptionAdvanced.h"

namespace open3d {
namespace visualization {

/** Visualizer for rendering with uv mapping **/
class VisualizerUV : public VisualizerWithKeyCallback {
public:
    /** This geometry object is supposed to include >= 1 texture(s) **/
    virtual bool AddGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) override;

    /** Handle forward / backward options **/
    virtual bool InitRenderOption() override;

    /** Call this function every time when
     * - AFTER @CreateVisualizerWindow
     *   :to ensure OpenGL context has been created.
     * - BEFORE @Run (or whatever customized rendering task)
     *   :to ensure target image is ready.
     * Currently we only support one target image.
     *   It would remove the previous bound image.
     * **/
    bool EnableForwardMode();
    bool EnableBackwardMode(const std::shared_ptr<geometry::Image> &image);

    std::pair<std::shared_ptr<geometry::Image>,
              std::shared_ptr<geometry::Image>>
    GetSubTextures() {
        auto advanced_render_option =
                (const RenderOptionAdvanced &)*render_option_ptr_;
        int width = advanced_render_option.tex_uv_width_;
        int height = advanced_render_option.tex_uv_height_;

        glBindTexture(GL_TEXTURE_2D,
                      advanced_render_option.tex_backward_uv_color_buffer_);
        auto delta_color = ReadTexture2D(width, height, 3, 4, GL_RGB, GL_FLOAT);

        glBindTexture(GL_TEXTURE_2D,
                      advanced_render_option.tex_backward_uv_weight_buffer_);
        auto delta_weight =
                ReadTexture2D(width, height, 3, 4, GL_RGB, GL_FLOAT);
        return std::make_pair(delta_color, delta_weight);
    }
};
}  // namespace visualization
}  // namespace open3d
