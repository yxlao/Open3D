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

    /** Reserved for accumulation **/
    std::shared_ptr<geometry::Image> sum_color_;
    std::shared_ptr<geometry::Image> sum_weight_;

    bool UpdateSumTextures() {
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

        for (int v = 0; v < height; ++v) {
            for (int u = 0; u < width; ++u) {
                for (int c = 0; c < 3; ++c) {
                    auto delta_c =
                            geometry::PointerAt<float>(*delta_color, u, v, c);
                    auto delta_w =
                            geometry::PointerAt<float>(*delta_weight, u, v, c);
                    auto sum_c =
                            geometry::PointerAt<float>(*sum_color_, u, v, c);
                    auto sum_w =
                            geometry::PointerAt<float>(*sum_weight_, u, v, c);

                    *delta_w *= 0.005;
                    float weight = *sum_w + *delta_w;
                    float color =
                            std::abs(weight) < 1e-6
                                    ? 0
                                    : (*delta_w * *delta_c + *sum_w * *sum_c) /
                                              weight;
                    //                    utility::PrintError("(%d %d %d): %f *
                    //                    %f + %f * %f => %f\n",
                    //                        u, v, c, *delta_w, *delta_c,
                    //                        *sum_w, *sum_c, color);
                    *sum_w = *delta_w;
                    *sum_c = *delta_c;
                }  // c
            }      // v
        }          // u
    }

    std::pair<std::shared_ptr<geometry::Image>,
              std::shared_ptr<geometry::Image>>
    GetSumTextures() {
        return std::make_pair(sum_color_, sum_weight_);
    }
};
}  // namespace visualization
}  // namespace open3d
