//
// Created by wei on 4/15/19.
//

#pragma once

#include <Open3D/Open3D.h>
#include <AdvancedRendering/Geometry/Lighting.h>
#include <AdvancedRendering/Visualization/Shader/LightingPreprocessRenderer.h>
#include <AdvancedRendering/Geometry/ImageExt.h>
#include "RenderOptionWithLighting.h"

namespace open3d {
namespace visualization {

/** Visualizer for rendering with textures **/
class VisualizerUV : public VisualizerWithKeyCallback {
public:
    /** Handle geometry (including textures) **/
    virtual bool AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) override;

    virtual bool InitRenderOption() override {
        render_option_ptr_ = std::unique_ptr<RenderOptionWithTargetImage>(
            new RenderOptionWithTargetImage);
        return true;
    }

    /** Call this function
     * - AFTER @CreateVisualizerWindow
     *   to ensure OpenGL context has been created.
     * - BEFORE @Run (or whatever customized rendering task)
     *   to ensure target image is ready.
     *   Currently we only support one lighting.
     *   It would remove the previous bound lighting.
     * **/
    bool UpdateTargetImage(
        const std::shared_ptr<geometry::Image> &image_ptr) {
        target_image_ = geometry::FlipImageExt(*image_ptr);

        auto &render_option_with_target =
            (std::shared_ptr<RenderOptionWithTargetImage> &) render_option_ptr_;
        if (!render_option_with_target->is_tex_allocated_) {
            glGenTextures(1, &render_option_with_target->tex_image_buffer_);
            render_option_with_target->is_tex_allocated_ = true;
        }

        auto texture_id = render_option_with_target->tex_image_buffer_;
        std::cout << glGetError() << " texture id: " << texture_id << "\n";

        glBindTexture(GL_TEXTURE_2D, texture_id);
        std::cout << glGetError() << " bind texture id: " << texture_id << "\n";

        GLenum format;
        switch (target_image_->num_of_channels_) {
            case 1: { format = GL_RED; break; }
            case 3: { format = GL_RGB; break; }
            case 4: { format = GL_RGBA; break; }
            default: {
                utility::PrintWarning("Unknown format, abort!\n");
                return false;
            }
        }

        GLenum type;
        switch (target_image_->bytes_per_channel_) {
            case 1: { type = GL_UNSIGNED_BYTE; break; }
            case 2: { type = GL_UNSIGNED_SHORT; break;}
            case 4: { type = GL_FLOAT; break; }
            default: {
                utility::PrintWarning("Unknown format, abort!\n");
                return false;
            }
        }

        std::cout << glGetError() << " Before binding: " << texture_id << "\n";

        glTexImage2D(GL_TEXTURE_2D, 0, format,
                     target_image_->width_, target_image_->height_,
                     0, format, type,
                     target_image_->data_.data());
        std::cout << glGetError() << " After binding: " << texture_id << "\n";

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        return true;
    }

public:
    std::shared_ptr<geometry::Image> target_image_;
};
}  // namespace visualization
}  // namespace open3d
