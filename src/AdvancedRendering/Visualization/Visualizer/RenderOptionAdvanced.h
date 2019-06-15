//
// Created by Wei Dong on 2019-05-28.
//

#pragma once

#include <AdvancedRendering/Geometry/Lighting.h>
#include <Open3D/Open3D.h>

namespace open3d {
namespace visualization {

class RenderOptionAdvanced : public RenderOption {
public:
    /**************************************/
    geometry::Lighting::LightingType type_;

    /** States for image based lighting **/
    bool is_env_tex_allocated_ = false;
    GLuint tex_hdr_buffer_;
    GLuint tex_env_buffer_;
    GLuint tex_env_diffuse_buffer_;
    GLuint tex_env_specular_buffer_;
    GLuint tex_lut_specular_buffer_;

    /** States for spot lighting **/
    std::vector<Eigen::Vector3f> spot_light_positions_;
    std::vector<Eigen::Vector3f> spot_light_colors_;

public:
    /**************************************/
    bool forward_ = true;
    bool render_to_fbo_ = false;

    /* Input image */
    bool is_image_tex_allocated_ = false;
    GLuint tex_image_buffer_;

    bool is_fbo_allocated_ = false;
    GLuint fbo_forward_;
    GLuint fbo_backward_;
    GLuint rbo_forward_;
    GLuint rbo_backward_;

    /* Output uv */
    bool is_fbo_tex_allocated_ = false;
    GLuint tex_forward_image_buffer_;
    GLuint tex_forward_depth_buffer_;
    GLuint tex_backward_uv_color_buffer_;
    GLuint tex_backward_uv_weight_buffer_;

    GLuint tex_visualize_buffer_;
    void SetVisualizeBuffer(GLuint tex_vis_buffer) {
        tex_visualize_buffer_ = tex_vis_buffer;
    }

    int tex_image_width_;
    int tex_image_height_;

    int tex_uv_width_;
    int tex_uv_height_;
};
}  // namespace visualization
}  // namespace open3d
