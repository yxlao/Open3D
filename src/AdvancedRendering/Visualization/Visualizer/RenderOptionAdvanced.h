//
// Created by Wei Dong on 2019-05-28.
//

#pragma once

#include <Open3D/Open3D.h>
#include <AdvancedRendering/Geometry/Lighting.h>

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
    /** States for differentiable rendering: we need a reference **/
    bool is_ref_tex_allocated_ = false;
    GLuint tex_ref_buffer_;
    bool forward_ = true;

public:
    /**************************************/
    /** States for rendering to buffer **/
    bool is_fbo_allocated_ = false;
    bool is_fbo_texture_allocated_ = false;
    bool render_to_fbo_ = false;
    int output_textures = 0;
    GLuint fbo_;
    GLuint rbo_;
    std::vector<GLuint> tex_output_buffer_;

    /** Manually select one in tex_output_buffer_ **/
    GLuint tex_visualize_buffer_;
    GLuint tex_depth_buffer_;

    void SetVisualizeBuffer(int i) {
        assert(i < tex_output_buffer_.size());
        tex_visualize_buffer_ = tex_output_buffer_[i];
    }

    void SetDepthBuffer(int i) {
        assert(i < tex_output_buffer_.size());
        tex_depth_buffer_ = tex_output_buffer_[i];
    }
};
}
}
