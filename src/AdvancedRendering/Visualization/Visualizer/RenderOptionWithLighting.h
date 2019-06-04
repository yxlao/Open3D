//
// Created by Wei Dong on 2019-05-28.
//

#pragma once

#include <Open3D/Open3D.h>
#include <AdvancedRendering/Geometry/Lighting.h>

namespace open3d {
namespace visualization {

class RenderOptionWithLighting : public RenderOption {
public:
    geometry::Lighting::LightingType type_;

    /** States for image based lighting **/
    bool is_tex_preprocessed_ = false;
    GLuint tex_hdr_buffer_;
    GLuint tex_env_buffer_;
    GLuint tex_env_diffuse_buffer_;
    GLuint tex_env_specular_buffer_;
    GLuint tex_lut_specular_buffer_;

    /** States for spot lighting **/
    std::vector<Eigen::Vector3f> spot_light_positions_;
    std::vector<Eigen::Vector3f> spot_light_colors_;
};

class RenderOptionWithTargetImage : public RenderOption {
public:

    /** States for differentiable rendering: we need an input image **/
    bool is_tex_allocated_ = false;
    GLuint tex_image_buffer_;

    bool forward_ = true;
};
}
}
