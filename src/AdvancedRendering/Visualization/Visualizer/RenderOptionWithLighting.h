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

    /** Reserved for IBL lighting **/
    bool is_tex_preprocessed_ = false;
    GLuint tex_hdr_buffer_;
    GLuint tex_env_buffer_;
    GLuint tex_env_diffuse_buffer_;
    GLuint tex_env_specular_buffer_;
    GLuint tex_lut_specular_buffer_;

    std::vector<Eigen::Vector3f> spot_light_positions_;
    std::vector<Eigen::Vector3f> spot_light_colors_;
};

class RenderOptionWithTargetImage : public RenderOption {
public:
    bool is_tex_allocated_ = false;
    GLuint tex_image_buffer_;
};
}
}
