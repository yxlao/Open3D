//
// Created by Wei Dong on 2019-05-28.
//

#pragma once

#include <Open3D/Open3D.h>
#include <InverseRendering/Geometry/Lighting.h>

namespace open3d {
namespace visualization {

class RenderOptionWithLighting : public RenderOption {
public:
    std::shared_ptr<geometry::Lighting> lighting_ptr_;

    /** Reserved for IBL lighting **/
    bool is_tex_preprocessed_;
    GLuint tex_hdr_buffer_;
    GLuint tex_env_buffer_;
    GLuint tex_env_diffuse_buffer_;
    GLuint tex_env_specular_buffer_;
    GLuint tex_lut_specular_buffer_;
};
}
}
