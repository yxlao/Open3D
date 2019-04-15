//
// Created by wei on 4/13/19.
//

#include "ShaderWrapperPBR.h"

namespace open3d {
namespace visualization {

namespace glsl {

bool ShaderWrapperPBR::Render(const geometry::Geometry &geometry,
                              const std::vector<geometry::Image> &textures,
                              const physics::Lighting &lighting,
                              const RenderOption &option,
                              const ViewControl &view) {
    if (!compiled_) {
        Compile();
    }
    if (!bound_) {
        BindGeometry(geometry, option, view);
        BindTextures(textures, option, view);
        BindLighting(lighting, option, view);
    }
    if (!compiled_ || !bound_) {
        PrintShaderWarning("Something is wrong in compiling or binding.");
        return false;
    }

    /** In fact, geometry is only used for sanity check.
     *  We should have bound everything before calling RenderGeometry **/
    return RenderGeometry(geometry, option, view);
}

GLuint ShaderWrapperPBR::BindTexture(
    const geometry::Image &texture,
    const visualization::RenderOption &option) {
    GLuint texture_id;
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                 texture.width_, texture.height_, 0, GL_RGB, GL_UNSIGNED_BYTE,
                 texture.data_.data());

    if (option.interpolation_option_ ==
        RenderOption::TextureInterpolationOption::Nearest) {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    } else {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                        GL_LINEAR_MIPMAP_LINEAR);
        glGenerateMipmap(GL_TEXTURE_2D);
    }

    return texture_id;
}

}
}
}
