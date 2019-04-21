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

    GLenum format;
    switch (texture.num_of_channels_) {
        case 1: {
            format = GL_RED; break;
        }
        case 3: {
            format = GL_RGB; break;
        }
        case 4: {
            format = GL_RGBA; break;
        }
        default: {
            format = GL_RGB;
            utility::PrintDebug("Unknown format, abort!\n");
            break;
        }
    }

    glTexImage2D(GL_TEXTURE_2D, 0, format,
                 texture.width_, texture.height_, 0, format,
                 texture.bytes_per_channel_ == 2 ?
                 GL_UNSIGNED_SHORT : GL_UNSIGNED_BYTE,
                 texture.data_.data());
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    return texture_id;
}

}
}
}
