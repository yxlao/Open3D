//
// Created by wei on 4/13/19.
//

#include "ShaderWrapperPBR.h"
#include "Primitives.h"

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

GLuint ShaderWrapperPBR::BindTexture2D(
    const geometry::Image &texture,
    const visualization::RenderOption &option) {

    GLuint texture_id;
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);

    GLenum format;
    switch (texture.num_of_channels_) {
        case 1: {
            format = GL_RED;
            break;
        }
        case 3: {
            format = GL_RGB;
            break;
        }
        case 4: {
            format = GL_RGBA;
            break;
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
    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    return texture_id;
}

GLuint ShaderWrapperPBR::CreateTexture2D(GLuint width, GLuint height,
                                         bool use_mipmap,
                                         const visualization::RenderOption &option) {
    GLuint texture_id;
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0,
                 GL_RGB16F, width, height, 0, GL_RG, GL_FLOAT, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    if (use_mipmap) {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                        GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                        GL_LINEAR);
    } else {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    return texture_id;
}

GLuint ShaderWrapperPBR::BindTextureCubemap(
    const std::vector<geometry::Image> &textures,
    const visualization::RenderOption &option) {

    assert(textures.size() == 6);

    GLuint texture_id;
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_CUBE_MAP, texture_id);
    for (int i = 0; i < 6; ++i) {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                     0, GL_RGB16F, textures[i].width_, textures[i].height_, 0,
                     GL_RGB, GL_FLOAT, textures[i].data_.data());
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER,
                    GL_LINEAR);
    return texture_id;
}

GLuint ShaderWrapperPBR::CreateTextureCubemap(
    GLuint size, bool use_mipmap,
    const visualization::RenderOption &option) {

    GLuint texture_id;
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_CUBE_MAP, texture_id);
    for (int i = 0; i < 6; ++i) {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                     0, GL_RGB16F, size, size, 0,
                     GL_RGB, GL_FLOAT, nullptr);
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    if (use_mipmap) {
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER,
                        GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER,
                        GL_LINEAR);
    } else {
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    return texture_id;
}

void ShaderWrapperPBR::LoadCube(std::vector<Eigen::Vector3f> &vertices,
                                std::vector<Eigen::Vector3i> &triangles) {
    vertices = physics::kCubeVertices;
    triangles = physics::kCubeTriangles;
}

void ShaderWrapperPBR::LoadQuad(std::vector<Eigen::Vector3f> &vertices,
                                std::vector<Eigen::Vector2f> &uvs) {
    vertices = physics::kQuadVertices;
    uvs = physics::kQuadUVs;
}

void ShaderWrapperPBR::LoadViews(
    std::vector<GLHelper::GLMatrix4f> &views) {
    views = {
        GLHelper::LookAt(Eigen::Vector3d::Zero(),
                         Eigen::Vector3d(+1, 0, 0),
                         Eigen::Vector3d(0, -1, 0)),
        GLHelper::LookAt(Eigen::Vector3d::Zero(),
                         Eigen::Vector3d(-1, 0, 0),
                         Eigen::Vector3d(0, -1, 0)),
        GLHelper::LookAt(Eigen::Vector3d::Zero(),
                         Eigen::Vector3d(0, +1, 0),
                         Eigen::Vector3d(0, 0, +1)),
        GLHelper::LookAt(Eigen::Vector3d::Zero(),
                         Eigen::Vector3d(0, -1, 0),
                         Eigen::Vector3d(0, 0, -1)),
        GLHelper::LookAt(Eigen::Vector3d::Zero(),
                         Eigen::Vector3d(0, 0, +1),
                         Eigen::Vector3d(0, -1, 0)),
        GLHelper::LookAt(Eigen::Vector3d::Zero(),
                         Eigen::Vector3d(0, 0, -1),
                         Eigen::Vector3d(0, -1, 0))
    };
}
}
}
}
