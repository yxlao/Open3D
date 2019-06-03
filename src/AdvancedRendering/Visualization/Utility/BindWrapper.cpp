//
// Created by wei on 4/13/19.
//

#include "BindWrapper.h"
#include "AdvancedRendering/Visualization/Shader/Primitives.h"

namespace open3d {
namespace visualization {

namespace glsl {

GLuint BindTexture2D(
    const geometry::Image &texture,
    const visualization::RenderOption &option) {

    GLuint texture_id;
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);

    bool success = BindTexture2D(texture_id, texture, option);

    return texture_id;
}

bool BindTexture2D(GLuint &texture_id,
                   const geometry::Image &texture,
                   const visualization::RenderOption &option) {
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
            utility::PrintWarning("Unknown format, abort!\n");
            return false;
        }
    }

    GLenum type;
    switch (texture.bytes_per_channel_) {
        case 1: {
            type = GL_UNSIGNED_BYTE;
            break;
        }
        case 2: {
            type = GL_UNSIGNED_SHORT;
            break;
        }
        case 4: {
            type = GL_FLOAT;
            break;
        }
        default: {
            utility::PrintWarning("Unknown format, abort!\n");
            return false;
        }
    }

    glTexImage2D(GL_TEXTURE_2D, 0, format,
                 texture.width_, texture.height_, 0, format, type,
                 texture.data_.data());

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    return true;
}

GLuint CreateTexture2D(
    GLuint width, GLuint height,
    GLenum internal_format, GLenum format, GLenum type,
    bool use_mipmap, const visualization::RenderOption &option) {
    GLuint texture_id;
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0,
                 internal_format, width, height, 0, format, type, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    if (use_mipmap) {
        glTexParameteri(GL_TEXTURE_2D,
                        GL_TEXTURE_MIN_FILTER,
                        GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    } else {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    return texture_id;
}

std::shared_ptr<geometry::Image> ReadTexture2D(
    GLuint width, GLuint height, int channels, int bytes_per_channel,
    GLenum format, GLenum type) {
    auto im = std::make_shared<geometry::Image>();
    im->PrepareImage(width, height, channels, bytes_per_channel);

    glGetTexImage(GL_TEXTURE_2D, 0, format, type, im->data_.data());

//    auto im_flipped = std::make_shared<geometry::Image>();
//    im_flipped->PrepareImage(width, height, channels, bytes_per_channel);
//    int bytes_per_line = im->BytesPerLine();
//    for (int i = 0; i < height; i++) {
//        memcpy(im_flipped->data_.data() + bytes_per_line * i,
//               im->data_.data() + bytes_per_line * (height - i - 1),
//               bytes_per_line);
//    }

    return im;
}

GLuint BindTextureCubemap(
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

    glTexParameteri(GL_TEXTURE_CUBE_MAP,
                    GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    return texture_id;
}

GLuint CreateTextureCubemap(
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
        glTexParameteri(GL_TEXTURE_CUBE_MAP,
                        GL_TEXTURE_MIN_FILTER,
                        GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    } else {
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    return texture_id;
}

void LoadCube(std::vector<Eigen::Vector3f> &vertices,
              std::vector<Eigen::Vector3i> &triangles) {
    vertices = geometry::kCubeVertices;
    triangles = geometry::kCubeTriangles;
}

void LoadQuad(std::vector<Eigen::Vector3f> &vertices,
              std::vector<Eigen::Vector2f> &uvs) {
    vertices = geometry::kQuadVertices;
    uvs = geometry::kQuadUVs;
}

void LoadViews(
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

bool CheckGLState(const std::string &msg) {
    GLenum ret = glGetError();
    if (ret != GL_NO_ERROR) {
        utility::PrintWarning(
            "[OpenGL error]: %d at %s\n", ret, msg.c_str());
        return false;
    }
    return true;
}
}
}
}
