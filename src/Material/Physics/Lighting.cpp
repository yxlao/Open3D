//
// Created by wei on 4/16/19.
//

#include "Lighting.h"

namespace open3d {
namespace physics {

IBLLighting::~IBLLighting() {

}

bool IBLLighting::ReadDataFromHDR() {
    stbi_set_flip_vertically_on_load(true);
    int width, height, channel;
    float *data = stbi_loadf(filename_.c_str(), &width, &height, &channel, 0);

    if (!data) {
        utility::PrintDebug("Unable to load HDR texture.\n");
        return false;
    }

    glGenTextures(1, &tex_hdr_buffer_);
    glBindTexture(GL_TEXTURE_2D, tex_hdr_buffer_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, width, height, 0,
                 GL_RGB, GL_FLOAT, data);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    stbi_image_free(data);
    return true;
}
}
}