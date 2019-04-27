//
// Created by wei on 4/16/19.
//

#include "Lighting.h"

namespace open3d {
namespace geometry {

IBLLighting::~IBLLighting() {
    std::cerr << "Dealloc!\n";
}

bool IBLLighting::ReadEnvFromHDR(const std::string &filename){
    if (! hdr_.ReadFromHDR(filename, true)) {
        utility::PrintDebug("Unable to load HDR texture.\n");
        return false;
    }

    return true;
}

bool IBLLighting::BindHDRTexture2D() {
    glGenTextures(1, &tex_hdr_buffer_);
    glBindTexture(GL_TEXTURE_2D, tex_hdr_buffer_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F,
        hdr_.image_->width_, hdr_.image_->height_, 0,
        GL_RGB, GL_FLOAT, hdr_.image_->data_.data());

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    return true;
}
}
}