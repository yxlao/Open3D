//
// Created by wei on 4/16/19.
//

#include "Lighting.h"
#include "ImageExt.h"

namespace open3d {
namespace geometry {

IBLLighting::~IBLLighting() {}

bool IBLLighting::ReadEnvFromHDR(const std::string &filename) {
    hdr_ = io::ReadImageFromHDR(filename);
    if (hdr_->IsEmpty()) {
        utility::PrintDebug("Unable to load HDR texture.\n");
        return false;
    }

    return true;
}

bool IBLLighting::BindHDRTexture2D() {
    if (! is_hdr_buffer_generated_) {
        glGenTextures(1, &tex_hdr_buffer_);
        is_hdr_buffer_generated_ = true;
    }

    glBindTexture(GL_TEXTURE_2D, tex_hdr_buffer_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F,
        hdr_->width_, hdr_->height_, 0, GL_RGB, GL_FLOAT, hdr_->data_.data());

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    return true;
}

void IBLLighting::ShiftValue(const Eigen::Vector3d &direction,
                             const Eigen::Vector3f &psi) {
    auto n = direction.normalized();
    double u = std::atan2(n(2), n(0)) * M_1_PI + 0.5;
    double v = std::asin(n(1)) * M_2_PI + 0.5;
    auto value_ptr = geometry::PointerAt<float>(*hdr_, (int)u, (int)v, 0);
    for (int i = 0; i < 3; ++i) {
        value_ptr[i] += psi(i);
    }

    is_preprocessed_ = false;
}
}
}