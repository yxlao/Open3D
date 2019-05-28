//
// Created by wei on 4/16/19.
//

#include "Lighting.h"
#include "ImageExt.h"
#include <Eigen/Eigen>
#include <random>

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
    if (!is_hdr_buffer_generated_) {
        glGenTextures(1, &tex_hdr_buffer_);
        is_hdr_buffer_generated_ = true;
    }

    utility::PrintInfo("Rebinding HDR Texture\n");
    glBindTexture(GL_TEXTURE_2D, tex_hdr_buffer_);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGB16F,
                 hdr_->width_,
                 hdr_->height_,
                 0,
                 GL_RGB,
                 GL_FLOAT,
                 hdr_->data_.data());

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    return true;
}

namespace {
Eigen::Vector2d SphereToLogLat(const Eigen::Vector3d &position) {
    auto n = position.normalized();
    auto uv = Eigen::Vector2d(std::atan2(n(2), n(0)) * (0.5 * M_1_PI) + 0.5,
                              std::asin(n(1)) * M_1_PI + 0.5);
    return uv;
}

int CycleIndex(double x, int range) {
    int x_ = (int) std::round(x * (range - 1));
    x_ = (x_ < 0) ? x_ + (range - 1) : x_;
    x_ = (x_ > range - 1) ? x_ - (range - 1) : x_;
    return x_;
}
}

Eigen::Vector3d IBLLighting::SampleAround(const Eigen::Vector3d &N,
                                          double sigma) {
    // First sample cos
    static std::random_device rd;
    static std::normal_distribution<float> nd(1, sigma);
    static std::uniform_real_distribution<float> ud(-1, 1);

    float cos_theta = nd(rd);
    cos_theta = cos_theta > 1 ? 2 - cos_theta : cos_theta;
    cos_theta = std::max(cos_theta, 0.0f);
    float sin_theta = std::sqrt(1 - cos_theta * cos_theta);

    float phi = M_PI * ud(rd);

    Eigen::Vector3d H;
    H(0) = cos(phi) * sin_theta;
    H(1) = sin(phi) * sin_theta;
    H(2) = cos_theta;

    Eigen::Vector3d up = std::abs(N(2)) < 0.999 ?
        Eigen::Vector3d(0.0, 0.0, 1.0) : Eigen::Vector3d(1.0, 0.0, 0.0);
    Eigen::Vector3d tangent = (up.cross(N)).normalized();
    Eigen::Vector3d bitangent = N.cross(tangent);

    return tangent * H(0) + bitangent * H(1) + N * H(2);
}

Eigen::Vector3f IBLLighting::GetValueAt(const Eigen::Vector3d &direction) {
    auto uv = SphereToLogLat(direction);

    int u = CycleIndex(uv(0), hdr_->width_);
    int v = CycleIndex(uv(1), hdr_->height_);

    auto value_ptr = geometry::PointerAt<float>(*hdr_, u, v, 0);
    return Eigen::Vector3f(value_ptr[0], value_ptr[1], value_ptr[2]);
}

void IBLLighting::SetValueAt(const Eigen::Vector3d &direction,
                             const Eigen::Vector3f &value) {
    auto uv = SphereToLogLat(direction);

    int u = CycleIndex(uv(0), hdr_->width_);
    int v = CycleIndex(uv(1), hdr_->height_);

    auto value_ptr = geometry::PointerAt<float>(*hdr_, u, v, 0);
    value_ptr[0] = value(0);
    value_ptr[1] = value(1);
    value_ptr[2] = value(2);
}
}
}
