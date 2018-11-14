//
// Created by wei on 11/5/18.
//

#include "RGBDImageCuda.h"

namespace open3d {
RGBDImageCuda::RGBDImageCuda(float depth_near,
                             float depth_far,
                             float depth_factor)
    : depth_near_(depth_near), depth_far_(depth_far),
      depth_factor_(depth_factor),
      width_(-1), height_(-1), server_(nullptr) {}

RGBDImageCuda::RGBDImageCuda(int width, int height,
                             float depth_near,
                             float depth_far,
                             float depth_factor)
    : depth_near_(depth_near), depth_far_(depth_far),
      depth_factor_(depth_factor) {
    Create(width, height);
}

RGBDImageCuda::RGBDImageCuda(const open3d::RGBDImageCuda &other) {
    server_ = other.server();

    depth_near_ = other.depth_near_;
    depth_far_ = other.depth_far_;
    depth_factor_ = other.depth_factor_;

    depth_raw_ = other.depth_raw();
    depthf_ = other.depthf();
    color_ = other.color();
    intensity_ = other.intensity();
}

RGBDImageCuda &RGBDImageCuda::operator=(const open3d::RGBDImageCuda &other) {
    if (this != &other) {
        Release();

        server_ = other.server();

        depth_near_ = other.depth_near_;
        depth_far_ = other.depth_far_;
        depth_factor_ = other.depth_factor_;

        depth_raw_ = other.depth_raw();
        depthf_ = other.depthf();
        color_ = other.color();
        intensity_ = other.intensity();
    }
    return *this;
}

RGBDImageCuda::~RGBDImageCuda() {
    Release();
}

bool RGBDImageCuda::Create(int width, int height) {
    assert(width > 0 && height > 0);

    if (server_ != nullptr) {
        if (width_ != width || height_ != height) {
            PrintError("[RGBDImageCuda] Incompatible image size,"
                       "@Create aborted.\n");
            return false;
        }
        return true;
    }

    server_ = std::make_shared<RGBDImageCudaServer>();

    width_ = width;
    height_ = height;

    depth_raw_.Create(width, height);
    color_.Create(width, height);
    depthf_.Create(width, height);
    intensity_.Create(width, height);

    UpdateServer();
    return true;
}

void RGBDImageCuda::Release() {
    server_ = nullptr;

    depth_raw_.Release();
    color_.Release();
    depthf_.Release();
    intensity_.Release();
}

void RGBDImageCuda::Upload(Image &depth_raw, Image &color_raw) {
    assert(depth_raw.width_ == color_raw.width_
               && depth_raw.height_ == color_raw.height_);
    width_ = depth_raw.width_;
    height_ = depth_raw.height_;

    bool success = Create(width_, height_);
    if (success) {
        depth_raw_.Upload(depth_raw);
        color_.Upload(color_raw);
        depth_raw_.ConvertToFloat(depthf_, 1.0f / depth_factor_);
        color_.ConvertRGBToIntensity(intensity_);

        UpdateServer();
    }
}

void RGBDImageCuda::CopyFrom(RGBDImageCuda &other) {
    if (&other == this) return;
    bool success = Create(other.width_, other.height_);
    if (success) {
        depth_raw_.CopyFrom(other.depth_raw());
        color_.CopyFrom(other.color());
        depthf_.CopyFrom(other.depthf());
        intensity_.CopyFrom(other.intensity());

        UpdateServer();
    }
}

void RGBDImageCuda::Build(
    ImageCuda<Vector1s> &depth_raw, ImageCuda<Vector3b> &color_raw) {

    assert(depth_raw.width_ == color_raw.width_
               && depth_raw.height_ == color_raw.height_);
    width_ = depth_raw.width_;
    height_ = depth_raw.height_;

    bool success = Create(width_, height_);
    if (success) {
        depth_raw_.CopyFrom(depth_raw);
        color_.CopyFrom(color_raw);
        depth_raw_.ConvertToFloat(depthf_, 1.0f / depth_factor_);
        color_.ConvertRGBToIntensity(intensity_);

        UpdateServer();
    }
}

void RGBDImageCuda::UpdateServer() {
    if (server_ != nullptr) {
        server_->width_ = width_;
        server_->height_ = height_;

        depthf_.UpdateServer();
        server_->depth() = *depthf_.server();

        color_.UpdateServer();
        server_->color() = *color_.server();

        intensity_.UpdateServer();
        server_->intensity() = *intensity_.server();
    }
}

/** Legacy **/
void RGBDImageCuda::Upload(cv::Mat &depth, cv::Mat &color) {
    assert(depth.cols == color.cols && depth.rows == color.rows);
    width_ = depth.cols;
    height_ = depth.rows;

    bool success = Create(width_, height_);
    if (success) {
        depth_raw_.Upload(depth);
        color_.Upload(color);

        depth_raw_.ConvertToFloat(depthf_, 1.0f / depth_factor_);
        color_.ConvertRGBToIntensity(intensity_);

        UpdateServer();
    }
}
}