//
// Created by wei on 11/5/18.
//

#include "RGBDImageCuda.h"

namespace open3d {
RGBDImageCuda::RGBDImageCuda(float depth_near,
                             float depth_far,
                             float depth_factor)
    : depth_near_(depth_near),
      depth_far_(depth_far),
      depth_factor_(depth_factor) {}

RGBDImageCuda::RGBDImageCuda(ImageCuda<Vector1f> &depth,
                             ImageCuda<Vector3b> &color,
                             float depth_near,
                             float depth_far,
                             float depth_factor)
    : depth_near_(depth_near),
      depth_far_(depth_far),
      depth_factor_(depth_factor) {
    Create(depth, color);
}

RGBDImageCuda::RGBDImageCuda(const open3d::RGBDImageCuda &other) {
    server_ = other.server();

    depth_near_ = other.depth_near_;
    depth_far_ = other.depth_far_;
    depth_factor_ = other.depth_factor_;

    Create(other.depth(), other.color());
}


RGBDImageCuda& RGBDImageCuda::operator=(const open3d::RGBDImageCuda &other) {
    if (this != &other) {
        server_ = other.server();

        depth_near_ = other.depth_near_;
        depth_far_ = other.depth_far_;
        depth_factor_ = other.depth_factor_;

        Create(other.depth(), other.color());
    }
    return *this;
}

RGBDImageCuda::~RGBDImageCuda() {
    Release();
}

void RGBDImageCuda::Create(const ImageCuda<Vector1f> &depth,
                           const ImageCuda<Vector3b> &color) {
    server_ = std::make_shared<RGBDImageCudaServer>();

    depth_ = depth;
    color_ = color;

    UpdateServer();
}

void RGBDImageCuda::Release() {
    server_ = nullptr;

    depth_.Release();
    color_.Release();
}

void RGBDImageCuda::Upload(cv::Mat &depth, cv::Mat &color) {
    depths_.Upload(depth);
    color_.Upload(color);

    depths_.ToFloat(depth_, 1.0f / depth_factor_);

    UpdateServer();
}

void RGBDImageCuda::UpdateServer() {
    if (server_ != nullptr) {
        server_->color() = *color_.server();
        server_->depth() = *depth_.server();

        server_->depth_near_ = depth_near_;
        server_->depth_far_ = depth_far_;
        server_->depth_factor_ = depth_factor_;
    }
}

}