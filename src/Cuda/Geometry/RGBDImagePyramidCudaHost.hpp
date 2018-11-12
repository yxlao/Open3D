//
// Created by wei on 11/8/18.
//

#include "RGBDImagePyramidCuda.h"

namespace open3d {
template<size_t N>
RGBDImagePyramidCuda<N>::RGBDImagePyramidCuda() {}

template<size_t N>
RGBDImagePyramidCuda<N>::~RGBDImagePyramidCuda() {
    Release();
}

template<size_t N>
RGBDImagePyramidCuda<N>::RGBDImagePyramidCuda(
    const RGBDImagePyramidCuda<N> &other) {
    server_ = other.server();
    for (size_t i = 0; i < N; ++i) {
        rgbd_[i] = other[i];
    }
}

template<size_t N>
RGBDImagePyramidCuda<N>& RGBDImagePyramidCuda<N>::operator=(
    const RGBDImagePyramidCuda<N> &other) {
    if (this != &other) {
        server_ = other.server();
        for (size_t i = 0; i < N; ++i) {
            rgbd_[i] = other[i];
        }
    }
    return *this;
}

template<size_t N>
void RGBDImagePyramidCuda<N>::Create(int width, int height) {
    assert(width > 0 && height > 0);
    if (server_ != nullptr) {
        PrintError("[RGBDImagePyramidCuda] Already created, abort!\n");
        return;
    }

    server_ = std::make_shared<RGBDImagePyramidCudaServer<N>> ();
    for (size_t i = 0; i < N; ++i) {
        int w = width >> i;
        int h = height >> i;
        if (w == 0 || h == 0) {
            PrintError("Invalid width %d || height %d at level %d!\n", w, h, i);
            return;
        }
        rgbd_[i].Create(w, h);
    }

    UpdateServer();
}

template<size_t N>
void RGBDImagePyramidCuda<N>::Release() {
    for (size_t i = 0; i < N; ++i) {
        rgbd_[i].Release();
    }
    server_ = nullptr;
}

template<size_t N>
void RGBDImagePyramidCuda<N>::Build(RGBDImageCuda &rgbd) {
    if (server_ == nullptr) {
        server_ = std::make_shared<RGBDImagePyramidCudaServer<N>>();
    }

    rgbd_[0].Upload(rgbd.depthf(), rgbd.color());
    for (size_t i = 1; i < N; ++i) {
        rgbd_[i - 1].depthf().Downsample(rgbd_[i].depthf());
        rgbd_[i - 1].color().Downsample(rgbd_[i].color());
    }

    UpdateServer();
}

template<size_t N>
void RGBDImagePyramidCuda<N>::UpdateServer() {
    if (server_ != nullptr) {
        for (size_t i = 0; i < N; ++i) {
            (*server_)[i] = *rgbd_[i].server();
        }
    }
}
}