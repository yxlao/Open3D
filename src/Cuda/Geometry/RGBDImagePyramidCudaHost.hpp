//
// Created by wei on 11/8/18.
//

#include "RGBDImagePyramidCuda.h"

namespace open3d {
namespace cuda {
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
RGBDImagePyramidCuda<N> &RGBDImagePyramidCuda<N>::operator=(
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
bool RGBDImagePyramidCuda<N>::Create(int width, int height) {
    assert(width > 0 && height > 0);

    if (server_ != nullptr) {
        if (rgbd_[0].width_ != width || rgbd_[0].height_ != height) {
            PrintError("[RGBDImagePyramidCuda] Incompatible image size,"
                       "@Create aborted.\n");
            return false;
        }
        return true;
    }

    if ((width >> N) == 0 || (height >> N) == 0) {
        PrintError("[RGBDImagePyramidCuda] Width %d || height %d too small,"
                   "@Create aborted.\n", width, height);
        return false;
    }

    server_ = std::make_shared<RGBDImagePyramidCudaServer<N>>();
    for (size_t i = 0; i < N; ++i) {
        int w = width >> i;
        int h = height >> i;
        rgbd_[i].Create(w, h);
    }

    UpdateServer();
    return true;
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
    bool success = Create(rgbd.width_, rgbd.height_);
    if (success) {
        rgbd_[0].CopyFrom(rgbd);
        for (size_t i = 1; i < N; ++i) {
            rgbd_[i - 1].depthf().Downsample(rgbd_[i].depthf(),
                                             BoxFilterWithHoles);
            rgbd_[i - 1].color().Downsample(rgbd_[i].color());
            rgbd_[i - 1].intensity().Downsample(rgbd_[i].intensity());
        }
        UpdateServer();
    }
}

template<size_t N>
void RGBDImagePyramidCuda<N>::UpdateServer() {
    if (server_ != nullptr) {
        for (size_t i = 0; i < N; ++i) {
            rgbd_[i].UpdateServer();
            (*server_)[i] = *rgbd_[i].server();
        }
    }
}
} // cuda
} // open3d