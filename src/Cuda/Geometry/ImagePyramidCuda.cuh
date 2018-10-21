//
// Created by wei on 9/27/18.
//

#ifndef OPEN3D_IMAGEPYRAMIDCUDA_CUH
#define OPEN3D_IMAGEPYRAMIDCUDA_CUH

#include "ImagePyramidCuda.h"
#include <Cuda/Common/UtilsCuda.h>
#include <Core/Core.h>

namespace open3d {
template<typename VecType, size_t N>
ImagePyramidCuda<VecType, N>::ImagePyramidCuda() {}

template<typename VecType, size_t N>
ImagePyramidCuda<VecType, N>::~ImagePyramidCuda() {
    Release();
}

template<typename VecType, size_t N>
ImagePyramidCuda<VecType, N>::ImagePyramidCuda(
    const ImagePyramidCuda<VecType, N> &other) {

    server_ = other.server();
    for (size_t i = 0; i < N; ++i) {
        images_[i] = other.level(i);
    }
}

template<typename VecType, size_t N>
ImagePyramidCuda<VecType, N> &ImagePyramidCuda<VecType, N>::operator=(
    const ImagePyramidCuda<VecType, N> &other) {
    if (this != &other) {
        server_ = other.server();
        for (size_t i = 0; i < N; ++i) {
            images_[i] = other.level(i);
        }
    }
    return *this;
}

template<typename VecType, size_t N>
void ImagePyramidCuda<VecType, N>::Create(int width, int height) {
    assert(width > 0 && height > 0);
    if (server_ != nullptr) {
        PrintWarning("Already created, stop re-creating!\n");
        return;
    }

    server_ = std::make_shared<ImagePyramidCudaServer<VecType, N>>();
    for (size_t i = 0; i < N; ++i) {
        int w = width >> i;
        int h = height >> i;
        if (w == 0 || h == 0) {
            PrintError("Invalid width %d || height %d at level %d!\n", w, h, i);
            return;
        }
        images_[i].Create(w, h);
    }

    UpdateServer();
}

template<typename VecType, size_t N>
void ImagePyramidCuda<VecType, N>::Release() {
    for (size_t i = 0; i < N; ++i) {
        images_[i].Release();
    }
    server_ = nullptr;
}

template<typename VecType, size_t N>
void ImagePyramidCuda<VecType, N>::Build(cv::Mat &m) {
    if (server_ == nullptr) {
        server_ = std::make_shared<ImagePyramidCudaServer<VecType, N>>();
    }

    images_[0].Upload(m);
    for (size_t i = 1; i < N; ++i) {
        images_[i - 1].Downsample(images_[i]);
    }

    UpdateServer();
}

template<typename VecType, size_t N>
void ImagePyramidCuda<VecType, N>::Build(const ImageCuda<VecType> &image) {
    if (server_ == nullptr) {
        server_ = std::make_shared<ImagePyramidCudaServer<VecType, N>>();
    }

    image.CopyTo(images_[0]);
    for (size_t i = 1; i < N; ++i) {
        images_[i - 1].Downsample(images_[i]);
    }

    UpdateServer();
}

template<typename VecType, size_t N>
void ImagePyramidCuda<VecType, N>::UpdateServer() {
    if (server_ != nullptr) {
        for (size_t i = 0; i < N; ++i) {
            server_->level(i) = *images_[i].server();
        }
    }
}

template<typename VecType, size_t N>
std::vector<cv::Mat> ImagePyramidCuda<VecType, N>::Download() {
    std::vector<cv::Mat> result;
    for (size_t i = 0; i < N; ++i) {
        result.emplace_back(images_[i].Download());
    }
    return result;
}

}
#endif //OPEN3D_IMAGEPYRAMIDCUDA_CUH
