//
// Created by wei on 9/27/18.
//


#pragma once

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
        images_[i] = other[i];
    }
}

template<typename VecType, size_t N>
ImagePyramidCuda<VecType, N> &ImagePyramidCuda<VecType, N>::operator=(
    const ImagePyramidCuda<VecType, N> &other) {
    if (this != &other) {
        server_ = other.server();
        for (size_t i = 0; i < N; ++i) {
            images_[i] = other[i];
        }
    }
    return *this;
}

template<typename VecType, size_t N>
void ImagePyramidCuda<VecType, N>::Create(int width, int height) {
    assert(width > 0 && height > 0);
    if (server_ != nullptr) {
        PrintError("[ImagePyramidCuda] Already created, abort!\n");
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
void ImagePyramidCuda<VecType, N>::Build(const ImageCuda<VecType> &image) {
    if (server_ == nullptr) {
        server_ = std::make_shared<ImagePyramidCudaServer<VecType, N>>();
    }

    images_[0].CopyFrom(image);
    for (size_t i = 1; i < N; ++i) {
        images_[i - 1].Downsample(images_[i]);
    }

    UpdateServer();
}

template<typename VecType, size_t N>
void ImagePyramidCuda<VecType, N>::UpdateServer() {
    if (server_ != nullptr) {
        for (size_t i = 0; i < N; ++i) {
            (*server_)[i] = *images_[i].server();
        }
    }
}

template<typename VecType, size_t N>
std::vector<cv::Mat> ImagePyramidCuda<VecType, N>::DownloadMats() {
    std::vector<cv::Mat> result;
    for (size_t i = 0; i < N; ++i) {
        result.emplace_back(images_[i].DownloadMat());
    }
    return result;
}


template<typename VecType, size_t N>
std::vector<std::shared_ptr<Image>> ImagePyramidCuda<VecType, N>
    ::DownloadImages(){
    std::vector<std::shared_ptr<Image> > result;
    for (size_t i = 0; i < N; ++i) {
        result.emplace_back(images_[i].DownloadImage());
    }
    return result;
}
}