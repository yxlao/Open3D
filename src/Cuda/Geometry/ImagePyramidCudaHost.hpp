//
// Created by wei on 9/27/18.
//


#pragma once

#include "ImagePyramidCuda.h"
#include <Cuda/Common/UtilsCuda.h>
#include <Core/Core.h>

namespace open3d {
namespace cuda {

template<typename VecType, size_t N>
ImagePyramidCuda<VecType, N>::ImagePyramidCuda() : device_(nullptr) {}

template<typename VecType, size_t N>
ImagePyramidCuda<VecType, N>::~ImagePyramidCuda() {
    Release();
}

template<typename VecType, size_t N>
ImagePyramidCuda<VecType, N>::ImagePyramidCuda(
    const ImagePyramidCuda<VecType, N> &other) {

    device_ = other.device_;
    for (size_t i = 0; i < N; ++i) {
        images_[i] = other[i];
    }
}

template<typename VecType, size_t N>
ImagePyramidCuda<VecType, N> &ImagePyramidCuda<VecType, N>::operator=(
    const ImagePyramidCuda<VecType, N> &other) {
    if (this != &other) {
        device_ = other.device_;
        for (size_t i = 0; i < N; ++i) {
            images_[i] = other[i];
        }
    }
    return *this;
}

template<typename VecType, size_t N>
bool ImagePyramidCuda<VecType, N>::Create(int width, int height) {
    assert(width > 0 && height > 0);
    if (device_ != nullptr) {
        if (this->width(0) != width || this->height(0) != height) {
            PrintError("[ImagePyramidCuda] Incompatible image size,"
                       "@Create aborted.\n");
            return false;
        }
        return true;
    }

    if ((width >> N) == 0 || (height >> N) == 0) {
        PrintError("[ImagePyramidCuda] Width %d || height %d too small,"
                   "@Create aborted.\n", width, height);
        return false;
    }

    device_ = std::make_shared<ImagePyramidCudaDevice<VecType, N>>();
    for (size_t i = 0; i < N; ++i) {
        int w = width >> i;
        int h = height >> i;
        bool success = images_[i].Create(w, h);
        assert(success);
    }

    UpdateDevice();
    return true;
}

template<typename VecType, size_t N>
void ImagePyramidCuda<VecType, N>::Release() {
    for (size_t i = 0; i < N; ++i) {
        images_[i].Release();
    }
    device_ = nullptr;
}

template<typename VecType, size_t N>
void ImagePyramidCuda<VecType, N>::Build(const ImageCuda<VecType> &image) {
    bool success = Create(image.width_, image.height_);
    if (success) {
        images_[0].CopyFrom(image);
        for (size_t i = 1; i < N; ++i) {
            images_[i - 1].Downsample(images_[i]);
        }
        UpdateDevice();
    }
}

template<typename VecType, size_t N>
void ImagePyramidCuda<VecType, N>::UpdateDevice() {
    if (device_ != nullptr) {
        for (size_t i = 0; i < N; ++i) {
            images_[i].UpdateDevice();
            (*device_)[i] = *images_[i].device_;
        }
    }
}

template<typename VecType, size_t N>
std::vector<std::shared_ptr<Image>> ImagePyramidCuda<VecType, N>::
DownloadImages() {
    std::vector<std::shared_ptr<Image> > result;
    if (device_ == nullptr) {
        PrintWarning("[ImagePyramidCuda] Not initialized,"
                     "@DownloadImages aborted.\n");
        return result;
    }
    for (size_t i = 0; i < N; ++i) {
        result.emplace_back(images_[i].DownloadImage());
    }
    return result;
}

template<typename VecType, size_t N>
std::vector<cv::Mat> ImagePyramidCuda<VecType, N>::DownloadMats() {
    std::vector<cv::Mat> result;
    if (device_ == nullptr) {
        PrintWarning("[ImagePyramidCuda] Not initialized,"
                     "@DownloadMats aborted.\n");
        return result;
    }
    for (size_t i = 0; i < N; ++i) {
        result.emplace_back(images_[i].DownloadMat());
    }
    return result;
}
} // cuda
} // open3d