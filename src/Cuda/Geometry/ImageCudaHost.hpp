//
// Created by wei on 11/9/18.
//

#pragma once

#include "ImageCuda.h"
#include "VectorCuda.h"
#include <iostream>
#include <driver_types.h>
#include <Cuda/Common/UtilsCuda.h>
#include <vector_types.h>
#include <Core/Core.h>

namespace open3d {
/**
 * Client end
 */
template<typename VecType>
ImageCuda<VecType>::ImageCuda() : width_(-1), height_(-1) {
#ifdef HOST_DEBUG_MONITOR_LIFECYCLE
    PrintInfo("Default ImageCuda constructor.\n");
#endif
}

template<typename VecType>
ImageCuda<VecType>::ImageCuda(const ImageCuda<VecType> &other) {
#ifdef HOST_DEBUG_MONITOR_LIFECYCLE
    PrintInfo("ImageCuda copy constructor.\n");
#endif
    server_ = other.server();

    width_ = other.width_;
    height_ = other.height_;
    pitch_ = other.pitch_;

#ifdef HOST_DEBUG_MONITOR_LIFECYCLE
    PrintInfo("Ref count after copy construction: %d\n", server_.use_count());
#endif
}

template<typename VecType>
ImageCuda<VecType>::ImageCuda(int width, int height) {
    Create(width, height);
}

template<typename VecType>
ImageCuda<VecType> &ImageCuda<VecType>::operator=(const ImageCuda<VecType> &other) {
#ifdef HOST_DEBUG_MONITOR_LIFECYCLE
    PrintInfo("ImageCuda assignment operator.\n");
#endif
    if (this != &other) {
        Release();

        server_ = other.server();
        width_ = other.width_;
        height_ = other.height_;
        pitch_ = other.pitch_;

#ifdef HOST_DEBUG_MONITOR_LIFECYCLE
        PrintInfo("Ref count after copy construction: %d\n", server_.use_count());
#endif
    }

    return *this;
}

template<typename VecType>
ImageCuda<VecType>::~ImageCuda() {
#ifdef HOST_DEBUG_MONITOR_LIFECYCLE
    PrintInfo("Destructor.\n");
#endif
    Release();
}

template<typename VecType>
void ImageCuda<VecType>::Create(int width, int height) {
    assert(width > 0 && height > 0);

    if (server_ != nullptr) {
        PrintWarning("Already created, stop re-creating!\n");
        return;
    }

#ifdef HOST_DEBUG_MONITOR_LIFECYCLE
    PrintInfo("Creating.\n");
#endif

    width_ = width;
    height_ = height;
    pitch_ = 0;
    server_ = std::make_shared<ImageCudaServer<VecType>>();

    size_t pitch_size_t = 0;
    CheckCuda(cudaMallocPitch((void **) &(server_->data_),
                              &pitch_size_t,
                              sizeof(VecType) * width_, height_));
    pitch_ = (int) pitch_size_t;
    server_->width_ = width_;
    server_->height_ = height_;
    server_->pitch_ = pitch_;
}

template<typename VecType>
void ImageCuda<VecType>::Release() {
#ifdef HOST_DEBUG_MONITOR_LIFECYCLE
    if (server_ != nullptr) {
        PrintInfo("ref count before releasing: %d\n", server_.use_count());
    }
#endif

    if (server_ != nullptr && server_.use_count() == 1) {
        CheckCuda(cudaFree(server_->data_));
    }

    server_ = nullptr;
    width_ = -1;
    height_ = -1;
    pitch_ = -1;
}

template<typename VecType>
void ImageCuda<VecType>::CopyFrom(const ImageCuda<VecType> &other) {
    if (this == &other) return;

    if (server_ == nullptr) {
        Create(other.width_, other.height_);
    }

    if (other.width_ != width_ || other.height_ != height_) {
        PrintError("[ImageCuda] Incompatible image size!\n");
        return;
    }

    CheckCuda(cudaMemcpy2D(server_->data_, pitch_,
                           other.server()->data(), other.pitch_,
                           sizeof(VecType) * width_, height_,
                           cudaMemcpyDeviceToDevice));
}

template<typename VecType>
void ImageCuda<VecType>::Upload(cv::Mat &m) {
    assert(m.rows > 0 && m.cols > 0);

    /* Type checking */
    if (typeid(VecType) == typeid(Vector1s)) {
        assert(m.type() == CV_16UC1);
    } else if (typeid(VecType) == typeid(Vector4b)) {
        assert(m.type() == CV_8UC4);
    } else if (typeid(VecType) == typeid(Vector3b)) {
        assert(m.type() == CV_8UC3);
    } else if (typeid(VecType) == typeid(Vector1b)) {
        assert(m.type() == CV_8UC1);
    } else if (typeid(VecType) == typeid(Vector4f)) {
        assert(m.type() == CV_32FC4);
    } else if (typeid(VecType) == typeid(Vector3f)) {
        assert(m.type() == CV_32FC3);
    } else if (typeid(VecType) == typeid(Vector1f)) {
        assert(m.type() == CV_32FC1);
    } else {
        PrintWarning("Unsupported format %d!\n");
        return;
    }

    if (server_ == nullptr) {
        Create(m.cols, m.rows);
    }
    if (width_ != m.cols || height_ != m.rows) {
        PrintWarning("Incompatible image size!\n");
        return;
    }

    CheckCuda(cudaMemcpy2D(server_->data_, pitch_, m.data, m.step,
                           sizeof(VecType) * m.cols, m.rows,
                           cudaMemcpyHostToDevice));
}

template<typename VecType>
void ImageCuda<VecType>::Upload(Image &image) {
    assert(image.width_ > 0 && image.height_ > 0);

    /* Type checking */
    if (typeid(VecType) == typeid(Vector1s)) {
        assert(image.bytes_per_channel_ == 2 && image.num_of_channels_ == 1);
    } else if (typeid(VecType) == typeid(Vector3b)) {
        assert(image.bytes_per_channel_ == 1 && image.num_of_channels_ == 3);
    } else if (typeid(VecType) == typeid(Vector1b)) {
        assert(image.bytes_per_channel_ == 1 && image.num_of_channels_ == 1);
    } else if (typeid(VecType) == typeid(Vector3f)) {
        assert(image.bytes_per_channel_ == 4 && image.num_of_channels_ == 3);
    } else if (typeid(VecType) == typeid(Vector1f)) {
        assert(image.bytes_per_channel_ == 4 && image.num_of_channels_ == 1);
    } else {
        PrintWarning("Unsupported format %d!\n");
        return;
    }

    if (server_ == nullptr) {
        Create(image.width_, image.height_);
    }
    if (width_ != image.width_ || height_ != image.height_) {
        PrintWarning("Incompatible image size!\n");
        return;
    }

    CheckCuda(cudaMemcpy2D(server_->data_, pitch_,
                           image.data_.data(), image.BytesPerLine(),
                           sizeof(VecType) * image.width_, image.height_,
                           cudaMemcpyHostToDevice));
}

template<typename VecType>
cv::Mat ImageCuda<VecType>::DownloadMat() {
    cv::Mat m;
    if (typeid(VecType) == typeid(Vector1s)) {
        m = cv::Mat(height_, width_, CV_16UC1);
    } else if (typeid(VecType) == typeid(Vector4b)) {
        m = cv::Mat(height_, width_, CV_8UC4);
    } else if (typeid(VecType) == typeid(Vector3b)) {
        m = cv::Mat(height_, width_, CV_8UC3);
    } else if (typeid(VecType) == typeid(Vector1b)) {
        m = cv::Mat(height_, width_, CV_8UC1);
    } else if (typeid(VecType) == typeid(Vector4f)) {
        m = cv::Mat(height_, width_, CV_32FC4);
    } else if (typeid(VecType) == typeid(Vector3f)) {
        m = cv::Mat(height_, width_, CV_32FC3);
    } else if (typeid(VecType) == typeid(Vector1f)) {
        m = cv::Mat(height_, width_, CV_32FC1);
    } else {
        PrintWarning("Unsupported format %d!\n");
        return m;
    }

    if (server_ == nullptr) {
        PrintWarning("ImageCuda not initialized!\n");
        return m;
    }

    CheckCuda(cudaMemcpy2D(m.data, m.step, server_->data_, pitch_,
                           sizeof(VecType) * width_, height_,
                           cudaMemcpyDeviceToHost));
    return m;
}

template<typename VecType>
std::shared_ptr<Image> ImageCuda<VecType>::DownloadImage() {
    std::shared_ptr<Image> image = std::make_shared<Image>();

    if (typeid(VecType) == typeid(Vector1s)) {
        image->PrepareImage(width_, height_, 1, 2);
    } else if (typeid(VecType) == typeid(Vector3b)) {
        image->PrepareImage(width_, height_, 3, 1);
    } else if (typeid(VecType) == typeid(Vector1b)) {
        image->PrepareImage(width_, height_, 1, 1);
    } else if (typeid(VecType) == typeid(Vector3f)) {
        image->PrepareImage(width_, height_, 3, 4);
    } else if (typeid(VecType) == typeid(Vector1f)) {
        image->PrepareImage(width_, height_, 1, 4);
    } else {
        PrintWarning("Unsupported format %d!\n");
        return image;
    }

    if (server_ == nullptr) {
        PrintWarning("ImageCuda not initialized!\n");
        return image;
    }

    CheckCuda(cudaMemcpy2D(image->data_.data(), image->BytesPerLine(),
                           server_->data_, pitch_,
                           sizeof(VecType) * width_, height_,
                           cudaMemcpyDeviceToHost));
    return image;
}

template<typename VecType>
ImageCuda<VecType> ImageCuda<VecType>::Downsample(DownsampleMethod method) {
    ImageCuda<VecType> dst;
    dst.Create(width_ / 2, height_ / 2);
    Downsample(dst, method);
    return dst;
}

template<typename VecType>
void ImageCuda<VecType>::Downsample(ImageCuda<VecType> &image,
                                    DownsampleMethod method) {
    if (image.server() == nullptr) {
        image.Create(width_ / 2, height_ / 2);
    } else if (image.width_ != width_ / 2 || image.height_ != height_ / 2) {
        PrintError("Incompatible image size!\n");
        return;
    }
    ImageCudaKernelCaller<VecType>::DownsampleImageKernelCaller(
        *server_, *image.server(), method);
}

template<typename VecType>
ImageCuda<VecType> ImageCuda<VecType>::Shift(float dx, float dy,
                                             bool with_holes) {
    ImageCuda<VecType> dst;
    dst.Create(width_, height_);
    Shift(dst, dx, dy, with_holes);
    return dst;
}

template<typename VecType>
void ImageCuda<VecType>::Shift(ImageCuda<VecType> &image,
                               float dx, float dy, bool with_holes) {
    if (image.server() == nullptr) {
        image.Create(width_, height_);
    } else if (image.width_ != width_ || image.height_ != height_) {
        PrintInfo("Incompatible image size!\n");
        return;
    }
    ImageCudaKernelCaller<VecType>::ShiftImageKernelCaller(
        *server_, *image.server(), dx, dy, with_holes);
}

template<typename VecType>
ImageCuda<VecType> ImageCuda<VecType>::Gaussian(GaussianKernelSize kernel,
                                                bool with_holes) {
    ImageCuda<VecType> dst;
    dst.Create(width_, height_);
    Gaussian(dst, kernel, with_holes);
    return dst;
}

template<typename VecType>
void ImageCuda<VecType>::Gaussian(ImageCuda<VecType> &image,
                                  GaussianKernelSize kernel,
                                  bool with_holes) {
    if (image.server() == nullptr) {
        image.Create(width_, height_);
    } else if (image.width_ != width_ || image.height_ != height_) {
        PrintInfo("Incompatible image size!\n");
        return;
    }
    ImageCudaKernelCaller<VecType>::GaussianImageKernelCaller(
        *server_, *image.server(), (int) kernel, with_holes);
}

template<typename VecType>
ImageCuda<VecType> ImageCuda<VecType>::Bilateral(GaussianKernelSize kernel,
                                                 float val_sigma,
                                                 bool with_holes) {
    ImageCuda<VecType> dst;
    dst.Create(width_, height_);
    Bilateral(dst, kernel, val_sigma, with_holes);
    return dst;
}

template<typename VecType>
void ImageCuda<VecType>::Bilateral(ImageCuda<VecType> &image,
                                   GaussianKernelSize kernel,
                                   float val_sigma,
                                   bool with_holes) {
    if (image.server() == nullptr) {
        image.Create(width_, height_);
    } else if (image.width_ != width_ || image.height_ != height_) {
        PrintInfo("Incompatible image size!\n");
        return;
    }
    ImageCudaKernelCaller<VecType>::BilateralImageKernelCaller(
        *server_, *image.server(),
        (int) kernel, val_sigma, with_holes);
}


template<typename VecType>
std::tuple<ImageCuda<typename VecType::VecTypef>,
           ImageCuda<typename VecType::VecTypef>> ImageCuda<VecType>::Sobel(
    bool with_holes) {
    ImageCuda<typename VecType::VecTypef> dx;
    ImageCuda<typename VecType::VecTypef> dy;
    dx.Create(width_, height_);
    dy.Create(width_, height_);
    Sobel(dx, dy, with_holes);
    return std::make_tuple(dx, dy);
}

template<typename VecType>
void ImageCuda<VecType>::Sobel(ImageCuda<typename VecType::VecTypef> &dx,
                               ImageCuda<typename VecType::VecTypef> &dy,
                               bool with_holes) {
    if (dx.server() == nullptr) {
        dx.Create(width_, height_);
    } else if (dx.width_ != width_ || dx.height_ != height_) {
        PrintError("Incompatible image size!\n");
        return;
    }

    if (dy.server() == nullptr) {
        dy.Create(width_, height_);
    } else if (dy.width_ != width_ || dy.height_ != height_) {
        PrintError("Incompatible image size!\n");
        return;
    }
    ImageCudaKernelCaller<VecType>::SobelImageKernelCaller(
        *server_, *dx.server(), *dy.server(), with_holes);
}

template<typename VecType>
ImageCuda<typename VecType::VecTypef> ImageCuda<VecType>::ToFloat(
    float scale, float offset) {
    ImageCuda<typename VecType::VecTypef> dst;
    dst.Create(width_, height_);
    ToFloat(dst, scale, offset);
    return dst;
}

template<typename VecType>
void ImageCuda<VecType>::ToFloat(ImageCuda<typename VecType::VecTypef> &image,
                                 float scale, float offset) {
    if (image.server() == nullptr) {
        image.Create(width_, height_);
    } else if (image.width_ != width_ || image.height_ != height_) {
        PrintInfo("Incompatible image size!\n");
        return;
    }
    ImageCudaKernelCaller<VecType>::ToFloatImageKernelCaller(
        *server_, *image.server(), scale, offset);
}
}