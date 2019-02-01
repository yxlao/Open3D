//
// Created by wei on 11/9/18.
//

#pragma once

#include "ImageCuda.h"
#include <Cuda/Common/LinearAlgebraCuda.h>
#include <Cuda/Common/UtilsCuda.h>

#include <cuda_runtime.h>
#include <driver_types.h>
#include <vector_types.h>

#include <Core/Core.h>

namespace open3d {

namespace cuda {
/**
 * Client end
 */
template<typename VecType>
ImageCuda<VecType>::ImageCuda()
    : width_(-1), height_(-1), pitch_(-1), device_(nullptr) {
#ifdef HOST_DEBUG_MONITOR_LIFECYCLE
    PrintInfo("Default ImageCuda constructor.\n");
#endif
}

template<typename VecType>
ImageCuda<VecType>::ImageCuda(const ImageCuda<VecType> &other) {
#ifdef HOST_DEBUG_MONITOR_LIFECYCLE
    PrintInfo("ImageCuda copy constructor.\n");
#endif
    device_ = other.device_;

    width_ = other.width_;
    height_ = other.height_;
    pitch_ = other.pitch_;

#ifdef HOST_DEBUG_MONITOR_LIFECYCLE
    PrintInfo("Ref count after copy construction: %d\n", device_.use_count());
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

        device_ = other.device_;
        width_ = other.width_;
        height_ = other.height_;
        pitch_ = other.pitch_;

#ifdef HOST_DEBUG_MONITOR_LIFECYCLE
        PrintInfo("Ref count after copy construction: %d\n", device_.use_count());
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
bool ImageCuda<VecType>::Create(int width, int height) {
    assert(width > 0 && height > 0);

    if (device_ != nullptr) {
        if (width_ != width || height_ != height) {
            PrintError("[ImageCuda] Incompatible image size, "
                       "@Create aborted.\n");
            return false;
        }
        return true;
    }

#ifdef HOST_DEBUG_MONITOR_LIFECYCLE
    PrintInfo("Creating.\n");
#endif
    device_ = std::make_shared<ImageCudaDevice<VecType>>();

    width_ = width;
    height_ = height;
    size_t pitch_size_t = 0;
    CheckCuda(cudaMallocPitch(&device_->data_, &pitch_size_t,
                              sizeof(VecType) * width_, (size_t) height_));
    pitch_ = (int) pitch_size_t;

    UpdateDevice();
    return true;
}

template<typename VecType>
void ImageCuda<VecType>::Release() {
#ifdef HOST_DEBUG_MONITOR_LIFECYCLE
    if (device_ != nullptr) {
        PrintInfo("ref count before releasing: %d\n", device_.use_count());
    }
#endif

    if (device_ != nullptr && device_.use_count() == 1) {
        CheckCuda(cudaFree(device_->data_));
    }

    device_ = nullptr;
    width_ = -1;
    height_ = -1;
    pitch_ = -1;
}

template<typename VecType>
void ImageCuda<VecType>::UpdateDevice() {
    assert(device_ != nullptr);
    device_->width_ = width_;
    device_->height_ = height_;
    device_->pitch_ = pitch_;
}

template<typename VecType>
void ImageCuda<VecType>::CopyFrom(const ImageCuda<VecType> &other) {
    if (this == &other) return;

    bool success = Create(other.width_, other.height_);
    if (!success) return;

    CheckCuda(cudaMemcpy2D(device_->data_, (size_t) pitch_,
                           other.device_->data(), (size_t) other.pitch_,
                           sizeof(VecType) * width_, (size_t) height_,
                           cudaMemcpyDeviceToDevice));
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
        PrintWarning("[ImageCuda] Unsupported format %d,"
                     "@Upload aborted.\n");
        return;
    }

    bool success = Create(image.width_, image.height_);
    if (!success) return;

    CheckCuda(cudaMemcpy2D(device_->data_,
                           (size_t) pitch_,
                           image.data_.data(),
                           (size_t) image.BytesPerLine(),
                           sizeof(VecType) * image.width_,
                           (size_t) image.height_,
                           cudaMemcpyHostToDevice));
}

template<typename VecType>
std::shared_ptr<Image> ImageCuda<VecType>::DownloadImage() {
    std::shared_ptr<Image> image = std::make_shared<Image>();
    if (device_ == nullptr) {
        PrintWarning("[ImageCuda] not initialized, "
                     "@DownloadImage aborted.\n");
        return image;
    }

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
        PrintWarning("[ImageCuda] Unsupported format %d,"
                     "@DownloadImage aborted.\n");
        return image;
    }

    CheckCuda(cudaMemcpy2D(image->data_.data(), (size_t) image->BytesPerLine(),
                           device_->data_, (size_t) pitch_,
                           sizeof(VecType) * width_, (size_t) height_,
                           cudaMemcpyDeviceToHost));
    return image;
}

template<typename VecType>
ImageCuda<VecType> ImageCuda<VecType>::Downsample(DownsampleMethod method) {
    ImageCuda<VecType> dst;
    dst.Create(width_ >> 1, height_ >> 1);
    Downsample(dst, method);
    return dst;
}

template<typename VecType>
void ImageCuda<VecType>::Downsample(ImageCuda<VecType> &image,
                                    DownsampleMethod method) {
    bool success = image.Create(width_ >> 1, height_ >> 1);
    if (success) {
        ImageCudaKernelCaller<VecType>::Downsample(*this, image, method);
    }
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
    bool success = image.Create(width_, height_);
    if (success) {
        ImageCudaKernelCaller<VecType>::Shift(*this, image, dx, dy, with_holes);
    }
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
    bool success = image.Create(width_, height_);
    if (success) {
        ImageCudaKernelCaller<VecType>::Gaussian(
            *this, image, (int) kernel, with_holes);
    }
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
    bool success = image.Create(width_, height_);
    if (success) {
        ImageCudaKernelCaller<VecType>::Bilateral(
            *this, image, (int) kernel, val_sigma, with_holes);
    }
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
    bool success = true;
    success &= dx.Create(width_, height_);
    success &= dy.Create(width_, height_);
    if (success) {
        ImageCudaKernelCaller<VecType>::Sobel(*this, dx, dy, with_holes);
    }
}

template<typename VecType>
ImageCuda<typename VecType::VecTypef> ImageCuda<VecType>::ConvertToFloat(
    float scale, float offset) {
    ImageCuda<typename VecType::VecTypef> dst;
    dst.Create(width_, height_);
    ConvertToFloat(dst, scale, offset);
    return dst;
}

template<typename VecType>
void ImageCuda<VecType>::ConvertToFloat(
    ImageCuda<typename VecType::VecTypef> &image, float scale, float offset) {
    bool success = image.Create(width_, height_);
    if (success) {
        ImageCudaKernelCaller<VecType>::ConvertToFloat(
            *this, image, scale, offset);
    }
}

template<typename VecType>
ImageCuda<Vector1f> ImageCuda<VecType>::ConvertRGBToIntensity() {
    assert(typeid(VecType) == typeid(Vector3b));
    ImageCuda<Vector1f> dst;
    dst.Create(width_, height_);
    ConvertRGBToIntensity(dst);
    return dst;
}

template<typename VecType>
void ImageCuda<VecType>::ConvertRGBToIntensity(ImageCuda<Vector1f> &image) {
    bool success = image.Create(width_, height_);
    if (success) {
        ImageCudaKernelCaller<VecType>::ConvertRGBToIntensity(*this, image);
    }
}

/** Legacy **/
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
        PrintWarning("[ImageCuda] Unsupported format %d,"
                     "@Upload aborted.\n");
        return;
    }

    bool success = Create(m.cols, m.rows);
    if (!success) return;

    CheckCuda(cudaMemcpy2D(device_->data_, (size_t) pitch_,
                           m.data, m.step,
                           sizeof(VecType) * m.cols, (size_t) m.rows,
                           cudaMemcpyHostToDevice));
}

template<typename VecType>
cv::Mat ImageCuda<VecType>::DownloadMat() {
    cv::Mat m;
    if (device_ == nullptr) {
        PrintWarning("[ImageCuda] Not initialized, "
                     "@DownloadMat aborted.\n");
        return m;
    }

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
        PrintWarning("[ImageCuda] Unsupported format %d,"
                     "@DownloadMat aborted.\n");
        return m;
    }

    CheckCuda(cudaMemcpy2D(m.data, m.step, device_->data_, (size_t) pitch_,
                           sizeof(VecType) * width_, (size_t) height_,
                           cudaMemcpyDeviceToHost));
    return m;
}
} // cuda
} // open3d