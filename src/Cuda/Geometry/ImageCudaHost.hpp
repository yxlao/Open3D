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

namespace open3d {

namespace cuda {
/**
 * Client end
 */
template<typename Scalar, size_t Channel>
ImageCuda<Scalar, Channel>::ImageCuda()
    : width_(-1), height_(-1), pitch_(-1), device_(nullptr) {
#ifdef HOST_DEBUG_MONITOR_LIFECYCLE
    PrintInfo("Default ImageCuda constructor.\n");
#endif
}

template<typename Scalar, size_t Channel>
ImageCuda<Scalar, Channel>::ImageCuda(const ImageCuda<Scalar, Channel> &other) {
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

template<typename Scalar, size_t Channel>
ImageCuda<Scalar, Channel>::ImageCuda(int width, int height) {
    Create(width, height);
}

template<typename Scalar, size_t Channel>
ImageCuda<Scalar, Channel> &ImageCuda<Scalar,
                                      Channel>::operator=(const ImageCuda<Scalar,
                                                                          Channel> &other) {
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

template<typename Scalar, size_t Channel>
ImageCuda<Scalar, Channel>::~ImageCuda() {
#ifdef HOST_DEBUG_MONITOR_LIFECYCLE
    PrintInfo("Destructor.\n");
#endif
    Release();
}

template<typename Scalar, size_t Channel>
bool ImageCuda<Scalar, Channel>::Create(int width, int height) {
    assert(width > 0 && height > 0);

    if (device_ != nullptr) {
        if (width_ != width || height_ != height) {
            utility::PrintError("[ImageCuda] Incompatible image size, "
                                "@Create aborted.\n");
            return false;
        }
        return true;
    }

#ifdef HOST_DEBUG_MONITOR_LIFECYCLE
    PrintInfo("Creating.\n");
#endif
    device_ = std::make_shared<ImageCudaDevice<Scalar, Channel>>();

    width_ = width;
    height_ = height;
    size_t pitch_size_t = 0;
    CheckCuda(cudaMallocPitch(&device_->data_,
                              &pitch_size_t,
                              sizeof(VectorCuda<Scalar, Channel>) * width_,
                              (size_t) height_));
    pitch_ = (int) pitch_size_t;

    UpdateDevice();
    return true;
}

template<typename Scalar, size_t Channel>
void ImageCuda<Scalar, Channel>::Release() {
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

template<typename Scalar, size_t Channel>
void ImageCuda<Scalar, Channel>::UpdateDevice() {
    assert(device_ != nullptr);
    device_->width_ = width_;
    device_->height_ = height_;
    device_->pitch_ = pitch_;
}

template<typename Scalar, size_t Channel>
void ImageCuda<Scalar, Channel>::CopyFrom(const ImageCuda<Scalar,
                                                          Channel> &other) {
    if (this == &other) return;

    bool success = Create(other.width_, other.height_);
    if (!success) return;

    CheckCuda(cudaMemcpy2D(device_->data_, (size_t) pitch_,
                           other.device_->data(), (size_t) other.pitch_,
                           sizeof(VectorCuda<Scalar, Channel>) * width_,
                           (size_t) height_,
                           cudaMemcpyDeviceToDevice));
}

template<typename Scalar, size_t Channel>
void ImageCuda<Scalar, Channel>::Upload(geometry::Image &image) {
    assert(image.width_ > 0 && image.height_ > 0);

    /* Type checking */
    if (typeid(Scalar) == typeid(int) && Channel == 1) {
        assert(image.bytes_per_channel_ == 4 && image.num_of_channels_ == 1);
    } else if (typeid(Scalar) == typeid(ushort) && Channel == 1) {
        assert(image.bytes_per_channel_ == 2 && image.num_of_channels_ == 1);
    } else if (typeid(Scalar) == typeid(uchar) && Channel == 3) {
        assert(image.bytes_per_channel_ == 1 && image.num_of_channels_ == 3);
    } else if (typeid(Scalar) == typeid(uchar) && Channel == 1) {
        assert(image.bytes_per_channel_ == 1 && image.num_of_channels_ == 1);
    } else if (typeid(Scalar) == typeid(float) && Channel == 3) {
        assert(image.bytes_per_channel_ == 4 && image.num_of_channels_ == 3);
    } else if (typeid(Scalar) == typeid(float) && Channel == 1) {
        assert(image.bytes_per_channel_ == 4 && image.num_of_channels_ == 1);
    } else {
        utility::PrintWarning("[ImageCuda] Unsupported format %d,"
                              "@Upload aborted.\n");
        return;
    }

    bool success = Create(image.width_, image.height_);
    if (!success) return;

    CheckCuda(cudaMemcpy2D(device_->data_,
                           (size_t) pitch_,
                           image.data_.data(),
                           (size_t) image.BytesPerLine(),
                           sizeof(VectorCuda<Scalar, Channel>) * image.width_,
                           (size_t) image.height_,
                           cudaMemcpyHostToDevice));
}

template<typename Scalar, size_t Channel>
std::shared_ptr<geometry::Image> ImageCuda<Scalar, Channel>::DownloadImage() {
    std::shared_ptr<geometry::Image> image =
        std::make_shared<geometry::Image>();
    if (device_ == nullptr) {
        utility::PrintWarning("[ImageCuda] not initialized, "
                              "@DownloadImage aborted.\n");
        return image;
    }

    if (typeid(Scalar) == typeid(int) && Channel == 1) {
        image->PrepareImage(width_, height_, 1, 4);
    } else if (typeid(Scalar) == typeid(ushort) && Channel == 1) {
        image->PrepareImage(width_, height_, 1, 2);
    } else if (typeid(Scalar) == typeid(uchar) && Channel == 3) {
        image->PrepareImage(width_, height_, 3, 1);
    } else if (typeid(Scalar) == typeid(uchar) && Channel == 1) {
        image->PrepareImage(width_, height_, 1, 1);
    } else if (typeid(Scalar) == typeid(float) && Channel == 3) {
        image->PrepareImage(width_, height_, 3, 4);
    } else if (typeid(Scalar) == typeid(float) && Channel == 1) {
        image->PrepareImage(width_, height_, 1, 4);
    } else {
        utility::PrintWarning("[ImageCuda] Unsupported format %d,"
                              "@DownloadImage aborted.\n");
        return image;
    }

    CheckCuda(cudaMemcpy2D(image->data_.data(), (size_t) image->BytesPerLine(),
                           device_->data_, (size_t) pitch_,
                           sizeof(VectorCuda<Scalar, Channel>) * width_,
                           (size_t) height_,
                           cudaMemcpyDeviceToHost));
    return image;
}

template<typename Scalar, size_t Channel>
ImageCuda<Scalar, Channel> ImageCuda<Scalar, Channel>::Downsample(
    DownsampleMethod method) {
    ImageCuda<Scalar, Channel> dst;
    dst.Create(width_ >> 1, height_ >> 1);
    Downsample(dst, method);
    return dst;
}

template<typename Scalar, size_t Channel>
void ImageCuda<Scalar, Channel>::Downsample(ImageCuda<Scalar, Channel> &image,
                                            DownsampleMethod method) {
    bool success = image.Create(width_ >> 1, height_ >> 1);
    if (success) {
        ImageCudaKernelCaller<Scalar, Channel>::Downsample(
            *this, image, method);
    }
}

template<typename Scalar, size_t Channel>
ImageCuda<Scalar, Channel> ImageCuda<Scalar, Channel>::Shift(
    float dx, float dy) {
    ImageCuda<Scalar, Channel> dst;
    dst.Create(width_, height_);
    Shift(dst, dx, dy);
    return dst;
}

template<typename Scalar, size_t Channel>
void ImageCuda<Scalar, Channel>::Shift(
    ImageCuda<Scalar, Channel> &image, float dx, float dy) {
    bool success = image.Create(width_, height_);
    if (success) {
        ImageCudaKernelCaller<Scalar, Channel>::Shift(
            *this, image, dx, dy);
    }
}

template<typename Scalar, size_t Channel>
ImageCuda<Scalar, Channel> ImageCuda<Scalar, Channel>::Gaussian(
    GaussianKernelSize kernel) {
    ImageCuda<Scalar, Channel> dst;
    dst.Create(width_, height_);
    Gaussian(dst, kernel);
    return dst;
}

template<typename Scalar, size_t Channel>
void ImageCuda<Scalar, Channel>::Gaussian(
    ImageCuda<Scalar, Channel> &image,
    GaussianKernelSize kernel) {
    bool success = image.Create(width_, height_);
    if (success) {
        ImageCudaKernelCaller<Scalar, Channel>::Gaussian(
            *this, image, (int) kernel);
    }
}

template<typename Scalar, size_t Channel>
ImageCuda<Scalar, Channel> ImageCuda<Scalar, Channel>::Bilateral(
    GaussianKernelSize kernel, float val_sigma) {
    ImageCuda<Scalar, Channel> dst;
    dst.Create(width_, height_);
    Bilateral(dst, kernel, val_sigma);
    return dst;
}

template<typename Scalar, size_t Channel>
void ImageCuda<Scalar, Channel>::Bilateral(
    ImageCuda<Scalar, Channel> &image, GaussianKernelSize kernel,
    float val_sigma) {
    bool success = image.Create(width_, height_);
    if (success) {
        ImageCudaKernelCaller<Scalar, Channel>::Bilateral(
            *this, image, (int) kernel, val_sigma);
    }
}

template<typename Scalar, size_t Channel>
std::tuple<ImageCuda<float, Channel>, ImageCuda<float, Channel>>
ImageCuda<Scalar, Channel>::Sobel() {
    ImageCuda<float, Channel> dx;
    ImageCuda<float, Channel> dy;
    dx.Create(width_, height_);
    dy.Create(width_, height_);
    Sobel(dx, dy);
    return std::make_tuple(dx, dy);
}

template<typename Scalar, size_t Channel>
void ImageCuda<Scalar, Channel>::Sobel(
    ImageCuda<float, Channel> &dx,
    ImageCuda<float, Channel> &dy) {
    bool success = true;
    success &= dx.Create(width_, height_);
    success &= dy.Create(width_, height_);
    if (success) {
        ImageCudaKernelCaller<Scalar, Channel>::Sobel(*this, dx, dy);
    }
}

template<typename Scalar, size_t Channel>
ImageCuda<float, Channel> ImageCuda<Scalar, Channel>::ConvertToFloat(
    float scale, float offset) {
    ImageCuda<float, Channel> dst;
    dst.Create(width_, height_);
    ConvertToFloat(dst, scale, offset);
    return dst;
}

template<typename Scalar, size_t Channel>
void ImageCuda<Scalar, Channel>::ConvertToFloat(
    ImageCuda<float, Channel> &image, float scale, float offset) {
    bool success = image.Create(width_, height_);
    if (success) {
        ImageCudaKernelCaller<Scalar, Channel>::ConvertToFloat(
            *this, image, scale, offset);
    }
}

template<typename Scalar, size_t Channel>
ImageCuda<float, 1> ImageCuda<Scalar, Channel>::ConvertRGBToIntensity() {
    assert(typeid(Scalar) == typeid(uchar) && Channel == 3);
    ImageCuda<float, 1> dst;
    dst.Create(width_, height_);
    ConvertRGBToIntensity(dst);
    return dst;
}

template<typename Scalar, size_t Channel>
void ImageCuda<Scalar, Channel>::ConvertRGBToIntensity(
    ImageCuda<float, 1> &image) {
    assert(typeid(Scalar) == typeid(uchar) && Channel == 3);
    bool success = image.Create(width_, height_);
    if (success) {
        ImageCudaKernelCaller<Scalar, Channel>::ConvertRGBToIntensity(
            *this, image);
    }
}

/** Legacy **/
template<typename Scalar, size_t Channel>
void ImageCuda<Scalar, Channel>::Upload(cv::Mat &m) {
    assert(m.rows > 0 && m.cols > 0);

    /* Type checking */
    if (typeid(Scalar) == typeid(int) && Channel == 1) {
        assert(m.type() == CV_32SC1);
    } else if (typeid(Scalar) == typeid(ushort) && Channel == 1) {
        assert(m.type() == CV_16UC1);
    } else if (typeid(Scalar) == typeid(uchar) && Channel == 4) {
        assert(m.type() == CV_8UC4);
    } else if (typeid(Scalar) == typeid(uchar) && Channel == 3) {
        assert(m.type() == CV_8UC3);
    } else if (typeid(Scalar) == typeid(uchar) && Channel == 1) {
        assert(m.type() == CV_8UC1);
    } else if (typeid(Scalar) == typeid(float) && Channel == 4) {
        assert(m.type() == CV_32FC4);
    } else if (typeid(Scalar) == typeid(float) && Channel == 3) {
        assert(m.type() == CV_32FC3);
    } else if (typeid(Scalar) == typeid(float) && Channel == 1) {
        assert(m.type() == CV_32FC1);
    } else {
        utility::PrintWarning("[ImageCuda] Unsupported format %d,"
                              "@Upload aborted.\n", m.type());
        return;
    }

    bool success = Create(m.cols, m.rows);
    if (!success) return;

    CheckCuda(cudaMemcpy2D(device_->data_, (size_t) pitch_,
                           m.data, m.step,
                           sizeof(VectorCuda<Scalar, Channel>) * m.cols,
                           (size_t) m.rows,
                           cudaMemcpyHostToDevice));
}

template<typename Scalar, size_t Channel>
cv::Mat ImageCuda<Scalar, Channel>::DownloadMat() {
    cv::Mat m;
    if (device_ == nullptr) {
        utility::PrintWarning("[ImageCuda] Not initialized, "
                              "@DownloadMat aborted.\n");
        return m;
    }

    if (typeid(Scalar) == typeid(int) && Channel == 1) {
        m = cv::Mat(height_, width_, CV_32SC1);
    } else if (typeid(Scalar) == typeid(ushort) && Channel == 1) {
        m = cv::Mat(height_, width_, CV_16UC1);
    } else if (typeid(Scalar) == typeid(uchar) && Channel == 4) {
        m = cv::Mat(height_, width_, CV_8UC4);
    } else if (typeid(Scalar) == typeid(uchar) && Channel == 3) {
        m = cv::Mat(height_, width_, CV_8UC3);
    } else if (typeid(Scalar) == typeid(uchar) && Channel == 1) {
        m = cv::Mat(height_, width_, CV_8UC1);
    } else if (typeid(Scalar) == typeid(float) && Channel == 4) {
        m = cv::Mat(height_, width_, CV_32FC4);
    } else if (typeid(Scalar) == typeid(float) && Channel == 3) {
        m = cv::Mat(height_, width_, CV_32FC3);
    } else if (typeid(Scalar) == typeid(float) && Channel == 1) {
        m = cv::Mat(height_, width_, CV_32FC1);
    } else {
        utility::PrintWarning("[ImageCuda] Unsupported format %d,"
                              "@DownloadMat aborted.\n");
        return m;
    }

    CheckCuda(cudaMemcpy2D(m.data, m.step, device_->data_, (size_t) pitch_,
                           sizeof(VectorCuda<Scalar, Channel>) * width_,
                           (size_t) height_,
                           cudaMemcpyDeviceToHost));
    return m;
}
} // cuda
} // open3d