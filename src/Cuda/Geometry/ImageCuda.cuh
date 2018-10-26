/**
 * Created by wei on 18-4-9.
 */

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
 * Server end
 */
/**
 * assert will make these functions super slow
 */
template<typename VecType>
__device__
VecType &ImageCudaServer<VecType>::get(int x, int y) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);
#endif
    VecType *value = (VecType *) ((char *) data_ + y * pitch_) + x;
    return (*value);
}

template<typename VecType>
__device__
VecType &ImageCudaServer<VecType>::operator()(int x, int y) {
    return get(x, y);
}

/**
 * Naive interpolation without considering holes in depth images.
 */
template<typename VecType>
__device__
VecType ImageCudaServer<VecType>::get_interp(float x, float y) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(x >= 0 && x < width_ - 1);
    assert(y >= 0 && y < height_ - 1);
#endif
    int x0 = (int) floor(x), y0 = (int) floor(y);
    float a = x - x0, b = y - y0;
    return VecType::FromVectorf(
        (1 - a) * (1 - b) * get(x0, y0).ToVectorf()
            + (1 - a) * b * get(x0, y0 + 1).ToVectorf()
            + a * b * get(x0 + 1, y0 + 1).ToVectorf()
            + a * (1 - b) * get(x0 + 1, y0).ToVectorf()
    );
}

/** SO many if. If it is slow, fall back to get **/
template<typename VecType>
__device__
VecType ImageCudaServer<VecType>::get_interp_with_holes(float x, float y) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(x >= 0 && x < width_ - 1);
    assert(y >= 0 && y < height_ - 1);
#endif

    int x0 = (int) floor(x), y0 = (int) floor(y);
    float a = x - x0, b = y - y0;

    float sum_w = 0;
    auto sum_val = VecType::VecTypef(0);
    auto zero = VecType::VecTypef(0);

    auto val = get(x0, y0).ToVectorf();
    float w = (1 - a) * (1 - b);
    sum_val += w * val;
    sum_w += (val == zero) ? 0 : w;

    val = get(x0, y0 + 1).ToVectorf();
    w = (1 - a) * b;
    sum_val += w * val;
    sum_w += (val == zero) ? 0 : w;

    val = get(x0 + 1, y0 + 1).ToVectorf();
    w = a * b;
    sum_val += w * val;
    sum_w += (val == zero) ? 0 : w;

    val = get(x0 + 1, y0).ToVectorf();
    w = a * (1 - b);
    sum_val += w * val;
    sum_w += (val == zero) ? 0 : w;

    return sum_w == 0 ? VecType(0) : VecType::FromVectorf(sum_val / sum_w);
}

template<typename VecType>
__device__
VecType ImageCudaServer<VecType>::BoxFilter2x2(int x, int y) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);
#endif

    int xp1 = min(width_ - 1, x + 1);
    int yp1 = min(height_ - 1, y + 1);

    auto sum_val = VecType::Vectorf();
    sum_val += get(x, y).ToVectorf();
    sum_val += get(x, yp1).ToVectorf();
    sum_val += get(xp1, y).ToVectorf();
    sum_val += get(xp1, yp1).ToVectorf();

    sum_val *= 0.25f;
    return VecType::FromVectorf(sum_val);
}

template<typename VecType>
__device__
VecType ImageCudaServer<VecType>::BoxFilter2x2WithHoles(int x, int y) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);
#endif

    int xp1 = min(width_ - 1, x + 1);
    int yp1 = min(height_ - 1, y + 1);

    auto sum_val = VecType::Vectorf();
    float cnt = 0.0;
    VecType val;
    VecType zero = VecType(0);

    val = get(x, y);
    sum_val += val.ToVectorf();
    cnt += (val == zero) ? 0.0f : 1.0f;

    val = get(x, yp1);
    sum_val += val.ToVectorf();
    cnt += (val == zero) ? 0.0f : 1.0f;

    val = get(xp1, y);
    sum_val += val.ToVectorf();
    cnt += (val == zero) ? 0.0f : 1.0f;

    val = get(xp1, yp1);
    sum_val += val.ToVectorf();
    cnt += (val == zero) ? 0.0f : 1.0f;

    return cnt == 0 ? VecType(0) : VecType::FromVectorf(sum_val / cnt);
}

template<typename VecType>
__device__
VecType ImageCudaServer<VecType>::GaussianFilter(int x, int y, int kernel_idx) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);
#endif

    static const int kernel_sizes[3] = {3, 5, 7};
    static const float gaussian_weights[3][7] = {
        {0.25f, 0.5f, 0.25f, 0, 0, 0, 0},
        {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f, 0, 0},
        {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}
    };

    const int kernel_size = kernel_sizes[kernel_idx];
    const int kernel_size_2 = kernel_size >> 1;
    const float *kernel = gaussian_weights[kernel_idx];

    int x_min = max(0, x - kernel_size_2);
    int y_min = max(0, y - kernel_size_2);
    int x_max = min(width_ - 1, x + kernel_size_2);
    int y_max = min(height_ - 1, y + kernel_size_2);

    auto sum_val = VecType::Vectorf();
    float sum_weight = 0;

    for (int xx = x_min; xx <= x_max; ++xx) {
        for (int yy = y_min; yy <= y_max; ++yy) {
            auto val = get(xx, yy).ToVectorf();
            float weight = kernel[abs(xx - x)] * kernel[abs(yy - y)];
            sum_val += val * weight;
            sum_weight += weight;
        }
    }

    sum_val /= sum_weight;
    return VecType::FromVectorf(sum_val);
}

template<typename VecType>
__device__
VecType ImageCudaServer<VecType>::GaussianFilterWithHoles(
    int x, int y, int kernel_idx) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);
#endif

    static const int kernel_sizes[3] = {3, 5, 7};
    static const float gaussian_weights[3][7] = {
        {0.25f, 0.5f, 0.25f, 0, 0, 0, 0},
        {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f, 0, 0},
        {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}
    };

    /** If it is already a hole, leave it alone **/
    VecType zero = VecType(0);
    if (get(x, y) == zero) return zero;

    const int kernel_size = kernel_sizes[kernel_idx];
    const int kernel_size_2 = kernel_size >> 1;
    const float *kernel = gaussian_weights[kernel_idx];

    int x_min = max(0, x - kernel_size_2);
    int y_min = max(0, y - kernel_size_2);
    int x_max = min(width_ - 1, x + kernel_size_2);
    int y_max = min(height_ - 1, y + kernel_size_2);

    auto sum_val = VecType::Vectorf();
    float sum_weight = 0;

    for (int xx = x_min; xx <= x_max; ++xx) {
        for (int yy = y_min; yy <= y_max; ++yy) {
            VecType val = get(xx, yy);
            auto valf = val.ToVectorf();
            float weight = kernel[abs(xx - x)] * kernel[abs(yy - y)];
            sum_val += valf * weight;
            sum_weight += (val == zero) ? 0 : weight;
        }
    }

    /** Center is not zero, so sum_weight > 0 **/
    sum_val /= sum_weight;
    return VecType::FromVectorf(sum_val);
}

template<typename VecType>
__device__
VecType ImageCudaServer<VecType>::BilateralFilter(
    int x, int y, int kernel_idx, float val_sigma) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);
#endif

    static const int kernel_sizes[3] = {3, 5, 7};
    static const float gaussian_weights[3][7] = {
        {0.25f, 0.5f, 0.25f, 0, 0, 0, 0},
        {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f, 0, 0},
        {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}
    };

    const int kernel_size = kernel_sizes[kernel_idx];
    const int kernel_size_2 = kernel_size >> 1;
    const float *kernel = gaussian_weights[kernel_idx];

    int x_min = max(0, x - kernel_size_2);
    int y_min = max(0, y - kernel_size_2);
    int x_max = min(width_ - 1, x + kernel_size_2);
    int y_max = min(height_ - 1, y + kernel_size_2);

    auto center_val = get(x, y).ToVectorf();
    auto sum_val = VecType::Vectorf();
    float sum_weight = 0;

    for (int xx = x_min; xx <= x_max; ++xx) {
        for (int yy = y_min; yy <= y_max; ++yy) {
            auto val = get(xx, yy).ToVectorf();
            float weight = kernel[abs(xx - x)] * kernel[abs(yy - y)];
            float value_diff = (val - center_val).norm() / val_sigma;
            weight *= expf(-value_diff);

            sum_val += val * weight;
            sum_weight += weight;
        }
    }

    sum_val /= sum_weight;
    return VecType::FromVectorf(sum_val);
}

template<typename VecType>
__device__
VecType ImageCudaServer<VecType>::BilateralFilterWithHoles(
    int x, int y, int kernel_idx, float val_sigma) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);
#endif

    static const int kernel_sizes[3] = {3, 5, 7};
    static const float gaussian_weights[3][7] = {
        {0.25f, 0.5f, 0.25f, 0, 0, 0, 0},
        {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f, 0, 0},
        {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}
    };

    VecType zero = VecType(0);
    if (get(x, y) == zero) return zero;

    const int kernel_size = kernel_sizes[kernel_idx];
    const int kernel_size_2 = kernel_size >> 1;
    const float *kernel = gaussian_weights[kernel_idx];

    int x_min = max(0, x - kernel_size_2);
    int y_min = max(0, y - kernel_size_2);
    int x_max = min(width_ - 1, x + kernel_size_2);
    int y_max = min(height_ - 1, y + kernel_size_2);

    auto center_valf = get(x, y).ToVectorf();
    auto sum_val = VecType::Vectorf();
    float sum_weight = 0;

    for (int xx = x_min; xx <= x_max; ++xx) {
        for (int yy = y_min; yy <= y_max; ++yy) {
            auto val = get(xx, yy);
            auto valf = val.ToVectorf();
            float weight = kernel[abs(xx - x)] * kernel[abs(yy - y)];
            float value_diff = (valf - center_valf).norm() / val_sigma;
            weight *= expf(-value_diff * value_diff);

            sum_val += valf * weight;
            sum_weight += (val == zero) ? 0 : weight;
        }
    }

    sum_val /= sum_weight;
    return VecType::FromVectorf(sum_val);
}

template<typename VecType>
__device__
ImageCudaServer<VecType>::Grad
ImageCudaServer<VecType>::Sobel(int x, int y) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(x >= 1 && x < width_ - 1);
    assert(y >= 1 && y < height_ - 1);
#endif

    auto Iumvm = get(x - 1, y - 1).ToVectorf();
    auto Iumv0 = get(x - 1, y).ToVectorf();
    auto Iumvp = get(x - 1, y + 1).ToVectorf();
    auto Iu0vm = get(x, y - 1).ToVectorf();
    auto Iu0vp = get(x, y + 1).ToVectorf();
    auto Iupvm = get(x + 1, y - 1).ToVectorf();
    auto Iupv0 = get(x + 1, y).ToVectorf();
    auto Iupvp = get(x + 1, y + 1).ToVectorf();

    return {
        (Iupvm - Iumvm) + (Iupv0 - Iumv0) * 2 + (Iupvp - Iumvp),
        (Iumvp - Iumvm) + (Iu0vp - Iu0vm) * 2 + (Iupvp - Iupvm)
    };
}

/**
 * It is a little bit strict ...
 * It will filter out all the (geometric) edges.
 * If it does not work, fall back to Sobel
 */
template<typename VecType>
__device__
ImageCudaServer<VecType>::Grad
ImageCudaServer<VecType>::SobelWithHoles(int x, int y) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(x >= 1 && x < width_ - 1);
    assert(y >= 1 && y < height_ - 1);
#endif

    auto zero = VecType::VecTypef(0);
    auto Iumvm = get(x - 1, y - 1).ToVectorf();
    auto Iumv0 = get(x - 1, y).ToVectorf();
    auto Iumvp = get(x - 1, y + 1).ToVectorf();
    auto Iu0vm = get(x, y - 1).ToVectorf();
    auto Iu0vp = get(x, y + 1).ToVectorf();
    auto Iupvm = get(x + 1, y - 1).ToVectorf();
    auto Iupv0 = get(x + 1, y).ToVectorf();
    auto Iupvp = get(x + 1, y + 1).ToVectorf();

    bool mask_corner = (Iumvm != zero) && (Iumvp != zero)
        && (Iupvm != zero) && (Iupvp != zero);
    bool mask_dx = mask_corner && (Iupv0 != zero) && (Iumv0 != zero);
    bool mask_dy = mask_corner && (Iu0vp != zero) && (Iu0vm != zero);
    return {
        mask_dx ?
        (Iupvm - Iumvm) + (Iupv0 - Iumv0) * 2 + (Iupvp - Iumvp) : zero,
        mask_dy ?
        (Iumvp - Iumvm) + (Iu0vp - Iu0vm) * 2 + (Iupvp - Iupvm) : zero
    };
}

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

    width_ = other.width();
    height_ = other.height();
    pitch_ = other.pitch();

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
        width_ = other.width();
        height_ = other.height();
        pitch_ = other.pitch();

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
void ImageCuda<VecType>::Resize(int width, int height) {
    assert(width > 0 && height > 0);
    if (width == width_ && height == height_) {
        PrintWarning("No need to resize!\n");
        return;
    }

    Release();
    Create(width, height);
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
    CheckCuda(cudaMallocPitch((void **) &(server_->data_), &pitch_size_t,
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
void ImageCuda<VecType>::CopyTo(ImageCuda<VecType> &other) const {
    if (this == &other) return;

    if (other.server() == nullptr) {
        other.Create(width_, height_);
    }

    if (other.width() != width_ || other.height() != height_) {
        PrintError("Incompatible image size!\n");
        return;
    }

    CheckCuda(cudaMemcpy2D(other.server()->data(), other.pitch(),
                           server_->data_, pitch_,
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
cv::Mat ImageCuda<VecType>::Download() {
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
    } else if (image.width() != width_ / 2 || image.height() != height_ / 2) {
        PrintError("Incompatible image size!\n");
        return;
    }

    const dim3 blocks(DIV_CEILING(image.width(), THREAD_2D_UNIT),
                      DIV_CEILING(image.height(), THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    DownsampleImageKernel << < blocks, threads >> > (
        *server_, *(image.server()), method);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
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
    } else if (dx.width() != width_ || dx.height() != height_) {
        PrintError("Incompatible image size!\n");
        return;
    }

    if (dy.server() == nullptr) {
        dy.Create(width_, height_);
    } else if (dy.width() != width_ || dy.height() != height_) {
        PrintError("Incompatible image size!\n");
        return;
    }

    const dim3 blocks(DIV_CEILING(width_, THREAD_2D_UNIT),
                      DIV_CEILING(height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    SobelImageKernel << < blocks, threads >> > (
        *server_, *(dx.server()), *(dy.server()), with_holes);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());

    return;
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
void ImageCuda<VecType>::Shift(ImageCuda<VecType> &image, float dx, float dy,
                               bool with_holes) {
    if (image.server() == nullptr) {
        image.Create(width_, height_);
    } else if (image.width() != width_ || image.height() != height_) {
        PrintInfo("Incompatible image size!\n");
        return;
    }

    const dim3 blocks(DIV_CEILING(width_, THREAD_2D_UNIT),
                      DIV_CEILING(height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    ShiftImageKernel << < blocks, threads >> > (
        *server_, *(image.server()), dx, dy, with_holes);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
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
    } else if (image.width() != width_ || image.height() != height_) {
        PrintInfo("Incompatible image size!\n");
        return;
    }

    const dim3 blocks(DIV_CEILING(width_, THREAD_2D_UNIT),
                      DIV_CEILING(height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    GaussianImageKernel << < blocks, threads >> > (
        *server_, *(image.server()), (int) kernel, with_holes);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
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
    } else if (image.width() != width_ || image.height() != height_) {
        PrintInfo("Incompatible image size!\n");
        return;
    }

    const dim3 blocks(DIV_CEILING(width_, THREAD_2D_UNIT),
                      DIV_CEILING(height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    BilateralImageKernel << < blocks, threads >> > (
        *server_, *(image.server()), (int) kernel, val_sigma, with_holes);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
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
    } else if (image.width() != width_ || image.height() != height_) {
        PrintInfo("Incompatible image size!\n");
        return;
    }

    const dim3 blocks(DIV_CEILING(width_, THREAD_2D_UNIT),
                      DIV_CEILING(height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    ToFloatImageKernel << < blocks, threads >> > (
        *server_, *(image.server()), scale, offset);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
}