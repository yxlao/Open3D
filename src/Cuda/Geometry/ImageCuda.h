//
// Created by wei on 9/27/18.
//

#pragma once

#include "GeometryClasses.h"
#include <Cuda/Common/Common.h>
#include <Cuda/Common/VectorCuda.h>
#include <Core/Geometry/Image.h>

#include <cstdlib>
#include <memory>
#include <vector_types.h>
#include <opencv2/opencv.hpp>

//#define HOST_DEBUG_MONITOR_LIFECYCLE

namespace open3d {

enum GaussianKernelSize {
    Gaussian3x3 = 0,
    Gaussian5x5 = 1,
    Gaussian7x7 = 2,
};

enum DownsampleMethod {
    BoxFilter = 0, /* Naive 2x2 */
    BoxFilterWithHoles = 1,
    GaussianFilter = 2, /* 5x5, suggested by OpenCV */
    GaussianFilterWithHoles = 3
};

/**
 * @tparam VecType:
 * Other templates are regarded as incompatible.
 */
template<typename VecType>
class ImageCudaServer {
private:
    VecType *data_;

public:
    int width_;
    int height_;
    int pitch_;

public:
    VecType *&data() {
        return data_;
    }

    __DEVICE__ inline VecType &at(int x, int y);
    __DEVICE__ inline VecType &operator()(int x, int y);
    __DEVICE__ inline VecType interp_at(float x, float y);
    __DEVICE__ inline VecType interp_with_holes_at(float x, float y);

    __DEVICE__
    VecType BoxFilter2x2(int x, int y);
    __DEVICE__
    VecType BoxFilter2x2WithHoles(int x, int y);
    __DEVICE__
    VecType GaussianFilter(int x, int y, int kernel_idx);
    __DEVICE__
    VecType GaussianFilterWithHoles(int x, int y, int kernel_idx);
    __DEVICE__
    VecType BilateralFilter(int x, int y, int kernel_idx, float val_sigma);
    __DEVICE__
    VecType BilateralFilterWithHoles(int x, int y, int kernel_idx, float val_sigma);

    /** Wish I could use std::pair here... **/
    struct Grad {
        typename VecType::VecTypef dx;
        typename VecType::VecTypef dy;
    };
    __DEVICE__
    Grad Sobel(int x, int y);
    __DEVICE__
    Grad SobelWithHoles(int x, int y);

    friend class ImageCuda<VecType>;
};

template<typename VecType>
class ImageCuda {
private:
    std::shared_ptr<ImageCudaServer<VecType>> server_ = nullptr;

public:
    int width_;
    int height_;
    int pitch_; /* bytes per row */

public:
    ImageCuda();
    /** The semantic of our copy constructor (and also operator =)
     *  is memory efficient. No moving semantic is needed.
     */
    ImageCuda(const ImageCuda<VecType> &other);
    ImageCuda(int width, int height);
    ~ImageCuda();
    ImageCuda<VecType> &operator=(const ImageCuda<VecType> &other);

    /**
     * @return true if already created with desired size, or newly created
     *         false if size is incompatible
     */
    bool Create(int width, int height);
    void Release();
    void UpdateServer();

    void CopyFrom(const ImageCuda<VecType> &other);
    void Upload(Image& image);
    std::shared_ptr<Image> DownloadImage();

    /********** Image Processing **********/
    /** 'switch' code in kernel can be slow, manually expand it if needed. **/
    ImageCuda<VecType> Downsample(
        DownsampleMethod method = GaussianFilter);
    void Downsample(ImageCuda<VecType> &image,
                    DownsampleMethod method = GaussianFilter);

    std::tuple<ImageCuda<typename VecType::VecTypef>,
               ImageCuda<typename VecType::VecTypef>> Sobel(
                   bool with_holes = true);
    void Sobel(ImageCuda<typename VecType::VecTypef> &dx,
               ImageCuda<typename VecType::VecTypef> &dy,
               bool with_holes = true);

    ImageCuda<VecType> Shift(float dx, float dy, bool with_holes = true);
    void Shift(ImageCuda<VecType> &image, float dx, float dy,
               bool with_holes = true);

    ImageCuda<VecType> Gaussian(GaussianKernelSize option,
                                bool with_holess = true);
    void Gaussian(ImageCuda<VecType> &image,
                  GaussianKernelSize option,
                  bool with_holes = true);

    ImageCuda<VecType> Bilateral(GaussianKernelSize option = Gaussian5x5,
                                 float val_sigma = 20.0f,
                                 bool with_holes = true);
    void Bilateral(ImageCuda<VecType> &image,
                   GaussianKernelSize option = Gaussian5x5,
                   float val_sigma = 20.0f,
                   bool with_holes = true);

    ImageCuda<typename VecType::VecTypef> ConvertToFloat(
        float scale = 1.0f, float offset = 0.0f);
    void ConvertToFloat(ImageCuda<typename VecType::VecTypef> &image,
                        float scale = 1.0f, float offset = 0.0f);

    ImageCuda<Vector1f> ConvertRGBToIntensity();
    void ConvertRGBToIntensity(ImageCuda<Vector1f> &image);

    /********** Legacy **********/
    void Upload(cv::Mat &m);
    cv::Mat DownloadMat();

    std::shared_ptr<ImageCudaServer<VecType>> &server() {
        return server_;
    }
    const std::shared_ptr<ImageCudaServer<VecType>> &server() const {
        return server_;
    }
};

template<typename VecType>
class ImageCudaKernelCaller {
public:
    static __HOST__ void DownsampleImageKernelCaller(
        ImageCudaServer<VecType> &src, ImageCudaServer<VecType> &dst,
        DownsampleMethod method);
    static __HOST__ void ShiftImageKernelCaller(
        ImageCudaServer<VecType> &src, ImageCudaServer<VecType> &dst,
        float dx, float dy, bool with_holes);
    static __HOST__ void GaussianImageKernelCaller(
        ImageCudaServer<VecType> &src,ImageCudaServer<VecType> &dst,
        int kernel_idx, bool with_holes);
    static __HOST__ void BilateralImageKernelCaller(
        ImageCudaServer<VecType> &src, ImageCudaServer<VecType> &dst,
        int kernel_idx, float val_sigma, bool with_holes);
    static __HOST__ void SobelImageKernelCaller(
        ImageCudaServer<VecType> &src,
        ImageCudaServer<typename VecType::VecTypef> &dx,
        ImageCudaServer<typename VecType::VecTypef> &dy,
        bool with_holes);
    static __HOST__ void ConvertToFloatImageKernelCaller (
        ImageCudaServer<VecType> &src,
        ImageCudaServer<typename VecType::VecTypef> &dst,
        float scale, float offset);
    static __HOST__ void ConvertRGBToIntensityKernelCaller(
        ImageCudaServer<VecType> &src,
        ImageCudaServer<Vector1f> &dst);
};

template<typename VecType>
__GLOBAL__
void DownsampleImageKernel(
    ImageCudaServer<VecType> src, ImageCudaServer<VecType> dst,
    DownsampleMethod method);

template<typename VecType>
__GLOBAL__
void ShiftImageKernel(
    ImageCudaServer<VecType> src, ImageCudaServer<VecType> dst,
    float dx, float dy, bool with_holes);

template<typename VecType>
__GLOBAL__
void GaussianImageKernel(
    ImageCudaServer<VecType> src,ImageCudaServer<VecType> dst,
    int kernel_idx, bool with_holes);

template<typename VecType>
__GLOBAL__
void BilateralImageKernel(
    ImageCudaServer<VecType> src, ImageCudaServer<VecType> dst,
    int kernel_idx, float val_sigma, bool with_holes);

template<typename VecType>
__GLOBAL__
void SobelImageKernel(
    ImageCudaServer<VecType> src,
    ImageCudaServer<typename VecType::VecTypef> dx,
    ImageCudaServer<typename VecType::VecTypef> dy,
    bool with_holes);

template<typename VecType>
__GLOBAL__
void ConvertToFloatImageKernel(
    ImageCudaServer<VecType> src,
    ImageCudaServer<typename VecType::VecTypef> dst,
    float scale, float offset);

template<typename VecType>
__GLOBAL__
void ConvertRGBToIntensityImageKernel(
    ImageCudaServer<VecType> src,
    ImageCudaServer<Vector1f> dst);
}
