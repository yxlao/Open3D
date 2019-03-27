//
// Created by wei on 9/27/18.
//

#pragma once

#include "GeometryClasses.h"
#include <Cuda/Common/Common.h>
#include <Cuda/Common/LinearAlgebraCuda.h>
#include <Open3D/Geometry/Image.h>

#include <cstdlib>
#include <memory>
// #include <vector_types.h>
#include <opencv2/opencv.hpp>

//#define HOST_DEBUG_MONITOR_LIFECYCLE

namespace open3d {

namespace cuda {
enum GaussianKernelSize {
    Gaussian3x3 = 0,
    Gaussian5x5 = 1,
    Gaussian7x7 = 2,
};

enum DownsampleMethod {
    BoxFilter = 0, /* Naive 2x2 */
    GaussianFilter = 2, /* 5x5, suggested by OpenCV */
};

/**
 * @tparam VecType:
 * Other templates are regarded as incompatible.
 */
template<typename Scalar, size_t Channel>
class ImageCudaDevice {
private:
    typedef VectorCuda<Scalar, Channel> ValueType;

    ValueType *data_;

public:
    int width_;
    int height_;
    int pitch_;

public:
    ValueType *&data() {
        return data_;
    }

    __DEVICE__ inline ValueType &at(int x, int y);
    __DEVICE__ inline Scalar &at(int x, int y, int channel);

    __DEVICE__ inline ValueType &operator()(int x, int y);
    __DEVICE__ inline Scalar &operator()(int x, int y, int channel);

    __DEVICE__ inline ValueType interp_at(float x, float y);

    __DEVICE__
    ValueType BoxFilter2x2(int x, int y);
    __DEVICE__
    ValueType GaussianFilter(int x, int y, int kernel_idx);
    __DEVICE__
    ValueType BilateralFilter(int x, int y, int kernel_idx, float val_sigma);

    struct Grad {
        VectorCuda<float, Channel> dx;
        VectorCuda<float, Channel> dy;
    };
    __DEVICE__
    Grad Sobel(int x, int y);

    friend class ImageCuda<Scalar, Channel>;
};

template<typename Scalar, size_t Channel>
class ImageCuda {
public:
    std::shared_ptr<ImageCudaDevice<Scalar, Channel>> device_ = nullptr;
    typedef ImageCuda<Scalar, Channel> ImageCudaType;
    typedef ImageCuda<float, Channel> ImageCudaTypef;

public:
    int width_;
    int height_;
    int pitch_; /* bytes per row */

public:
    ImageCuda();
    /** The semantic of our copy constructor (and also operator =)
     *  is memory efficient. No moving semantic is needed.
     */
    ImageCuda(const ImageCudaType &other);
    ImageCuda(int width, int height);
    ~ImageCuda();
    ImageCudaType &operator=(const ImageCudaType &other);

    /**
     * @return true if already created with desired size, or newly created
     *         false if size is incompatible
     */
    bool Create(int width, int height);
    void Release();
    void UpdateDevice();

    void CopyFrom(const ImageCudaType &other);
    void Upload(geometry::Image &image);
    std::shared_ptr<geometry::Image> DownloadImage();

    /********** Image Processing **********/
    /** 'switch' code in kernel can be slow, manually expand it if needed. **/
    ImageCudaType Downsample(
        DownsampleMethod method = GaussianFilter);
    void Downsample(ImageCudaType &image,
                    DownsampleMethod method = GaussianFilter);

    std::tuple<ImageCudaTypef, ImageCudaTypef> Sobel();
    void Sobel(ImageCudaTypef &dx, ImageCudaTypef &dy);

    ImageCudaType Shift(float dx, float dy);
    void Shift(ImageCudaType &image, float dx, float dy);

    ImageCudaType Gaussian(GaussianKernelSize option);
    void Gaussian(ImageCudaType &image, GaussianKernelSize option);

    ImageCudaType Bilateral(GaussianKernelSize option = Gaussian5x5,
                            float val_sigma = 20.0f);
    void Bilateral(ImageCudaType &image,
                   GaussianKernelSize option = Gaussian5x5,
                   float val_sigma = 20.0f);

    ImageCudaTypef ConvertToFloat(
        float scale = 1.0f, float offset = 0.0f);
    void ConvertToFloat(ImageCudaTypef &image,
                        float scale = 1.0f, float offset = 0.0f);

    ImageCuda<float, 1> ConvertRGBToIntensity();
    void ConvertRGBToIntensity(ImageCuda<float, 1> &image);

    /********** Legacy **********/
    void Upload(cv::Mat &m);
    cv::Mat DownloadMat();
};

template<typename Scalar, size_t Channel>
class ImageCudaKernelCaller {
public:
    typedef ImageCuda<Scalar, Channel> ImageCudaType;
    typedef ImageCuda<float, Channel> ImageCudaTypef;

    static void Downsample(ImageCudaType &src, ImageCudaType &dst,
                           DownsampleMethod method);
    static void Shift(ImageCudaType &src, ImageCudaType &dst,
                      float dx, float dy);
    static void Gaussian(ImageCudaType &src, ImageCudaType &dst,
                         int kernel_idx);
    static void Bilateral(ImageCudaType &src, ImageCudaType &dst,
                          int kernel_idx, float val_sigma);
    static void Sobel(ImageCudaType &src,
                      ImageCudaTypef &dx, ImageCudaTypef &dy);
    static void ConvertToFloat(ImageCudaType &src,
                               ImageCudaTypef &dst,
                               float scale, float offset);
    static void ConvertRGBToIntensity(ImageCudaType &src,
                                      ImageCuda<float, 1> &dst);
};

template<typename Scalar, size_t Channel>
__GLOBAL__
void DownsampleKernel(ImageCudaDevice<Scalar, Channel> src,
                      ImageCudaDevice<Scalar, Channel> dst,
                      DownsampleMethod method);

template<typename Scalar, size_t Channel>
__GLOBAL__
void ShiftKernel(ImageCudaDevice<Scalar, Channel> src,
                 ImageCudaDevice<Scalar, Channel> dst,
                 float dx, float dy);

template<typename Scalar, size_t Channel>
__GLOBAL__
void GaussianKernel(ImageCudaDevice<Scalar, Channel> src,
                    ImageCudaDevice<Scalar, Channel> dst,
                    int kernel_idx);

template<typename Scalar, size_t Channel>
__GLOBAL__
void BilateralKernel(ImageCudaDevice<Scalar, Channel> src,
                     ImageCudaDevice<Scalar, Channel> dst,
                     int kernel_idx, float val_sigma);

template<typename Scalar, size_t Channel>
__GLOBAL__
void SobelKernel(ImageCudaDevice<Scalar, Channel> src,
                 ImageCudaDevice<float, Channel> dx,
                 ImageCudaDevice<float, Channel> dy);

template<typename Scalar, size_t Channel>
__GLOBAL__
void ConvertToFloatKernel(ImageCudaDevice<Scalar, Channel> src,
                          ImageCudaDevice<float, Channel> dst,
                          float scale, float offset);

template<typename Scalar, size_t Channel>
__GLOBAL__
void ConvertRGBToIntensityKernel(ImageCudaDevice<Scalar, Channel> src,
                                 ImageCudaDevice<float, 1> dst);
} // cuda
} // open3d