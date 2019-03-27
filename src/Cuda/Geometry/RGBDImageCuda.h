//
// Created by wei on 11/5/18.
//

#pragma once

#include "ImageCuda.h"
#include <Cuda/Common/LinearAlgebraCuda.h>
#include <memory>
#include <opencv2/opencv.hpp>

/** In VO, we need depth & intensity
 *  In fusion, we need depth & color
 */
namespace open3d {
namespace cuda {
class RGBDImageCudaDevice {
public:
    ImageCudaDevice<ushort, 1> depth_raw_;
    ImageCudaDevice<uchar, 3> color_raw_;

    ImageCudaDevice<float, 1> depth_;

public:
    int width_;
    int height_;
};

class RGBDImageCuda {
public:
    std::shared_ptr<RGBDImageCudaDevice> device_ = nullptr;

    /* Raw input */
    ImageCuda<ushort, 1> depth_raw_;
    ImageCuda<uchar, 3> color_raw_;

    ImageCuda<float, 1> depth_;

public:
    float depth_trunc_;
    float depth_factor_;

    int width_;
    int height_;

public:
    RGBDImageCuda(float depth_trunc = 3.0f, float depth_factor = 1000.0f);
    RGBDImageCuda(int width, int height,
                  float depth_trunc = 3.0f, float depth_factor = 1000.0f);

    RGBDImageCuda(const RGBDImageCuda &other);
    RGBDImageCuda &operator=(const RGBDImageCuda &other);
    ~RGBDImageCuda();

    bool Create(int width, int height);
    void Release();
    void UpdateDevice();

    void CopyFrom(RGBDImageCuda &other);
    void Build(ImageCuda<ushort, 1> &depth_raw, ImageCuda<uchar, 3> &color_raw);
    void Upload(geometry::Image &depth_raw, geometry::Image &color_raw);

    /** Legacy **/
    void Upload(cv::Mat &depth, cv::Mat &color);
};

class RGBDImageCudaKernelCaller {
public:
    static void ConvertDepthToFloat(RGBDImageCuda &rgbd);
};

__GLOBAL__
void ConvertDepthToFloatKernel(RGBDImageCudaDevice device,
    float factor, float depth_trunc);

} // cuda
} // open3d