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
    ImageCudaDevice<Vector1f> depth_;
    ImageCudaDevice<Vector3b> color_;
    ImageCudaDevice<Vector1f> intensity_;

public:
    int width_;
    int height_;
};

class RGBDImageCuda {
public:
    std::shared_ptr<RGBDImageCudaDevice> device_ = nullptr;

    /* Raw input */
    ImageCuda<Vector1s> depth_raw_;
    ImageCuda<Vector3b> color_;

    ImageCuda<Vector1f> depthf_;
    ImageCuda<Vector1f> intensity_;

public:
    float depth_near_;
    float depth_far_;
    float depth_factor_;

    int width_;
    int height_;

public:
    RGBDImageCuda(float depth_near = 0.1f, float depth_far = 3.5f,
                  float depth_factor = 1000.0f);
    RGBDImageCuda(int width, int height,
                  float depth_near = 0.1f, float depth_far = 3.5f,
                  float depth_factor = 1000.0f);

    RGBDImageCuda(const RGBDImageCuda &other);
    RGBDImageCuda &operator=(const RGBDImageCuda &other);
    ~RGBDImageCuda();

    bool Create(int width, int height);
    void Release();
    void UpdateDevice();

    void CopyFrom(RGBDImageCuda &other);
    void Build(ImageCuda<Vector1s> &depth_raw,
               ImageCuda<Vector3b> &color_raw);
    void Upload(Image &depth_raw, Image &color_raw);

    /** Legacy **/
    void Upload(cv::Mat &depth, cv::Mat &color);
};

} // cuda
} // open3d