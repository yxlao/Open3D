//
// Created by wei on 11/5/18.
//

#pragma once

#include "ImageCuda.h"
#include <Cuda/Common/VectorCuda.h>
#include <memory>
#include <opencv2/opencv.hpp>

/** In VO, we need depth & intensity
 *  In fusion, we need depth & color
 */
namespace open3d {
class RGBDImageCudaServer {
private:
    ImageCudaServer<Vector1f> depth_;
    ImageCudaServer<Vector3b> color_;
    ImageCudaServer<Vector1f> intensity_;

public:
    int width_;
    int height_;

public:
    __HOSTDEVICE__ ImageCudaServer<Vector1f> &depth() { return depth_; }
    __HOSTDEVICE__ ImageCudaServer<Vector3b> &color() { return color_; }
    __HOSTDEVICE__ ImageCudaServer<Vector1f> &intensity() { return intensity_; }
};

class RGBDImageCuda {
private:
    std::shared_ptr<RGBDImageCudaServer> server_ = nullptr;

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
    void UpdateServer();

    void CopyFrom(RGBDImageCuda &other);
    void Build(ImageCuda<Vector1s> &depth_raw,
               ImageCuda<Vector3b> &color_raw);
    void Upload(Image &depth_raw, Image &color_raw);

    /** Legacy **/
    void Upload(cv::Mat &depth, cv::Mat &color);

public:
    ImageCuda<Vector1s> &depth_raw() { return depth_raw_; }
    const ImageCuda<Vector1s> &depth_raw() const { return depth_raw_; }
    ImageCuda<Vector1f> &depthf() { return depthf_; }
    const ImageCuda<Vector1f> &depthf() const { return depthf_; }
    ImageCuda<Vector3b> &color() { return color_; }
    const ImageCuda<Vector3b> &color() const { return color_; }
    ImageCuda<Vector1f> &intensity() { return intensity_; }
    const ImageCuda<Vector1f> &intensity() const { return intensity_; }

    std::shared_ptr<RGBDImageCudaServer> &server() { return server_; }
    const std::shared_ptr<RGBDImageCudaServer> &server() const {
        return server_;
    }
};

}


