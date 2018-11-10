//
// Created by wei on 11/5/18.
//

#pragma once

#include "ImageCuda.h"
#include <Cuda/Common/VectorCuda.h>
#include <memory>
#include <opencv2/opencv.hpp>

namespace open3d {
class RGBDImageCudaServer {
private:
    ImageCudaServer<Vector1f> depth_;
    ImageCudaServer<Vector3b> color_;

public:
    __HOSTDEVICE__ ImageCudaServer<Vector1f>& depth() {
        return depth_;
    }
    __HOSTDEVICE__ ImageCudaServer<Vector3b>& color() {
        return color_;
    }
};

class RGBDImageCuda {
private:
    std::shared_ptr<RGBDImageCudaServer> server_ = nullptr;

    ImageCuda<Vector1f> depth_;
    ImageCuda<Vector3b> color_;

    ImageCuda<Vector1s> depths_;

public:
    float depth_near_;
    float depth_far_;
    float depth_factor_;

public:
    RGBDImageCuda(float depth_near = 0.1f,
                  float depth_far = 3.5f,
                  float depth_factor = 1000.0f);
    RGBDImageCuda(ImageCuda<Vector1f> &depth,
                  ImageCuda<Vector3b> &color,
                  float depth_near = 0.1f,
                  float depth_far = 3.5f,
                  float depth_factor = 1000.0f);

    RGBDImageCuda(const RGBDImageCuda &other);
    RGBDImageCuda &operator = (const RGBDImageCuda &other);
    ~RGBDImageCuda();

    void Create(const ImageCuda<Vector1f> &depth,
                const ImageCuda<Vector3b> &color);
    void Release();

    void UpdateServer();

    void Upload(Image &depth, Image &color);
    void Upload(ImageCuda<Vector1f> &depth, ImageCuda<Vector3b> &color);

    /** Legacy **/
    void Upload(cv::Mat &depth, cv::Mat &color);

public:
    ImageCuda<Vector1f>& depth() {
        return depth_;
    }
    const ImageCuda<Vector1f>& depth() const {
        return depth_;
    }

    ImageCuda<Vector3b>& color() {
        return color_;
    }
    const ImageCuda<Vector3b>& color() const {
        return color_;
    }

    std::shared_ptr<RGBDImageCudaServer>& server() {
        return server_;
    }
    const std::shared_ptr<RGBDImageCudaServer>& server() const {
        return server_;
    }
};

}


