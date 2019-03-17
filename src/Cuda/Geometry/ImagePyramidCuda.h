//
// Created by wei on 9/27/18.
//

#pragma once

#include "GeometryClasses.h"
#include "ImageCuda.h"

namespace open3d {

namespace cuda {
template<typename Scalar, size_t Channel, size_t N>
class ImagePyramidCudaDevice {
private:
    /** Unfortunately, we cannot use shared_ptr for CUDA data structures **/
    /** We even cannot use CPU pointers **/
    /** -- We have to call ConnectSubServers() to explicitly link them. **/
    ImageCudaDevice<Scalar, Channel> images_[N];

public:
    __HOSTDEVICE__ int width(size_t level) const {
#ifdef DEBUG_CUDA_ENABLE_ASSERTION
        assert(level < N);
#endif
        return images_[level].width_;
    }
    __HOSTDEVICE__ int height(size_t level) const {
#ifdef DEBUG_CUDA_ENABLE_ASSERTION
        assert(level < N);
#endif
        return images_[level].height_;
    }

    __HOSTDEVICE__ ImageCudaDevice<Scalar, Channel> &operator[](size_t level) {
#ifdef DEBUG_CUDA_ENABLE_ASSERTION
        assert(level < N);
#endif
        return images_[level];
    }
    __HOSTDEVICE__ const ImageCudaDevice<Scalar, Channel> &operator[](size_t level) const {
#ifdef DEBUG_CUDA_ENABLE_ASSERTION
        assert(level < N);
#endif
        return images_[level];
    }

    friend class ImagePyramidCuda<Scalar, Channel, N>;
};

template<typename Scalar, size_t Channel, size_t N>
class ImagePyramidCuda {
public:
    std::shared_ptr<ImagePyramidCudaDevice<Scalar, Channel, N>> device_ = nullptr;

private:
    ImageCuda<Scalar, Channel> images_[N];

public:
    ImagePyramidCuda();
    ~ImagePyramidCuda();
    ImagePyramidCuda(const ImagePyramidCuda<Scalar, Channel, N> &other);
    ImagePyramidCuda<Scalar, Channel, N> &operator=(
        const ImagePyramidCuda<Scalar, Channel, N> &other);

    bool Create(int width, int height);
    void Release();
    void UpdateDevice();

    void Build(const ImageCuda<Scalar, Channel> &image);
    std::vector<std::shared_ptr<geometry::Image>> DownloadImages();

    /** Legacy **/
    std::vector<cv::Mat> DownloadMats();

    int width(size_t level) const {
        assert(level < N);
        return images_[level].width_;
    }
    int height(size_t level) const {
        assert(level < N);
        return images_[level].height_;
    }

    ImageCuda<Scalar, Channel> &operator[](size_t level) {
        assert(level < N);
        return images_[level];
    }
    const ImageCuda<Scalar, Channel> &operator[](size_t level) const {
        assert(level < N);
        return images_[level];
    }
};
} // cuda
} // open3d