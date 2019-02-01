//
// Created by wei on 9/27/18.
//

#pragma once

#include "GeometryClasses.h"
#include "ImageCuda.h"

namespace open3d {

namespace cuda {
template<typename VecType, size_t N>
class ImagePyramidCudaDevice {
private:
    /** Unfortunately, we cannot use shared_ptr for CUDA data structures **/
    /** We even cannot use CPU pointers **/
    /** -- We have to call ConnectSubServers() to explicitly link them. **/
    ImageCudaDevice<VecType> images_[N];

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

    __HOSTDEVICE__ ImageCudaDevice<VecType> &operator[](size_t level) {
#ifdef DEBUG_CUDA_ENABLE_ASSERTION
        assert(level < N);
#endif
        return images_[level];
    }
    __HOSTDEVICE__ const ImageCudaDevice<VecType> &operator[](size_t level) const {
#ifdef DEBUG_CUDA_ENABLE_ASSERTION
        assert(level < N);
#endif
        return images_[level];
    }

    friend class ImagePyramidCuda<VecType, N>;
};

template<typename VecType, size_t N>
class ImagePyramidCuda {
public:
    std::shared_ptr<ImagePyramidCudaDevice<VecType, N>> device_ = nullptr;

private:
    ImageCuda<VecType> images_[N];

public:
    ImagePyramidCuda();
    ~ImagePyramidCuda();
    ImagePyramidCuda(const ImagePyramidCuda<VecType, N> &other);
    ImagePyramidCuda<VecType, N> &operator=(
        const ImagePyramidCuda<VecType, N> &other);

    bool Create(int width, int height);
    void Release();
    void UpdateDevice();

    void Build(const ImageCuda<VecType> &image);
    std::vector<std::shared_ptr<Image>> DownloadImages();

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

    ImageCuda<VecType> &operator[](size_t level) {
        assert(level < N);
        return images_[level];
    }
    const ImageCuda<VecType> &operator[](size_t level) const {
        assert(level < N);
        return images_[level];
    }
};
} // cuda
} // open3d