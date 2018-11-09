//
// Created by wei on 9/27/18.
//

#pragma once

#include "GeometryClasses.h"
#include "ImageCuda.h"

namespace open3d {
template<typename VecType, size_t N>
class ImagePyramidCudaServer {
private:
    /** Unfortunately, we cannot use shared_ptr for CUDA data structures **/
    /** We even cannot use CPU pointers **/
    /** -- We have to call ConnectSubServers() to explicitly link them. **/
    ImageCudaServer<VecType> images_[N];

public:
    __HOSTDEVICE__ ImageCudaServer<VecType> &operator[] (size_t level) {
#ifdef DEBUG_CUDA_ENABLE_ASSERTION
        assert(level < N);
#endif
        return images_[level];
    }

    __HOSTDEVICE__ const ImageCudaServer<VecType> &operator[] (size_t level) const {
#ifdef DEBUG_CUDA_ENABLE_ASSERTION
        assert(level < N);
#endif
        return images_[level];
    }

    friend class ImagePyramidCuda<VecType, N>;
};

template<typename VecType, size_t N>
class ImagePyramidCuda {
private:
    std::shared_ptr<ImagePyramidCudaServer<VecType, N>> server_ = nullptr;

private:
    ImageCuda<VecType> images_[N];

public:
    ImagePyramidCuda();
    ~ImagePyramidCuda();
    ImagePyramidCuda(const ImagePyramidCuda<VecType, N> &other);
    ImagePyramidCuda<VecType, N> &operator=(
        const ImagePyramidCuda<VecType, N> &other);

    void Create(int width, int height);
    void Release();

    void Build(const ImageCuda<VecType> &image);
    std::vector<std::shared_ptr<Image>> DownloadImages();
    std::vector<cv::Mat> DownloadMats();

    void UpdateServer();

    ImageCuda<VecType> & operator[] (size_t level) {
        assert(level < N);
        return images_[level];
    }
    const ImageCuda<VecType> &operator[] (size_t level) const {
        assert(level < N);
        return images_[level];
    }

    const std::shared_ptr<ImagePyramidCudaServer<VecType, N>> &server() const {
        return server_;
    }
    std::shared_ptr<ImagePyramidCudaServer<VecType, N>> &server() {
        return server_;
    }
};
}

