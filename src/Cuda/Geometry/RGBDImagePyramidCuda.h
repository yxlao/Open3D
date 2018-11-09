//
// Created by wei on 11/8/18.
//

#pragma once

#include "GeometryClasses.h"
#include "RGBDImageCuda.h"

namespace open3d {
template<size_t N>
class RGBDImagePyramidCudaServer {
private:
    RGBDImageCudaServer rgbd_[N];

public:
    __HOSTDEVICE__ RGBDImageCudaServer &operator[] (size_t level) {
#ifdef DEBUG_CUDA_ENABLE_ASSERTION
        assert(level < N);
#endif
        return rgbd_[level];
    }

    __HOSTDEVICE__ const RGBDImageCudaServer &operator[] (size_t level) const {
#ifdef DEBUG_CUDA_ENABLE_ASSERTION
        assert(level < N);
#endif
        return rgbd_[level];
    }

    friend class RGBDImagePyramidCuda<N>;
};

template<size_t N>
class RGBDImagePyramidCuda {
private:
    std::shared_ptr<RGBDImagePyramidCudaServer<N>> server_ = nullptr;

private:
    RGBDImageCuda rgbd_[N];

public:
    RGBDImagePyramidCuda();
    ~RGBDImagePyramidCuda();
    RGBDImagePyramidCuda(const RGBDImagePyramidCuda<N> &other);
    RGBDImagePyramidCuda<N>& operator=(const RGBDImagePyramidCuda<N> &other);

    void Create(int width, int height);
    void Release();

    void Build(RGBDImageCuda &rgbd);

    void UpdateServer();

    RGBDImageCuda &operator[] (size_t level) {
        assert(level < N);
        return rgbd_[level];
    }
    const RGBDImageCuda &operator[] (size_t level) const {
        assert(level < N);
        return rgbd_[level];
    }

    const std::shared_ptr<RGBDImagePyramidCudaServer<N>> &server() const {
        return server_;
    }
    std::shared_ptr<RGBDImagePyramidCudaServer<N>> & server() {
        return server_;
    }
};
}


