//
// Created by wei on 11/8/18.
//

#pragma once

#include "GeometryClasses.h"
#include "RGBDImageCuda.h"

namespace open3d {
namespace cuda {
template<size_t N>
class RGBDImagePyramidCudaDevice {
private:
    RGBDImageCudaDevice rgbd_[N];

public:
    __HOSTDEVICE__ RGBDImageCudaDevice &operator[](size_t level) {
#ifdef DEBUG_CUDA_ENABLE_ASSERTION
        assert(level < N);
#endif
        return rgbd_[level];
    }

    __HOSTDEVICE__ const RGBDImageCudaDevice &operator[](size_t level) const {
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
    std::shared_ptr<RGBDImagePyramidCudaDevice<N>> server_ = nullptr;

private:
    RGBDImageCuda rgbd_[N];

public:
    RGBDImagePyramidCuda();
    ~RGBDImagePyramidCuda();
    RGBDImagePyramidCuda(const RGBDImagePyramidCuda<N> &other);
    RGBDImagePyramidCuda<N> &operator=(const RGBDImagePyramidCuda<N> &other);

    bool Create(int width, int height);
    void Release();
    void UpdateServer();

    void Build(RGBDImageCuda &rgbd);

    RGBDImageCuda &operator[](size_t level) {
        assert(level < N);
        return rgbd_[level];
    }
    const RGBDImageCuda &operator[](size_t level) const {
        assert(level < N);
        return rgbd_[level];
    }

    const std::shared_ptr<RGBDImagePyramidCudaDevice<N>> &server() const {
        return server_;
    }
    std::shared_ptr<RGBDImagePyramidCudaDevice<N>> &server() {
        return server_;
    }
};
} // cuda
} // open3d