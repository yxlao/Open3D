#include "ImageCudaDevice.cuh"

namespace open3d {

/**
 * Downsampling
 */
template<typename VecType>
__global__
void DownsampleImageKernel(ImageCudaServer<VecType> src,
                           ImageCudaServer<VecType> dst,
                           DownsampleMethod method) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= dst.width_ || v >= dst.height_)
        return;

    /** Re-write the function by hard-coding if we want to accelerate **/
    switch (method) {
        case BoxFilter: dst.at(u, v) = src.BoxFilter2x2(u << 1, v << 1);
            return;
        case BoxFilterWithHoles:dst.at(u, v) = src.BoxFilter2x2WithHoles(u << 1, v << 1);
            return;
        case GaussianFilter:dst.at(u, v) = src.GaussianFilter(u << 1, v << 1, Gaussian5x5);
            return;
        case GaussianFilterWithHoles:dst.at(u, v) = src.GaussianFilterWithHoles(u << 1, v << 1, Gaussian5x5);
            return;
        default: printf("Unsupported method.\n");
    }
}

template<typename VecType>
__host__
void ImageCudaKernelCaller<VecType>::DownsampleImageKernelCaller(
    ImageCudaServer<VecType> &src, ImageCudaServer<VecType> &dst,
    DownsampleMethod method) {

    const dim3 blocks(DIV_CEILING(src.width_, THREAD_2D_UNIT),
                      DIV_CEILING(dst.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    DownsampleImageKernel << < blocks, threads >> > (src, dst, method);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

/**
 * Shift
 */
template<typename VecType>
__global__
void ShiftImageKernel(
    ImageCudaServer<VecType> src,
    ImageCudaServer<VecType> dst,
    float dx, float dy, bool with_holes) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    dst.at(u, v) = VecType(0);
    if (u >= dst.width_ || v >= dst.height_)
        return;

    if (u + dx < 0 || u + dx >= src.width_ - 1
        || v + dy < 0 || v + dy >= src.height_ - 1)
        return;

    if (with_holes) {
        dst.at(u, v) = src.interp_with_holes_at(u + dx, v + dy);
    } else {
        dst.at(u, v) = src.interp_at(u + dx, v + dy);
    }
}

template<typename VecType>
__host__
void ImageCudaKernelCaller<VecType>::ShiftImageKernelCaller(
    ImageCudaServer<VecType> &src, ImageCudaServer<VecType> &dst,
    float dx, float dy, bool with_holes) {

    const dim3 blocks(DIV_CEILING(src.width_, THREAD_2D_UNIT),
                      DIV_CEILING(src.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    ShiftImageKernel << < blocks, threads >> > (src, dst, dx, dy, with_holes);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

/**
 * Gaussian
 */
template<typename VecType>
__global__
void GaussianImageKernel(
    ImageCudaServer<VecType> src,
    ImageCudaServer<VecType> dst,
    int kernel_idx, bool with_holes) {

    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= dst.width_ || v >= dst.height_)
        return;

    if (with_holes) {
        dst.at(u, v) = src.GaussianFilterWithHoles(u, v, kernel_idx);
    } else {
        dst.at(u, v) = src.GaussianFilter(u, v, kernel_idx);
    }
}
template<typename VecType>
__host__
void ImageCudaKernelCaller<VecType>::GaussianImageKernelCaller(
    ImageCudaServer<VecType> &src, ImageCudaServer<VecType> &dst,
    int kernel_idx, bool with_holes) {

    const dim3 blocks(DIV_CEILING(src.width_, THREAD_2D_UNIT),
                      DIV_CEILING(src.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    GaussianImageKernel << < blocks, threads >> > (
        src, dst, kernel_idx, with_holes);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

/**
 * Bilateral
 */
template<typename VecType>
__global__
void BilateralImageKernel(ImageCudaServer<VecType> src,
                          ImageCudaServer<VecType> dst,
                          int kernel_idx,
                          float val_sigma,
                          bool with_holes) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= dst.width_ || v >= dst.height_)
        return;
    if (with_holes) {
        dst.at(u, v) =
            src.BilateralFilterWithHoles(u, v, kernel_idx, val_sigma);
    } else {
        dst.at(u, v) = src.BilateralFilter(u, v, kernel_idx, val_sigma);
    }
}
template<typename VecType>
__host__
void ImageCudaKernelCaller<VecType>::BilateralImageKernelCaller(
    ImageCudaServer<VecType> &src, ImageCudaServer<VecType> &dst,
    int kernel_idx, float val_sigma, bool with_holes) {
    const dim3 blocks(DIV_CEILING(src.width_, THREAD_2D_UNIT),
                      DIV_CEILING(src.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    BilateralImageKernel << < blocks, threads >> > (
        src, dst, kernel_idx, val_sigma, with_holes);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

/**
 * Gradient, using Sobel operator
 */
template<typename VecType>
__global__
void SobelImageKernel(
    ImageCudaServer<VecType> src,
    ImageCudaServer<typename VecType::VecTypef> dx,
    ImageCudaServer<typename VecType::VecTypef> dy,
    bool with_holes) {

    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= src.width_ || v >= src.height_) return;

    if (u == 0 || u == src.width_ - 1 || v == 0 || v == src.height_ - 1) {
        dx.at(u, v) = VecType::VecTypef(0);
        dy.at(u, v) = VecType::VecTypef(0);
        return;
    }

    typename ImageCudaServer<VecType>::Grad grad;
    if (with_holes) {
        grad = src.SobelWithHoles(u, v);
    } else {
        grad = src.Sobel(u, v);
    }
    dx(u, v) = grad.dx;
    dy(u, v) = grad.dy;
}
template<typename VecType>
__host__
void ImageCudaKernelCaller<VecType>::SobelImageKernelCaller(
    ImageCudaServer<VecType> &src,
    ImageCudaServer<typename VecType::VecTypef> &dx,
    ImageCudaServer<typename VecType::VecTypef> &dy,
    bool with_holes) {
    const dim3 blocks(DIV_CEILING(src.width_, THREAD_2D_UNIT),
                      DIV_CEILING(src.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    SobelImageKernel << < blocks, threads >> > (src, dx, dy, with_holes);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

/**
 * Conversion
 */
template<typename VecType>
__global__
void ToFloatImageKernel(
    ImageCudaServer<VecType> src,
    ImageCudaServer<typename VecType::VecTypef> dst,
    float scale,
    float offset) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= dst.width_ || v >= dst.height_) return;

    dst.at(u, v) =
        src.at(u, v).ToVectorf() * scale + VecType::VecTypef(offset);
}

template<typename VecType>
__host__
void ImageCudaKernelCaller<VecType>::ToFloatImageKernelCaller(
    ImageCudaServer<VecType> &src,
    ImageCudaServer<typename VecType::VecTypef> &dst,
    float scale, float offset) {
    const dim3 blocks(DIV_CEILING(src.width_, THREAD_2D_UNIT),
                      DIV_CEILING(src.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    ToFloatImageKernel << < blocks, threads >> > (
        src, dst, scale, offset);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
}