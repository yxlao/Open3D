#include "ImageCudaDevice.cuh"

namespace open3d {

namespace cuda {
/**
 * Downsampling
 */
template<typename VecType>
__global__
void DownsampleKernel(ImageCudaDevice<VecType> src,
                      ImageCudaDevice<VecType> dst,
                      DownsampleMethod method) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= dst.width_ || v >= dst.height_)
        return;

    /** Re-write the function by hard-coding if we want to accelerate **/
    switch (method) {
        case BoxFilter: dst.at(u, v) = src.BoxFilter2x2(u << 1, v << 1);
            return;
        case BoxFilterWithHoles:
            dst.at(u, v) = src.BoxFilter2x2WithHoles(u << 1, v << 1);
            return;
        case GaussianFilter:
            dst.at(u, v) = src.GaussianFilter(u << 1, v << 1, Gaussian5x5);
            return;
        case GaussianFilterWithHoles:
            dst.at(u, v) = src.GaussianFilterWithHoles(u << 1,
                                                       v << 1,
                                                       Gaussian5x5);
            return;
        default: printf("Unsupported method.\n");
    }
}

template<typename VecType>
__host__
void ImageCudaKernelCaller<VecType>::Downsample(
    ImageCuda<VecType> &src, ImageCuda<VecType> &dst,
    DownsampleMethod method) {

    const dim3 blocks(DIV_CEILING(src.width_, THREAD_2D_UNIT),
                      DIV_CEILING(dst.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    DownsampleKernel << < blocks, threads >> > (
        *src.device_, *dst.device_, method);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

/**
 * Shift
 */
template<typename VecType>
__global__
void ShiftKernel(ImageCudaDevice<VecType> src,
                      ImageCudaDevice<VecType> dst,
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
void ImageCudaKernelCaller<VecType>::Shift(
    ImageCuda<VecType> &src, ImageCuda<VecType> &dst,
    float dx, float dy, bool with_holes) {

    const dim3 blocks(DIV_CEILING(src.width_, THREAD_2D_UNIT),
                      DIV_CEILING(src.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    ShiftKernel << < blocks, threads >> > (
        *src.device_, *dst.device_, dx, dy, with_holes);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

/**
 * Gaussian
 */
template<typename VecType>
__global__
void GaussianKernel(ImageCudaDevice<VecType> src,
                         ImageCudaDevice<VecType> dst,
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
void ImageCudaKernelCaller<VecType>::Gaussian(
    ImageCuda<VecType> &src, ImageCuda<VecType> &dst,
    int kernel_idx, bool with_holes) {

    const dim3 blocks(DIV_CEILING(src.width_, THREAD_2D_UNIT),
                      DIV_CEILING(src.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    GaussianKernel << < blocks, threads >> > (
        *src.device_, *dst.device_, kernel_idx, with_holes);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

/**
 * Bilateral
 */
template<typename VecType>
__global__
void BilateralKernel(ImageCudaDevice<VecType> src,
                     ImageCudaDevice<VecType> dst,
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
void ImageCudaKernelCaller<VecType>::Bilateral(
    ImageCuda<VecType> &src, ImageCuda<VecType> &dst,
    int kernel_idx, float val_sigma, bool with_holes) {
    const dim3 blocks(DIV_CEILING(src.width_, THREAD_2D_UNIT),
                      DIV_CEILING(src.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    BilateralKernel << < blocks, threads >> > (
        *src.device_, *dst.device_, kernel_idx, val_sigma, with_holes);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

/**
 * Gradient, using Sobel operator
 */
template<typename VecType>
__global__
void SobelKernel(ImageCudaDevice<VecType> src,
                 ImageCudaDevice<typename VecType::VecTypef> dx,
                 ImageCudaDevice<typename VecType::VecTypef> dy,
                 bool with_holes) {

    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= src.width_ || v >= src.height_) return;

    if (u == 0 || u == src.width_ - 1 || v == 0 || v == src.height_ - 1) {
        dx.at(u, v) = VecType::VecTypef(0);
        dy.at(u, v) = VecType::VecTypef(0);
        return;
    }

    typename ImageCudaDevice<VecType>::Grad grad;
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
void ImageCudaKernelCaller<VecType>::Sobel(
    ImageCuda<VecType> &src,
    ImageCuda<typename VecType::VecTypef> &dx,
    ImageCuda<typename VecType::VecTypef> &dy,
    bool with_holes) {
    const dim3 blocks(DIV_CEILING(src.width_, THREAD_2D_UNIT),
                      DIV_CEILING(src.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    SobelKernel << < blocks, threads >> > (
        *src.device_, *dx.device_, *dy.device_, with_holes);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

/**
 * Conversion
 */
template<typename VecType>
__global__
void ConvertToFloatKernel(ImageCudaDevice<VecType> src,
                          ImageCudaDevice<typename VecType::VecTypef> dst,
                          float scale, float offset) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= dst.width_ || v >= dst.height_) return;

    dst.at(u, v) =
        src.at(u, v).ToVectorf() * scale + VecType::VecTypef(offset);
}

template<typename VecType>
__host__
void ImageCudaKernelCaller<VecType>::ConvertToFloat(
    ImageCuda<VecType> &src,
    ImageCuda<typename VecType::VecTypef> &dst,
    float scale, float offset) {
    const dim3 blocks(DIV_CEILING(src.width_, THREAD_2D_UNIT),
                      DIV_CEILING(src.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    ConvertToFloatKernel << < blocks, threads >> > (
        *src.device_, *dst.device_, scale, offset);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename VecType>
__global__
void ConvertRGBToIntensityKernel(ImageCudaDevice<VecType> src,
                                 ImageCudaDevice<Vector1f> dst) {

    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= dst.width_ || v >= dst.height_) return;

    VecType &rgb = src.at(u, v);
    dst.at(u, v) = Vector1f(
        (0.2990f * rgb(0) + 0.5870f * rgb(1) + 0.1140f * rgb(2)) / 255.0f);
}

template<typename VecType>
__host__
void ImageCudaKernelCaller<VecType>::ConvertRGBToIntensity(
    ImageCuda<VecType> &src, ImageCuda<Vector1f> &dst) {
    const dim3 blocks(DIV_CEILING(src.width_, THREAD_2D_UNIT),
                      DIV_CEILING(src.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    ConvertRGBToIntensityKernel << < blocks, threads >> > (
        *src.device_, *dst.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
} // cuda
} // open3d