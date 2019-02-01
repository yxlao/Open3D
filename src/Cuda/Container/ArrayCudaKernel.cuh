/**
 * Created by wei on 18-4-2.
 */

#include "ArrayCudaDevice.cuh"

namespace open3d {

namespace cuda {
template<typename T>
__global__
void FillKernel(ArrayCudaDevice<T> device, T val) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < device.max_capacity_) {
        device.at(i) = val;
    }
}

template<typename T>
__host__
void ArrayCudaKernelCaller<T>::Fill(
    ArrayCuda<T> &array, const T &val) {

    const int blocks = DIV_CEILING(array.max_capacity_, THREAD_1D_UNIT);
    const int threads = THREAD_1D_UNIT;
    FillKernel << < blocks, threads >> > (*array.device_, val);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
} // cuda
} // open3d