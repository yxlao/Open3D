/**
 * Created by wei on 18-4-2.
 */

#include "ArrayCudaDevice.cuh"

namespace open3d {

namespace cuda {
template<typename T>
__global__
void FillArrayKernel(ArrayCudaDevice<T> server, T val) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < server.max_capacity_) {
        server.at(i) = val;
    }
}

template<typename T>
__host__
void ArrayCudaKernelCaller<T>::FillArrayKernelCaller(
    ArrayCudaDevice<T> &server, const T &val, int max_capacity) {

    const int blocks = DIV_CEILING(max_capacity, THREAD_1D_UNIT);
    const int threads = THREAD_1D_UNIT;
    FillArrayKernel << < blocks, threads >> > (server, val);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
} // cuda
} // open3d