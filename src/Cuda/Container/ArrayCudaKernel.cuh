/**
 * Created by wei on 18-4-2.
 */

#include "ArrayCuda.cuh"

#include <cassert>

namespace open3d {

template<typename T>
__global__
void FillArrayKernel(ArrayCudaServer<T> server, T val) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < server.max_capacity_) {
        server.get(i) = val;
    }
}
}