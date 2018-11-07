//
// Created by wei on 18-9-24.
//

#include "MemoryHeapCuda.cuh"

namespace open3d {

template<typename T>
__global__
void ResetMemoryHeapKernel(MemoryHeapCudaServer<T> server) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < server.max_capacity_) {
		server.value_at(i) = T(); /* This is not necessary. */
        server.internal_addr_at(i) = i;
	}
}
}