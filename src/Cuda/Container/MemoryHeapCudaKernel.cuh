//
// Created by wei on 18-9-24.
//

#include "MemoryHeapCudaDevice.cuh"

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

template<typename T>
void ResetMemoryHeapKernelCaller(MemoryHeapCudaServer<T>& server,
								 int max_capacity) {
	const int blocks = DIV_CEILING(max_capacity, THREAD_1D_UNIT);
	const int threads = THREAD_1D_UNIT;

	ResetMemoryHeapKernel << < blocks, threads >> > (server);
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
}
}