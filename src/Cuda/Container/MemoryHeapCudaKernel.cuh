//
// Created by wei on 18-9-24.
//

#include "MemoryHeapCuda.cuh"
#include "LinkedListCuda.h"
#include "HashTableCuda.h"

namespace three {

template<typename T>
__global__
void ResetMemoryHeapKernel(MemoryHeapCudaServer<T> server) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < server.max_capacity_) {
		server.get_value(i) = T(); /* This is not necessary. */
		server.get_heap(i) = server.max_capacity_ - i - 1;
	}
}

template
__global__
void ResetMemoryHeapKernel<int>(MemoryHeapCudaServer<int> server);

template
__global__
void ResetMemoryHeapKernel<float>(MemoryHeapCudaServer<float> server);

template
__global__
void ResetMemoryHeapKernel<LinkedListNodeCuda<int>>
(MemoryHeapCudaServer<LinkedListNodeCuda<int>> server);

template
__global__
void ResetMemoryHeapKernel<LinkedListNodeCuda<SpatialEntry>>
(MemoryHeapCudaServer<LinkedListNodeCuda<SpatialEntry>> server);
}