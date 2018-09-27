/**
 * Created by wei on 18-9-27
 */

#include "UtilsCuda.h"
#include "HelperCuda.h"

namespace three {

void CudaMalloc(void** ptr, size_t size) {
	checkCudaErrors(cudaMalloc(ptr, size));
}

void CudaFree(void* ptr) {
	checkCudaErrors(cudaFree(ptr));
}

void CudaMemcpy(void *dst, const void *src, size_t size,
	enum MemcpyKind kind) {
	cudaMemcpyKind kind_;
	switch (kind) {
		case HostToHost:
			kind_ = cudaMemcpyHostToHost; break;
		case HostToDevice:
			kind_ = cudaMemcpyHostToDevice; break;
		case DeviceToDevice:
			kind_ = cudaMemcpyDeviceToDevice; break;
		case DeviceToHost:
			kind_ = cudaMemcpyDeviceToHost; break;
		default:
			/* Should never reach here */
			break;
	}
	checkCudaErrors(cudaMemcpy(dst, src, size, kind_));
}

void CudaMemset(void *ptr, int value, size_t size) {
	checkCudaErrors(cudaMemset(ptr, value, size));
}

void CudaSynchronize() {
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

}