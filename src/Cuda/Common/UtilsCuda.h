//
// Created by wei on 9/27/18.
//

#ifndef OPEN3D_UTILSCUDA_H
#define OPEN3D_UTILSCUDA_H

#include "Common.h"
#include "HelperCuda.h"
#include <cstdlib>

namespace three {

enum MemcpyKind {
	HostToHost,
	HostToDevice,
	DeviceToDevice,
	DeviceToHost
};

void CudaMalloc(void** ptr, size_t size);

void CudaFree(void* ptr);

void CudaMemcpy(void *dst, const void *src, size_t size, enum MemcpyKind kind);

void CudaMemset(void *ptr, int value, size_t size);

void CudaSynchronize();

}

#endif //OPEN3D_UTILSCUDA_H
