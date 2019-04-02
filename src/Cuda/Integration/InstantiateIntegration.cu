//
// Created by wei on 10/10/18.
//

#include <Cuda/Container/HashTableCudaDevice.cuh>
#include <Cuda/Container/HashTableCudaKernel.cuh>

#include "UniformTSDFVolumeCudaDevice.cuh"
#include "UniformTSDFVolumeCudaKernel.cuh"
#include "UniformMeshVolumeCudaDevice.cuh"
#include "UniformMeshVolumeCudaKernel.cuh"
#include "ScalableTSDFVolumeCudaDevice.cuh"
#include "ScalableTSDFVolumeCudaKernel.cuh"
#include "ScalableMeshVolumeCudaDevice.cuh"
#include "ScalableMeshVolumeCudaKernel.cuh"

namespace open3d {
namespace cuda {
template
class HashTableCudaKernelCaller
    <Vector3i, UniformTSDFVolumeCudaDevice, SpatialHasher>;
template
class MemoryHeapCudaKernelCaller<UniformTSDFVolumeCudaDevice>;

}
}