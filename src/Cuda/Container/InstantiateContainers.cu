/**
 * Created by wei on 18-9-23
 */

#include "ArrayCudaDevice.cuh"
#include "ArrayCudaKernel.cuh"
#include "MemoryHeapCuda.cuh"
#include "MemoryHeapCudaKernel.cuh"
#include "LinkedListCuda.cuh"
#include "LinkedListCudaKernel.cuh"
#include "HashTableCuda.cuh"
#include "HashTableCudaKernel.cuh"

#include <Cuda/Common/VectorCuda.h>

namespace open3d {

template class ArrayCudaServer<int>;
template class ArrayCudaServer<float>;
template class ArrayCudaServer<Vector3i>;
template class ArrayCudaServer<Vector3f>;
template class ArrayCudaServer<HashEntry<Vector3i>>;
template class ArrayCudaServer<LinkedListCudaServer<HashEntry<Vector3i>>>;

template void FillArrayKernelCaller<int>(
    ArrayCudaServer<int> &server, const int &val, int max_capacity);
template void FillArrayKernelCaller<float>(
    ArrayCudaServer<float> &server, const float &val, int max_capacity);
template void FillArrayKernelCaller<Vector3i>(
    ArrayCudaServer<Vector3i> &server, const Vector3i &val, int max_capacity);
template void FillArrayKernelCaller<Vector3f>(
    ArrayCudaServer<Vector3f> &server, const Vector3f &val, int max_capacity);
template void FillArrayKernelCaller<HashEntry<Vector3i>>(
    ArrayCudaServer<HashEntry<Vector3i>> &server,
    const HashEntry<Vector3i> &val, int max_capacity);
template void FillArrayKernelCaller<LinkedListCudaServer<HashEntry<Vector3i>>>(
    ArrayCudaServer<LinkedListCudaServer<HashEntry<Vector3i>>> &server,
    const LinkedListCudaServer<HashEntry<Vector3i>> &val, int max_capacity);

template
class MemoryHeapCuda<int>;

template
class MemoryHeapCuda<float>;

template
class MemoryHeapCuda<LinkedListNodeCuda<int>>;

template
class LinkedListCuda<int>;

template
class HashTableCuda<Vector3i, int, SpatialHasher>;
}