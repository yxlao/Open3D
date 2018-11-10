/**
 * Created by wei on 18-9-23
 */

#include "ArrayCudaDevice.cuh"
#include "ArrayCudaKernel.cuh"
#include "MemoryHeapCudaDevice.cuh"
#include "MemoryHeapCudaKernel.cuh"
#include "LinkedListCudaDevice.cuh"
#include "LinkedListCudaKernel.cuh"
#include "HashTableCudaDevice.cuh"
#include "HashTableCudaKernel.cuh"

#include <Cuda/Common/VectorCuda.h>

namespace open3d {

/** ArrayCuda **/
template class ArrayCudaServer<int>;
template class ArrayCudaServer<float>;
template class ArrayCudaServer<Vector3i>;
template class ArrayCudaServer<Vector3f>;
template class ArrayCudaServer<HashEntry<Vector3i>>;
template class ArrayCudaServer<LinkedListCudaServer<HashEntry<Vector3i>>>;

template class ArrayCudaKernelCaller<int>;
template class ArrayCudaKernelCaller<float>;
template class ArrayCudaKernelCaller<Vector3i>;
template class ArrayCudaKernelCaller<Vector3f>;
template class ArrayCudaKernelCaller<HashEntry<Vector3i>>;
template class ArrayCudaKernelCaller<LinkedListCudaServer<HashEntry<Vector3i>>>;

/** Memory Heap **/
template class MemoryHeapCudaServer<int>;
template class MemoryHeapCudaServer<float>;
template class MemoryHeapCudaServer<LinkedListNodeCuda<int>>;
template class MemoryHeapCudaServer<LinkedListNodeCuda<HashEntry<Vector3i>>>;

template class MemoryHeapCudaKernelCaller<int>;
template class MemoryHeapCudaKernelCaller<float>;
template class MemoryHeapCudaKernelCaller<LinkedListNodeCuda<int>>;
template class MemoryHeapCudaKernelCaller<LinkedListNodeCuda<HashEntry<Vector3i>>>;

/** LinkedList **/
template class LinkedListCudaServer<int>;
template class LinkedListCudaKernelCaller<int>;

/** HashTable **/
template class HashTableCudaServer<Vector3i, int, SpatialHasher>;
template class HashTableCudaKernelCaller<Vector3i, int, SpatialHasher>;
}