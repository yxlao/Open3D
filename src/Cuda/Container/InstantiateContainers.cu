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

#include <Cuda/Common/LinearAlgebraCuda.h>

namespace open3d {

namespace cuda {

/** ArrayCuda **/
template
class ArrayCudaDevice<int>;
template
class ArrayCudaDevice<float>;
template
class ArrayCudaDevice<Vector3i>;
template
class ArrayCudaDevice<Vector4i>;
template
class ArrayCudaDevice<Vector3f>;
template
class ArrayCudaDevice<Vector2i>;
template
class ArrayCudaDevice<HashEntry<Vector3i>>;
template
class ArrayCudaDevice<LinkedListCudaDevice<HashEntry<Vector3i>>>;

template
class ArrayCudaKernelCaller<int>;
template
class ArrayCudaKernelCaller<float>;
template
class ArrayCudaKernelCaller<Vector3i>;
template
class ArrayCudaKernelCaller<Vector4i>;
template
class ArrayCudaKernelCaller<Vector3f>;
template
class ArrayCudaKernelCaller<Vector2i>;
template
class ArrayCudaKernelCaller<HashEntry<Vector3i>>;
template
class ArrayCudaKernelCaller<LinkedListCudaDevice<HashEntry<Vector3i>>>;

/** Memory Heap **/
template
class MemoryHeapCudaDevice<int>;
template
class MemoryHeapCudaDevice<float>;
template
class MemoryHeapCudaDevice<LinkedListNodeCuda<int>>;
template
class MemoryHeapCudaDevice<LinkedListNodeCuda<HashEntry<Vector3i>>>;

template
class MemoryHeapCudaKernelCaller<int>;
template
class MemoryHeapCudaKernelCaller<float>;
template
class MemoryHeapCudaKernelCaller<LinkedListNodeCuda<int>>;
template
class MemoryHeapCudaKernelCaller<LinkedListNodeCuda<HashEntry<Vector3i>>>;

/** LinkedList **/
template
class LinkedListCudaDevice<int>;
template
class LinkedListCudaKernelCaller<int>;

/** HashTable **/
template
class HashTableCudaDevice<Vector3i, int, SpatialHasher>;
template
class HashTableCudaKernelCaller<Vector3i, int, SpatialHasher>;
} // cuda
} // open3d