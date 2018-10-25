/**
 * Created by wei on 18-9-23
 */

#include "ArrayCuda.cuh"
#include "ArrayCudaKernel.cuh"
#include "MemoryHeapCuda.cuh"
#include "MemoryHeapCudaKernel.cuh"
#include "LinkedListCuda.cuh"
#include "LinkedListCudaKernel.cuh"
#include "HashTableCuda.cuh"
#include "HashTableCudaKernel.cuh"

#include <Cuda/Geometry/VectorCuda.h>

namespace open3d {
template
class ArrayCuda<int>;

template
class ArrayCuda<float>;

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