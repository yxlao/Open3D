/**
 * Created by wei on 18-9-23
 */

#include "ArrayCuda.cuh"
#include "MemoryHeapCuda.cuh"
#include "LinkedListCuda.cuh"
#include "HashTableCuda.cuh"

#include <Cuda/Geometry/Vector.h>

namespace three {
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