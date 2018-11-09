//
// Created by wei on 11/9/18.
//

#include "ArrayCudaHost.hpp"
#include "HashTableCudaHost.hpp"
#include "MemoryHeapCudaHost.hpp"
#include "LinkedListCuda.h"

namespace open3d {

template class ArrayCuda<int>;
template class ArrayCuda<float>;
template class ArrayCuda<Vector3i>;
template class ArrayCuda<Vector3f>;
template class ArrayCuda<HashEntry<Vector3i>>;
template class ArrayCuda<LinkedListCudaServer<HashEntry<Vector3i>>>;

/** Memory Heap **/
template class MemoryHeapCuda<int>;
template class MemoryHeapCuda<float>;
template class MemoryHeapCuda<LinkedListNodeCuda<int>>;

template class HashTableCuda<Vector3i, int, SpatialHasher>;


}