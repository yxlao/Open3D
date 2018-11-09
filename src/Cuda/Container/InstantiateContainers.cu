/**
 * Created by wei on 18-9-23
 */

#include "ArrayCudaDevice.cuh"
#include "ArrayCudaKernel.cuh"
#include "MemoryHeapCudaDevice.cuh"
#include "MemoryHeapCudaKernel.cuh"
#include "LinkedListCuda.cuh"
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

/** Memory Heap **/
template class MemoryHeapCudaServer<int>;
template class MemoryHeapCudaServer<float>;
template class MemoryHeapCudaServer<LinkedListNodeCuda<int>>;
template class MemoryHeapCudaServer<LinkedListNodeCuda<HashEntry<Vector3i>>>;

template void ResetMemoryHeapKernelCaller<int>(
    MemoryHeapCudaServer<int>&server, int max_capacity);
template void ResetMemoryHeapKernelCaller<float>(
    MemoryHeapCudaServer<float>&server, int max_capacity);
template void ResetMemoryHeapKernelCaller<LinkedListNodeCuda<int>>(
    MemoryHeapCudaServer<LinkedListNodeCuda<int>>&server, int max_capacity);
template void ResetMemoryHeapKernelCaller<LinkedListNodeCuda<HashEntry<Vector3i>>>(
    MemoryHeapCudaServer<LinkedListNodeCuda<HashEntry<Vector3i>>> &server, int max_capacity);

template
class LinkedListCuda<int>;

/** HashTable **/
template class HashTableCudaServer<Vector3i, int, SpatialHasher>;
template void CreateHashTableEntriesKernelCaller<Vector3i, int, SpatialHasher>(
    HashTableCudaServer<Vector3i, int, SpatialHasher>& server,
    int bucket_count);
template void ReleaseHashTableEntriesKernelCaller<Vector3i, int, SpatialHasher>(
    HashTableCudaServer<Vector3i, int, SpatialHasher>& server,
    int bucket_count);
template void ResetHashTableEntriesKernelCaller<Vector3i, int, SpatialHasher>(
    HashTableCudaServer<Vector3i, int, SpatialHasher>& server,
    int bucket_count);
template void GetHashTableAssignedEntriesKernelCaller<Vector3i, int, SpatialHasher>(
    HashTableCudaServer<Vector3i, int, SpatialHasher> &server,
    int bucket_count);
template void InsertHashTableEntriesKernelCaller<Vector3i, int, SpatialHasher>(
    HashTableCudaServer<Vector3i, int, SpatialHasher>& server,
    ArrayCudaServer<Vector3i>& keys,
    ArrayCudaServer<int> &values,
    int num_pairs, int bucket_count);
template void DeleteHashTableEntriesKernelCaller<Vector3i, int, SpatialHasher>(
    HashTableCudaServer<Vector3i, int, SpatialHasher> &server,
    ArrayCudaServer<Vector3i> &keys, int num_keys, int bucket_count);
template void ProfileHashTableKernelCaller(
    HashTableCudaServer<Vector3i, int, SpatialHasher> &server,
    ArrayCudaServer<int> &array_entry_count,
    ArrayCudaServer<int> &linked_list_entry_count,
    int bucket_count);
}