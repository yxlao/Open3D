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

/** HashTable **/
template
__global__
void FillArrayKernel<int>(ArrayCudaServer<int>, int val);

template
__global__
void FillArrayKernel<float>(ArrayCudaServer<float>, float val);

template
__global__
void CreateHashTableEntriesKernel<Vector3i, int, SpatialHasher>
	(HashTableCudaServer<Vector3i, int, SpatialHasher> server);

template
__global__
void ReleaseHashTableEntriesKernel<Vector3i, int, SpatialHasher>
	(HashTableCudaServer<Vector3i, int, SpatialHasher> server);

template
__global__
void ResetHashTableEntriesKernel<Vector3i, int, SpatialHasher>
	(HashTableCudaServer<Vector3i, int, SpatialHasher> server);

template
__global__
void GetHashTableAssignedEntriesKernel<Vector3i, int, SpatialHasher>
	(HashTableCudaServer<Vector3i, int, SpatialHasher> server);

template
__global__
void InsertHashTableEntriesKernel<Vector3i, int, SpatialHasher>
	(HashTableCudaServer<Vector3i, int, SpatialHasher> server,
	 Vector3i* keys, int *values, const int num_pairs);

template
__global__
void DeleteHashTableEntriesKernel<Vector3i, int, SpatialHasher>
	(HashTableCudaServer<Vector3i, int, SpatialHasher> server,
	 Vector3i* keys, const int num_pairs);

template
__global__
void ProfileHashTableKernel<Vector3i, int, SpatialHasher>
	(HashTableCudaServer<Vector3i, int, SpatialHasher> server,
	 int *array_entry_count, int *list_entry_count);

/** LinkedList **/
template
__global__
void InsertLinkedListKernel<int>
	(LinkedListCudaServer<int> server, int *data, const int N);

template
__global__
void ClearLinkedListKernel<int>(LinkedListCudaServer<int> server);

template
__global__
void FindLinkedListKernel<int>
	(LinkedListCudaServer<int> server, int *query, const int N);

template
__global__
void DeleteLinkedListKernel<int>
	(LinkedListCudaServer<int> server, int *query, const int N);

template
__global__
void DownloadLinkedListKernel<int>
	(LinkedListCudaServer<int> server, int *data, const int N);

/** MemoryHeap **/
template
__global__
void ResetMemoryHeapKernel<int>(MemoryHeapCudaServer<int> server);

template
__global__
void ResetMemoryHeapKernel<float>(MemoryHeapCudaServer<float> server);

template
__global__
void ResetMemoryHeapKernel<LinkedListNodeCuda<int>>
	(MemoryHeapCudaServer<LinkedListNodeCuda<int>> server);

template
__global__
void ResetMemoryHeapKernel<LinkedListNodeCuda<SpatialEntry>>
	(MemoryHeapCudaServer<LinkedListNodeCuda<SpatialEntry>> server);
}