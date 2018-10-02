/**
 * Created by wei on 18-9-25.
 */

#include "LinkedListCuda.cuh"

namespace three {
template<typename T>
__global__
void InsertLinkedListKernel(LinkedListCudaServer<T> server, T *data, const int N) {
	for (int i = 0; i < N; ++i) {
		server.Insert(data[i]);
	}
}

template<typename T>
__global__
void FindLinkedListKernel(LinkedListCudaServer<T> server, T* query, const int N) {
	for (int i = 0; i < N; ++i) {
		if (NODE_NOT_FOUND == server.Find(query[i])) {
			printf("val[%d] Not found!\n", i);
		}
	}
}

template<typename T>
__global__
void DeleteLinkedListKernel(LinkedListCudaServer<T> server, T* query, const int N) {
	for (int i = 0; i < N; ++i) {
		if (SUCCESS != server.FindAndDelete(query[i])) {
			printf("val[%d] Not found!\n", i);
		}
	}
}

template<typename T>
__global__
void ClearLinkedListKernel(LinkedListCudaServer<T> server) {
	server.Clear();
}

template<typename T>
__global__
void DownloadLinkedListKernel(LinkedListCudaServer<T> server, T* data, const int N) {
	int node_ptr = server.head_node_ptr();

	int cnt = 0;
	while (node_ptr != NULL_PTR) {
		assert(cnt < N);
		LinkedListNodeCuda<T> &node = server.get_node(node_ptr);
		data[cnt] = node.data;
		node_ptr = node.next_node_ptr;
		++cnt;
	}

	assert(cnt == N);
}
}