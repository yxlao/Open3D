/**
 * Created by wei on 18-9-25.
 */

#include "LinkedListCudaDevice.cuh"
#include "ArrayCuda.h"

namespace open3d {
template<typename T>
__global__
void InsertLinkedListKernel(LinkedListCudaServer<T> server,
                            ArrayCudaServer<T> data) {
    for (int i = 0; i < data.max_capacity_; ++i) {
        server.Insert(data[i]);
    }
}
template<typename T>
__host__
void LinkedListCudaKernelCaller<T>::
    InsertLinkedListKernelCaller(LinkedListCudaServer<T> &server,
                                 ArrayCudaServer<T> &data) {
    InsertLinkedListKernel << < 1, 1 >> > (server, data);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename T>
__global__
void FindLinkedListKernel(LinkedListCudaServer<T> server,
                          ArrayCudaServer<T> query) {
    for (int i = 0; i < query.max_capacity_; ++i) {
        if (NULLPTR_CUDA == server.Find(query[i])) {
            printf("val[%d] Not found!\n", i);
        }
    }
}
template<typename T>
__host__
void LinkedListCudaKernelCaller<T>::
    FindLinkedListKernelCaller(LinkedListCudaServer<T> &server,
                               ArrayCudaServer<T> &query) {
    FindLinkedListKernel << < 1, 1 >> > (server, query);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename T>
__global__
void DeleteLinkedListKernel(LinkedListCudaServer<T> server,
                            ArrayCudaServer<T> query) {
    for (int i = 0; i < query.max_capacity_; ++i) {
        if (SUCCESS != server.FindAndDelete(query[i])) {
            printf("val[%d] Not found!\n", i);
        }
    }
}
template<typename T>
void LinkedListCudaKernelCaller<T>::
    DeleteLinkedListKernelCaller(LinkedListCudaServer<T> &server,
                                 ArrayCudaServer<T> &query) {
    DeleteLinkedListKernel << < 1, 1 >> > (server, query);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename T>
__global__
void ClearLinkedListKernel(LinkedListCudaServer<T> server) {
    server.Clear();
}

template<typename T>
__host__
void LinkedListCudaKernelCaller<T>::
    ClearLinkedListKernelCaller(LinkedListCudaServer<T> &server) {
    ClearLinkedListKernel << < 1, 1 >> > (server);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename T>
__global__
void DownloadLinkedListKernel(LinkedListCudaServer<T> server,
                              ArrayCudaServer<T> data) {
    int node_ptr = server.head_node_ptr();

    int cnt = 0;
    while (node_ptr != NULLPTR_CUDA) {
        assert(cnt < data.max_capacity_);
        LinkedListNodeCuda<T> &node = server.get_node(node_ptr);
        data[cnt] = node.data;
        node_ptr = node.next_node_ptr;
        ++cnt;
    }

    assert(cnt == data.max_capacity_);
}

template<typename T>
__host__
void LinkedListCudaKernelCaller<T>::
    DownloadLinkedListKernelCaller(LinkedListCudaServer<T> &server,
                                    ArrayCudaServer<T> &data) {
    DownloadLinkedListKernel << < 1, 1 >> > (server, data);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
}