/**
 * Created by wei on 18-9-25.
 */

#include "LinkedListCudaDevice.cuh"
#include "ArrayCuda.h"

namespace open3d {

namespace cuda {
template<typename T>
__global__
void InsertLinkedListKernel(LinkedListCudaDevice<T> server,
                            ArrayCudaDevice<T> data) {
    for (int i = 0; i < data.max_capacity_; ++i) {
        server.Insert(data[i]);
    }
}
template<typename T>
__host__
void LinkedListCudaKernelCaller<T>::
InsertLinkedListKernelCaller(LinkedListCudaDevice<T> &server,
                             ArrayCudaDevice<T> &data) {
    InsertLinkedListKernel << < 1, 1 >> > (server, data);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename T>
__global__
void FindLinkedListKernel(LinkedListCudaDevice<T> server,
                          ArrayCudaDevice<T> query) {
    for (int i = 0; i < query.max_capacity_; ++i) {
        if (NULLPTR_CUDA == server.Find(query[i])) {
            printf("val[%d] Not found!\n", i);
        }
    }
}
template<typename T>
__host__
void LinkedListCudaKernelCaller<T>::
FindLinkedListKernelCaller(LinkedListCudaDevice<T> &server,
                           ArrayCudaDevice<T> &query) {
    FindLinkedListKernel << < 1, 1 >> > (server, query);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename T>
__global__
void DeleteLinkedListKernel(LinkedListCudaDevice<T> server,
                            ArrayCudaDevice<T> query) {
    for (int i = 0; i < query.max_capacity_; ++i) {
        if (SUCCESS != server.FindAndDelete(query[i])) {
            printf("val[%d] Not found!\n", i);
        }
    }
}
template<typename T>
void LinkedListCudaKernelCaller<T>::
DeleteLinkedListKernelCaller(LinkedListCudaDevice<T> &server,
                             ArrayCudaDevice<T> &query) {
    DeleteLinkedListKernel << < 1, 1 >> > (server, query);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename T>
__global__
void ClearLinkedListKernel(LinkedListCudaDevice<T> server) {
    server.Clear();
}

template<typename T>
__host__
void LinkedListCudaKernelCaller<T>::
ClearLinkedListKernelCaller(LinkedListCudaDevice<T> &server) {
    ClearLinkedListKernel << < 1, 1 >> > (server);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename T>
__global__
void DownloadLinkedListKernel(LinkedListCudaDevice<T> server,
                              ArrayCudaDevice<T> data) {
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
DownloadLinkedListKernelCaller(LinkedListCudaDevice<T> &server,
                               ArrayCudaDevice<T> &data) {
    DownloadLinkedListKernel << < 1, 1 >> > (server, data);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
} // cuda
} // open3d