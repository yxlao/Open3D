/**
 * Created by wei on 18-9-25.
 */

#include "LinkedListCudaDevice.cuh"
#include "ArrayCuda.h"

namespace open3d {

namespace cuda {
template<typename T>
__global__
void InsertKernel(LinkedListCudaDevice<T> list,
                  ArrayCudaDevice<T> data) {
    for (int i = 0; i < data.max_capacity_; ++i) {
        list.Insert(data[i]);
    }
}
template<typename T>
__host__
void LinkedListCudaKernelCaller<T>::Insert(
    LinkedListCuda<T> &list, ArrayCuda<T> &data) {
    InsertKernel << < 1, 1 >> > (*list.device_, *data.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename T>
__global__
void FindKernel(LinkedListCudaDevice<T> list,
                          ArrayCudaDevice<T> query) {
    for (int i = 0; i < query.max_capacity_; ++i) {
        if (NULLPTR_CUDA == list.Find(query[i])) {
            printf("val[%d] Not found!\n", i);
        }
    }
}
template<typename T>
__host__
void LinkedListCudaKernelCaller<T>::Find(LinkedListCuda<T> &list,
                                         ArrayCuda<T> &query) {
    FindKernel << < 1, 1 >> > (*list.device_, *query.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename T>
__global__
void DeleteKernel(LinkedListCudaDevice<T> list,
                            ArrayCudaDevice<T> query) {
    for (int i = 0; i < query.max_capacity_; ++i) {
        if (SUCCESS != list.FindAndDelete(query[i])) {
            printf("val[%d] Not found!\n", i);
        }
    }
}
template<typename T>
void LinkedListCudaKernelCaller<T>::Delete(
    LinkedListCuda<T> &list, ArrayCuda<T> &query) {
    DeleteKernel << < 1, 1 >> > (*list.device_, *query.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename T>
__global__
void ClearKernel(LinkedListCudaDevice<T> list) {
    list.Clear();
}

template<typename T>
__host__
void LinkedListCudaKernelCaller<T>::Clear(LinkedListCuda<T> &list) {
    ClearKernel << < 1, 1 >> > (*list.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename T>
__global__
void DownloadKernel(LinkedListCudaDevice<T> list,
                    ArrayCudaDevice<T> data) {
    int node_ptr = list.head_node_ptr();

    int cnt = 0;
    while (node_ptr != NULLPTR_CUDA) {
        assert(cnt < data.max_capacity_);
        LinkedListNodeCuda<T> &node = list.get_node(node_ptr);
        data[cnt] = node.data;
        node_ptr = node.next_node_ptr;
        ++cnt;
    }

    assert(cnt == data.max_capacity_);
}

template<typename T>
__host__
void LinkedListCudaKernelCaller<T>::Download(
    LinkedListCuda<T> &list, ArrayCuda<T> &data) {
    DownloadKernel << < 1, 1 >> > (*list.device_, *data.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
} // cuda
} // open3d