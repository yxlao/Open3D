/**
* Created by wei on 18-4-2.
*/

#pragma once

#include "ContainerClasses.h"
#include <Cuda/Common/Common.h>
#include <cstdlib>
#include <vector>
#include <memory>

namespace open3d {

template<typename T>
class ArrayCudaServer {
private:
    /* atomicAdd works on int and unsigned int, so we prefer int than size_t */
    T *data_;
    int *iterator_;

public:
    int max_capacity_;

public:
    __HOSTDEVICE__ inline T *&data() { return data_; }
    __DEVICE__ inline int size() { return *iterator_; }
    __DEVICE__ inline int push_back(T value);
    __DEVICE__ inline T &get(size_t index);
    __DEVICE__ inline T& operator[] (size_t index);

public:
    friend class ArrayCuda<T>;
};

template<typename T>
class ArrayCuda {
private:
    std::shared_ptr<ArrayCudaServer<T>> server_ = nullptr;
    int max_capacity_;

public:
    ArrayCuda();
    explicit ArrayCuda(int max_capacity);
    ArrayCuda(const ArrayCuda<T> &other);
    ArrayCuda<T> &operator=(const ArrayCuda<T> &other);
    ~ArrayCuda();

    void Create(int max_capacity);
    void Release();

    void FromCudaArray(const T* array, int size);
    void CopyTo(ArrayCuda<T> &other) const;
    void Upload(std::vector<T> &data);
    void Upload(const T *data, int size);

    /* Download valid parts (.size() elements by GPU push_back operations) */
    std::vector<T> Download();
    std::vector<T> DownloadAll();

    /* Fill is non-trivial assignment to specific values, needs kernel call */
    /* Memset is trivial setting, usually to all zero */
    void Fill(const T& val);
    void Memset(int val);
    void Clear();

    int size() const;
    void set_size(int iterator_position);

    int max_capacity() const {
        return max_capacity_;
    }

    std::shared_ptr<ArrayCudaServer<T>> &server() {
        return server_;
    }
    const std::shared_ptr<ArrayCudaServer<T>> &server() const {
        return server_;
    }
};

template<typename T>
__GLOBAL__
void FillArrayKernel(ArrayCudaServer<T> server, T val);

}
