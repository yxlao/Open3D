/**
* Created by wei on 18-4-2.
*/

#pragma once

#include "ContainerClasses.h"

#include <Cuda/Common/Common.h>

#include <cstdlib>
#include <memory>
#include <vector>

namespace open3d {

namespace cuda {

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
    __DEVICE__ inline T &at(size_t index);
    __DEVICE__ inline T &operator[](size_t index);

public:
    friend class ArrayCuda<T>;
};

template<typename T>
class ArrayCuda {
private:
    std::shared_ptr<ArrayCudaServer<T>> server_ = nullptr;

public:
    int max_capacity_;

public:
    ArrayCuda();
    explicit ArrayCuda(int max_capacity);
    ArrayCuda(const ArrayCuda<T> &other);
    ArrayCuda<T> &operator=(const ArrayCuda<T> &other);
    ~ArrayCuda();

    void Resize(int max_capacity);

    void Create(int max_capacity);
    void Release();

    void CopyFromDeviceArray(const T *array, int size);
    void CopyTo(ArrayCuda<T> &other) const;
    void Upload(std::vector<T> &data);
    void Upload(const T *data, int size);

    /** Download size() elements **/
    std::vector<T> Download();
    /** Download max_capacity_ elements **/
    std::vector<T> DownloadAll();

    /** Fill non-trivial values **/
    void Fill(const T &val);
    /** Fill trivial values (e.g. 0, 0xff)  **/
    void Memset(int val);
    void Clear();

    int size() const;
    void set_iterator(int iterator_position);

    std::shared_ptr<ArrayCudaServer<T>> &server() {
        return server_;
    }
    const std::shared_ptr<ArrayCudaServer<T>> &server() const {
        return server_;
    }
};

/** For less instantiation code! **/
template<typename T>
class ArrayCudaKernelCaller {
public:
    __HOST__ static void FillArrayKernelCaller(ArrayCudaServer<T> &server,
                                               const T &val, int max_capacity);
};

template<typename T>
__GLOBAL__
void FillArrayKernel(ArrayCudaServer<T> server, T val);

} // cuda
} // open3d