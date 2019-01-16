//
// Created by wei on 1/14/19.
//

#pragma once

#include <Cuda/Common/Common.h>
#include "ContainerClasses.h"

#include <cstdlib>
#include <memory>
#include <vector>
#include <Eigen/Eigen>

namespace open3d {
namespace cuda {

/** We assume rows are extendable, while colume sizes are fixed **/
template<typename T>
class Array2DCudaDevice {
private:
    T *data_;
    int *iterator_rows_;

public:
    int max_rows_;
    int max_cols_;
    int pitch_;

public:
    __HOSTDEVICE__ inline T *data() { return data_; }
    __DEVICE__ inline int rows() { return *iterator_rows_; }

    __DEVICE__ inline int expand_rows(int num);
    __DEVICE__ inline T *row(int r);
    __DEVICE__ T &at(int r, int c);
    __DEVICE__ T &operator()(int r, int c);

public:
    friend class Array2DCuda<T>;
};

template<typename T>
class Array2DCuda {
private:
    std::shared_ptr<Array2DCudaDevice<T>> server_ = nullptr;

public:
    int max_rows_;
    int max_cols_;
    int pitch_;

public:
    Array2DCuda();
    Array2DCuda(int max_rows, int max_cols);
    Array2DCuda(const Array2DCuda<T> &other);
    Array2DCuda<T> &operator=(const Array2DCuda<T> &other);
    ~Array2DCuda();

    void Create(int max_rows, int max_cols);
    void Release();
    void UpdateServer();

    void Upload(
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &
        matrix);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Download();

    void Fill(const T &val);
    void Memset(int val);

    int rows() const;
    void set_iterator_rows(int row_position);

    std::shared_ptr<Array2DCudaDevice<T>> &server() {
        return server_;
    }
    const std::shared_ptr<Array2DCudaDevice<T>> &server() const {
        return server_;
    }
};
} // cuda
} // open3d