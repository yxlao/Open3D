/**
 * Created by wei on 18-4-2.
 */

#ifndef _ARRAY_CUDA_CUH_
#define _ARRAY_CUDA_CUH_

#include "ArrayCuda.h"
#include <Cuda/Common/UtilsCuda.h>

#include <cuda_runtime.h>
#include <cassert>

namespace three {

/**
 * Server end
 */
template<typename T>
__device__
void ArrayCudaServer<T>::push_back(T value) {
	assert(*iterator_ < max_capacity_);
	int addr = atomicAdd(iterator_, 1);
	data_[addr] = value;
}

template<typename T>
__device__
T &ArrayCudaServer<T>::get(int index) {
	assert(index >= 0 && index < max_capacity_);
	return data_[index];
}

/**
 * Client end
 */
template<typename T>
ArrayCuda<T>::ArrayCuda(int max_capacity) {
	Init(max_capacity);
}

template<typename T>
void ArrayCuda<T>::Init(int max_capacity) {
  max_capacity_ = max_capacity;

  server_.max_capacity_ = max_capacity;
  CheckCuda(cudaMalloc((void**)&server_.data_, sizeof(T) * max_capacity));
  CheckCuda(cudaMemset(server_.data_, 0, sizeof(T) * max_capacity));
  CheckCuda(cudaMalloc((void**)&server_.iterator_, sizeof(int)));
  CheckCuda(cudaMemset(server_.iterator_, 0, sizeof(int)));
}

template<typename T>
void ArrayCuda<T>::Destroy() {
	CheckCuda(cudaFree(server_.data_));
	CheckCuda(cudaFree(server_.iterator_));
}

template<typename T>
void ArrayCuda<T>::Upload(std::vector<T> &data) {
	int size = data.size();
	assert(size < max_capacity_);
	CheckCuda(cudaMemcpy(server_.data_, data.data(), sizeof(T) * size,
		cudaMemcpyHostToDevice));
	CheckCuda(cudaMemcpy(server_.iterator_, &size, sizeof(int), cudaMemcpyHostToDevice));
}

template<class T>
std::vector<T> ArrayCuda<T>::Download() {
	std::vector<T> ret;
	int iterator_count = size();
	ret.resize(iterator_count);

	CheckCuda(cudaMemcpy(ret.data(), server_.data_,
		sizeof(T) * iterator_count, cudaMemcpyDeviceToHost));

	return ret;
}

template<typename T>
std::vector<T> ArrayCuda<T>::DownloadAll() {
	std::vector<T> ret;
	ret.resize(max_capacity_);

	CheckCuda(cudaMemcpy(ret.data(), server_.data_,
		sizeof(T) * max_capacity_, cudaMemcpyDeviceToHost));

	return ret; /* RVO will handle this */
}

template<typename T>
void ArrayCuda<T>::Fill(const T val) {
	const int threads = THREAD_1D_UNIT;
	const int blocks = UPPER_ALIGN(max_capacity_, THREAD_1D_UNIT);
	FillArrayKernel<<<blocks, threads>>>(server_, val);
	CheckCuda(cudaDeviceSynchronize());
}

template<typename T>
void ArrayCuda<T>::Memset(const int val) {
	CheckCuda(cudaMemset(server_.data_, val, sizeof(T) * max_capacity_));
}

template<class T>
void ArrayCuda<T>::Clear(){
	CheckCuda(cudaMemset(server_.iterator_, 0, sizeof(int)));
}

template<class T>
int ArrayCuda<T>::size() {
	int ret;
	CheckCuda(cudaMemcpy(&ret, server_.iterator_, sizeof(int), cudaMemcpyDeviceToHost));
	return ret;
}
}
#endif