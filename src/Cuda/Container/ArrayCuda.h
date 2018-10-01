/**
* Created by wei on 18-4-2.
*/

#pragma once

#include "ContainerClasses.h"
#include <Cuda/Common/Common.h>

#include <vector>

namespace three {

template<typename T>
class ArrayCudaServer {
private:
	/* atomicAdd works on int and unsigned int, so we prefer int than size_t */
	T* data_;
	int* iterator_;

public:
	int max_capacity_;

public:
	__DEVICE__ void push_back(T value);
	__DEVICE__ T& get(int index);

	friend class ArrayCuda<T>;
};

template<typename T>
class ArrayCuda {
private:
	ArrayCudaServer<T> server_;
	int max_capacity_;

public:
	ArrayCuda() { max_capacity_ = -1; }
	~ArrayCuda() = default;
	ArrayCuda(int max_capacity);

	void Create(int max_capacity);
	void Release();

	void Upload(std::vector<T> &data);
	/* Download valid parts (GPU push_back operations) */
	std::vector<T> Download();
	std::vector<T> DownloadAll();

	/* Fill is non-trivial assignment to specific values, needs kernel call */
	/* Memset is trivial setting, usually to all zero */
	void Fill(const T val);
	void Memset(const int val);
	void Clear();

	int size();
	int max_capacity() {
		return max_capacity_;
	}

	ArrayCudaServer<T>& server() {
		return server_;
	}
};

template<typename T>
__GLOBAL__
void FillArrayKernel(ArrayCudaServer<T> server, T val);

}
