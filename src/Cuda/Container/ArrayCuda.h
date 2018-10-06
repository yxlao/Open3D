/**
* Created by wei on 18-4-2.
*/

#pragma once

#include "ContainerClasses.h"
#include <Cuda/Common/Common.h>
#include <cstdlib>
#include <vector>

namespace three {

template<typename T>
class ArrayCudaServer {
private:
	/* atomicAdd works on int and unsigned int, so we prefer int than size_t */
	T *data_;
	int *iterator_;

public:
	int max_capacity_;

public:
	/** This is a CPU pointer for shared reference counting.
	 *  How many ArrayCuda clients are using this server?
	 */
	int *ref_count_ = nullptr;

public:
	__HOSTDEVICE__ T* &data() { return data_; }
	__DEVICE__ void push_back(T value);
	__DEVICE__ T &get(size_t index);

	friend class ArrayCuda<T>;
};

template<typename T>
class ArrayCuda {
private:
	ArrayCudaServer<T> server_;
	int max_capacity_;

public:
	ArrayCuda();
	explicit ArrayCuda(int max_capacity);
	ArrayCuda(const ArrayCuda<T> &other);
	ArrayCuda<T>& operator=(const ArrayCuda<T> &other);
	~ArrayCuda();

	void Create(int max_capacity);
	void Release();

	void CopyTo(ArrayCuda<T> &other) const;
	void Upload(std::vector<T> &data);

	/* Download valid parts (.size() elements by GPU push_back operations) */
	std::vector<T> Download();
	std::vector<T> DownloadAll();

	/* Fill is non-trivial assignment to specific values, needs kernel call */
	/* Memset is trivial setting, usually to all zero */
	void Fill(const T val);
	void Memset(const int val);
	void Clear();

	int size();
	void set_size(int iterator_position);

	int max_capacity() const {
		return max_capacity_;
	}

	ArrayCudaServer<T> &server() {
		return server_;
	}
	const ArrayCudaServer<T> &server() const {
		return server_;
	}
};

template<typename T>
__GLOBAL__
void FillArrayKernel(ArrayCudaServer<T> server, T val);

}
