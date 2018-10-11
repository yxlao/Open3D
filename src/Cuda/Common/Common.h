/**
 * Created by wei on 18-3-29.
 */

#ifndef _COMMON_H_
#define _COMMON_H_
#include <cmath>

/**
 * Make Eigen work on CUDA.
 * DO NOT USE CLANG!
 * - https://eigen.tuxfamily.org/dox/TopicCUDA.html
 * workaround issue between gcc >= 4.7 and cuda 5.5
 */
#if (defined __GNUC__) && (__GNUC__ > 4 || __GNUC_MINOR__ >= 7)
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128
#endif

#if defined(__CUDACC__)
#define __ALIGN__(n)  __align__(n)
/* Use these to avoid redundant conditional macro code ONLY in headers */
#define __HOST__ __host__
#define __DEVICE__ __device__
#define __HOSTDEVICE__ __host__ __device__
#define __GLOBAL__ __global__
#else
#define __HOST__
#define __DEVICE__
#define __HOSTDEVICE__
#define __GLOBAL__
#define __ALIGN__(n) alignas(n)
#define __int_as_float(n) float(int(n))
#endif

/* Basic types */
#ifndef uchar
typedef unsigned char uchar;
#endif
#ifndef uint
typedef unsigned int uint;
#endif

__HOSTDEVICE__
inline uchar safe_add(uchar a, uchar b) {
	return uchar(fminf(a + b, 255));
}
__HOSTDEVICE__
inline uchar safe_mul(float a, uchar b) {
	return uchar(fminf(a * b, 255));
}

/* @TODO: make this part modern, using enum, const, etc. */

/* Default values */
#define EMPTY_ENTRY (-1)
#define NULL_PTR (-1)

/* Error code */
#define SUCCESS  0
#define UNLOCKED 0

#define LOCKED    (-1)
#define FILE_NOT_FOUND (-2)
#define NODE_NOT_FOUND (-2)
#define ENTRY_EXISTED  (-3)

/* Volume configuration */
#define BLOCK_SIDE_LENGTH  8
#define SQR_BLOCK_SIDE_LENGTH (BLOCK_SIDE_LENGTH * BLOCK_SIDE_LENGTH)
#define BLOCK_SIZE            (BLOCK_SIDE_LENGTH * BLOCK_SIDE_LENGTH * BLOCK_SIDE_LENGTH)

#define THREAD_3D_UNIT   8
#define THREAD_2D_UNIT   16
#define THREAD_1D_UNIT   32
#define UPPER_ALIGN(a, b) ((a + b - 1) / b)

#define EPSILON (1e-6f)
#define MINF __int_as_float(0xff800000)
#define PINF __int_as_float(0x7f800000)
#define INV(x) (x == 0 ? PINF : 1.0f / x)

#define KEY_ESCAPE 27
#define KEY_SPACE  32
#define KEY_ENTER  13

#endif /* _COMMON_H_ */
