/**
 * Created by wei on 18-3-29.
 */

#pragma once
#include <cmath>

namespace open3d {
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
#ifndef ushort
typedef unsigned short ushort;
#endif
#ifndef uint
typedef unsigned int uint;
#endif

/* @TODO: make this part modern, using enum, const, etc. */
#define NULLPTR_CUDA (-1)

#define LINKED_LIST_NODE_NOT_FOUND (-1)

#define HASH_ENTRY_EMPTY    (-1)
#define HASH_ENTRY_EXISTING (-2)
#define HASH_ENTRY_LOCKED   (-3)

#define SUCCESS  0

/* Atomic Lock */
#define UNLOCKED  0
#define LOCKED    (-1)

#define THREAD_3D_UNIT   8
#define THREAD_2D_UNIT   16
#define THREAD_1D_UNIT   64
#define DIV_CEILING(a, b) ((a + b - 1) / b)

#define EPSILON (1e-6f)
#define MINF __int_as_float(0xff800000)
#define PINF __int_as_float(0x7f800000)
#define INV(x) (x == 0 ? PINF : 1.0f / x)

#define KEY_ESCAPE 27
#define KEY_SPACE  32
#define KEY_ENTER  13

}
