#include "ImageCuda.cuh"
#include "ImagePyramidCuda.cuh"
#include "ImageCudaKernel.cuh"
#include "TriangleMeshCuda.cuh"
#include "VectorCuda.h"
#include <Cuda/Common/Common.h>

namespace open3d {

class TriangleMeshCuda;

template
class ImageCuda<Vector1s>;

template
class ImageCuda<Vector4b>;

template
class ImageCuda<Vector3b>;

template
class ImageCuda<Vector1b>;

template
class ImageCuda<Vector4f>;

template
class ImageCuda<Vector3f>;

template
class ImageCuda<Vector1f>;

template
class ImagePyramidCuda<Vector1s, 3>;

template
class ImagePyramidCuda<Vector1s, 4>;

template
class ImagePyramidCuda<Vector4b, 3>;

template
class ImagePyramidCuda<Vector4b, 4>;

template
class ImagePyramidCuda<Vector3b, 3>;

template
class ImagePyramidCuda<Vector3b, 4>;

template
class ImagePyramidCuda<Vector1b, 3>;

template
class ImagePyramidCuda<Vector1b, 4>;

template
class ImagePyramidCuda<Vector4f, 3>;

template
class ImagePyramidCuda<Vector4f, 4>;

template
class ImagePyramidCuda<Vector3f, 3>;

template
class ImagePyramidCuda<Vector3f, 4>;

template
class ImagePyramidCuda<Vector1f, 3>;

template
class ImagePyramidCuda<Vector1f, 4>;

/** Vector **/
template
Vector4f operator*<float, 4>(float s, const Vector4f &vec);

template
Vector3f operator*<float, 3>(float s, const Vector3f &vec);

template
Vector1f operator*<float, 1>(float s, const Vector1f &vec);

/** Downsample **/
template
__global__
void DownsampleImageKernel<Vector1s>(
    ImageCudaServer<Vector1s> src, ImageCudaServer<Vector1s> dst,
    DownsampleMethod method);
template
__global__
void DownsampleImageKernel<Vector4b>(
    ImageCudaServer<Vector4b> src, ImageCudaServer<Vector4b> dst,
    DownsampleMethod method);
template
__global__
void DownsampleImageKernel<Vector3b>(
    ImageCudaServer<Vector3b> src, ImageCudaServer<Vector3b> dst,
    DownsampleMethod method);
template
__global__
void DownsampleImageKernel<Vector1b>(
    ImageCudaServer<Vector1b> src, ImageCudaServer<Vector1b> dst,
    DownsampleMethod method);
template
__global__
void DownsampleImageKernel<Vector4f>(
    ImageCudaServer<Vector4f> src, ImageCudaServer<Vector4f> dst,
    DownsampleMethod method);
template
__global__
void DownsampleImageKernel<Vector3f>(
    ImageCudaServer<Vector3f> src, ImageCudaServer<Vector3f> dst,
    DownsampleMethod method);
template
__global__
void DownsampleImageKernel<Vector1f>(
    ImageCudaServer<Vector1f> src, ImageCudaServer<Vector1f> dst,
    DownsampleMethod method);

/** Shift **/
template
__global__
void ShiftImageKernel<Vector1s>(
    ImageCudaServer<Vector1s> src, ImageCudaServer<Vector1s> dst,
    float dx, float dy, bool with_holes);
template
__global__
void ShiftImageKernel<Vector4b>(
    ImageCudaServer<Vector4b> src, ImageCudaServer<Vector4b> dst,
    float dx, float dy, bool with_holes);
template
__global__
void ShiftImageKernel<Vector3b>(
    ImageCudaServer<Vector3b> src, ImageCudaServer<Vector3b> dst,
    float dx, float dy, bool with_holes);
template
__global__
void ShiftImageKernel<Vector1b>(
    ImageCudaServer<Vector1b> src, ImageCudaServer<Vector1b> dst,
    float dx, float dy, bool with_holes);
template
__global__
void ShiftImageKernel<Vector4f>(
    ImageCudaServer<Vector4f> src, ImageCudaServer<Vector4f> dst,
    float dx, float dy, bool with_holes);
template
__global__
void ShiftImageKernel<Vector3f>(
    ImageCudaServer<Vector3f> src, ImageCudaServer<Vector3f> dst,
    float dx, float dy, bool with_holes);
template
__global__
void ShiftImageKernel<Vector1f>(
    ImageCudaServer<Vector1f> src, ImageCudaServer<Vector1f> dst,
    float dx, float dy, bool with_holes);

/** Gaussian **/
template
__global__
void GaussianImageKernel<Vector1s>(
    ImageCudaServer<Vector1s> src, ImageCudaServer<Vector1s> dst,
    const int kernel_idx, bool with_holes);
template
__global__
void GaussianImageKernel<Vector4b>(
    ImageCudaServer<Vector4b> src, ImageCudaServer<Vector4b> dst,
    const int kernel_idx, bool with_holes);
template
__global__
void GaussianImageKernel<Vector3b>(
    ImageCudaServer<Vector3b> src, ImageCudaServer<Vector3b> dst,
    const int kernel_idx, bool with_holes);
template
__global__
void GaussianImageKernel<Vector1b>(
    ImageCudaServer<Vector1b> src, ImageCudaServer<Vector1b> dst,
    const int kernel_idx, bool with_holes);
template
__global__
void GaussianImageKernel<Vector4f>(
    ImageCudaServer<Vector4f> src, ImageCudaServer<Vector4f> dst,
    const int kernel_idx, bool with_holes);
template
__global__
void GaussianImageKernel<Vector3f>(
    ImageCudaServer<Vector3f> src, ImageCudaServer<Vector3f> dst,
    const int kernel_idx, bool with_holes);
template
__global__
void GaussianImageKernel<Vector1f>(
    ImageCudaServer<Vector1f> src, ImageCudaServer<Vector1f> dst,
    const int kernel_idx, bool with_holes);

/** Bilateral **/
template
__global__
void BilateralImageKernel<Vector1s>(
    ImageCudaServer<Vector1s> src, ImageCudaServer<Vector1s> dst,
    const int kernel_idx, float val_sigma, bool with_holes);
template
__global__
void BilateralImageKernel<Vector4b>(
    ImageCudaServer<Vector4b> src, ImageCudaServer<Vector4b> dst,
    const int kernel_idx, float val_sigma, bool with_holes);
template
__global__
void BilateralImageKernel<Vector3b>(
    ImageCudaServer<Vector3b> src, ImageCudaServer<Vector3b> dst,
    const int kernel_idx, float val_sigma, bool with_holes);
template
__global__
void BilateralImageKernel<Vector1b>(
    ImageCudaServer<Vector1b> src, ImageCudaServer<Vector1b> dst,
    const int kernel_idx, float val_sigma, bool with_holes);
template
__global__
void BilateralImageKernel<Vector4f>(
    ImageCudaServer<Vector4f> src, ImageCudaServer<Vector4f> dst,
    const int kernel_idx, float val_sigma, bool with_holes);
template
__global__
void BilateralImageKernel<Vector3f>(
    ImageCudaServer<Vector3f> src, ImageCudaServer<Vector3f> dst,
    const int kernel_idx, float val_sigma, bool with_holes);
template
__global__
void BilateralImageKernel<Vector1f>(
    ImageCudaServer<Vector1f> src, ImageCudaServer<Vector1f> dst,
    const int kernel_idx, float val_sigma, bool with_holes);

/** Conversion **/
template
__global__
void ToFloatImageKernel<Vector1s>(
    ImageCudaServer<Vector1s>, ImageCudaServer<Vector1f> dst,
    float scale, float offset);
template
__global__
void ToFloatImageKernel<Vector4b>(
    ImageCudaServer<Vector4b>, ImageCudaServer<Vector4f> dst,
    float scale, float offset);
template
__global__
void ToFloatImageKernel<Vector3b>(
    ImageCudaServer<Vector3b>, ImageCudaServer<Vector3f> dst,
    float scale, float offset);
template
__global__
void ToFloatImageKernel<Vector1b>(
    ImageCudaServer<Vector1b>, ImageCudaServer<Vector1f> dst,
    float scale, float offset);
template
__global__
void ToFloatImageKernel<Vector4f>(
    ImageCudaServer<Vector4f>, ImageCudaServer<Vector4f> dst,
    float scale, float offset);
template
__global__
void ToFloatImageKernel<Vector3f>(
    ImageCudaServer<Vector3f>, ImageCudaServer<Vector3f> dst,
    float scale, float offset);
template
__global__
void ToFloatImageKernel<Vector1f>(
    ImageCudaServer<Vector1f>, ImageCudaServer<Vector1f> dst,
    float scale, float offset);

/** Sobel **/
template
__global__
void SobelImageKernel<Vector1s>(
    ImageCudaServer<Vector1s> src,
    ImageCudaServer<Vector1f> dx, ImageCudaServer<Vector1f> dy,
    bool with_holes);
template
__global__
void SobelImageKernel<Vector4b>(
    ImageCudaServer<Vector4b> src,
    ImageCudaServer<Vector4f> dx, ImageCudaServer<Vector4f> dy,
    bool with_holes);
template
__global__
void SobelImageKernel<Vector3b>(
    ImageCudaServer<Vector3b> src,
    ImageCudaServer<Vector3f> dx, ImageCudaServer<Vector3f> dy,
    bool with_holes);
template
__global__
void SobelImageKernel<Vector1b>(
    ImageCudaServer<Vector1b> src,
    ImageCudaServer<Vector1f> dx, ImageCudaServer<Vector1f> dy,
    bool with_holes);
template
__global__
void SobelImageKernel<Vector4f>(
    ImageCudaServer<Vector4f> src,
    ImageCudaServer<Vector4f> dx, ImageCudaServer<Vector4f> dy,
    bool with_holes);
template
__global__
void SobelImageKernel<Vector3f>(
    ImageCudaServer<Vector3f> src,
    ImageCudaServer<Vector3f> dx, ImageCudaServer<Vector3f> dy,
    bool with_holes);
template
__global__
void SobelImageKernel<Vector1f>(
    ImageCudaServer<Vector1f> src,
    ImageCudaServer<Vector1f> dx, ImageCudaServer<Vector1f> dy,
    bool with_holes);
}