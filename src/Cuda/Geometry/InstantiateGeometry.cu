#include "ImageCudaDevice.cuh"
#include "ImageCudaKernel.cuh"
#include "PointCloudCudaKernel.cuh"
#include "TriangleMeshCudaKernel.cuh"
#include "NNCudaKernel.cuh"
#include "RGBDImageCudaKernel.cuh"

#include "LinearAlgebraCuda.h"
#include <src/Cuda/Common/Common.h>

namespace open3d {

namespace cuda {
template
class ImageCudaDevice<Vector1s>;
template
class ImageCudaDevice<Vector4b>;
template
class ImageCudaDevice<Vector3b>;
template
class ImageCudaDevice<Vector1b>;
template
class ImageCudaDevice<Vector4f>;
template
class ImageCudaDevice<Vector3f>;
template
class ImageCudaDevice<Vector1f>;

template
class ImageCudaKernelCaller<Vector1s>;
template
class ImageCudaKernelCaller<Vector4b>;
template
class ImageCudaKernelCaller<Vector3b>;
template
class ImageCudaKernelCaller<Vector1b>;
template
class ImageCudaKernelCaller<Vector4f>;
template
class ImageCudaKernelCaller<Vector3f>;
template
class ImageCudaKernelCaller<Vector1f>;

/** Vector **/
template Vector4f operator*<float, 4>(float s, const Vector4f &vec);
template Vector3f operator*<float, 3>(float s, const Vector3f &vec);
template Vector1f operator*<float, 1>(float s, const Vector1f &vec);
} // cuda
} // open3d