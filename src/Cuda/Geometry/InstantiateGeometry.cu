#include "ImageCudaDevice.cuh"
#include "ImageCudaKernel.cuh"
#include "PointCloudCudaKernel.cuh"
#include "TriangleMeshCudaKernel.cuh"
#include "NNCudaKernel.cuh"
#include "RGBDImageCudaKernel.cuh"

#include <Cuda/Common/LinearAlgebraCuda.h>
#include <Cuda/Common/Common.h>

namespace open3d {

namespace cuda {
template
class ImageCudaDevice<ushort, 1>;
template
class ImageCudaDevice<uchar, 4>;
template
class ImageCudaDevice<uchar, 3>;
template
class ImageCudaDevice<uchar, 1>;
template
class ImageCudaDevice<float, 4>;
template
class ImageCudaDevice<float, 3>;
template
class ImageCudaDevice<float, 1>;

template
class ImageCudaKernelCaller<ushort, 1>;
template
class ImageCudaKernelCaller<uchar, 4>;
template
class ImageCudaKernelCaller<uchar, 3>;
template
class ImageCudaKernelCaller<uchar, 1>;
template
class ImageCudaKernelCaller<float, 4>;
template
class ImageCudaKernelCaller<float, 3>;
template
class ImageCudaKernelCaller<float, 1>;

/** Vector **/
template Vector4f operator*<float, 4>(float s, const Vector4f &vec);
template Vector3f operator*<float, 3>(float s, const Vector3f &vec);
template Vector1f operator*<float, 1>(float s, const Vector1f &vec);
} // cuda
} // open3d