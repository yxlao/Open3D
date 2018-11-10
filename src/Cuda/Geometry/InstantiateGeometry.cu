#include "ImageCudaDevice.cuh"
#include "ImageCudaKernel.cuh"

#include "TriangleMeshCuda.cuh"
#include "TriangleMeshCudaKernel.cuh"
#include "VectorCuda.h"
#include <Cuda/Common/Common.h>

namespace open3d {

template class ImageCudaServer<Vector1s>;
template class ImageCudaServer<Vector4b>;
template class ImageCudaServer<Vector3b>;
template class ImageCudaServer<Vector1b>;
template class ImageCudaServer<Vector4f>;
template class ImageCudaServer<Vector3f>;
template class ImageCudaServer<Vector1f>;

template class ImageCudaKernelCaller<Vector1s>;
template class ImageCudaKernelCaller<Vector4b>;
template class ImageCudaKernelCaller<Vector3b>;
template class ImageCudaKernelCaller<Vector1b>;
template class ImageCudaKernelCaller<Vector4f>;
template class ImageCudaKernelCaller<Vector3f>;
template class ImageCudaKernelCaller<Vector1f>;

/** Vector **/
template Vector4f operator*<float, 4>(float s, const Vector4f &vec);
template Vector3f operator*<float, 3>(float s, const Vector3f &vec);
template Vector1f operator*<float, 1>(float s, const Vector1f &vec);
}