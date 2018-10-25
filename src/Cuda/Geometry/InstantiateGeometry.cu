#include "ImageCuda.cuh"
#include "ImagePyramidCuda.cuh"
#include "ImageCudaKernel.cuh"
#include "TriangleMeshCuda.cuh"
#include "VectorCuda.h"
#include <Cuda/Common/Common.h>

namespace open3d {

template
class TriangleMeshCuda<VertexRaw>;

template
class TriangleMeshCuda<VertexWithNormal>;

template
class TriangleMeshCuda<VertexWithColor>;

template
class TriangleMeshCuda<VertexWithNormalAndColor>;

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
}