#include "ImageCuda.cuh"
#include "Vector.h"
#include <Cuda/Common/Common.h>

namespace three {

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

}