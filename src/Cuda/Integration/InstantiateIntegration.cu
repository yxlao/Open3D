//
// Created by wei on 10/10/18.
//

#include "UniformTSDFVolumeCuda.cuh"
#include "UniformTSDFVolumeCudaKernel.cuh"

namespace open3d {
template
class UniformTSDFVolumeCudaServer<8>;

template
class UniformTSDFVolumeCudaServer<16>;

template
class UniformTSDFVolumeCudaServer<256>;

template
class UniformTSDFVolumeCudaServer<512>;

template
class UniformTSDFVolumeCuda<8>;

template
class UniformTSDFVolumeCuda<16>;

template
class UniformTSDFVolumeCuda<256>;

template
class UniformTSDFVolumeCuda<512>;

template
__global__
void IntegrateKernel<8>(UniformTSDFVolumeCudaServer<8> server,
                        ImageCudaServer<Vector1f> depth,
                        MonoPinholeCameraCuda camera,
                        TransformCuda transform_camera_to_world);

template
__global__
void IntegrateKernel<16>(UniformTSDFVolumeCudaServer<16> server,
                         ImageCudaServer<Vector1f> depth,
                         MonoPinholeCameraCuda camera,
                         TransformCuda transform_camera_to_world);
template
__global__
void IntegrateKernel<256>(UniformTSDFVolumeCudaServer<256> server,
                          ImageCudaServer<Vector1f> depth,
                          MonoPinholeCameraCuda camera,
                          TransformCuda transform_camera_to_world);
template
__global__
void IntegrateKernel<512>(UniformTSDFVolumeCudaServer<512> server,
                          ImageCudaServer<Vector1f> depth,
                          MonoPinholeCameraCuda camera,
                          TransformCuda transform_camera_to_world);

template
__global__
void RayCastingKernel<8>(UniformTSDFVolumeCudaServer<8> server,
                         ImageCudaServer<Vector3f> image,
                         MonoPinholeCameraCuda camera,
                         TransformCuda transform_camera_to_world);
template
__global__
void RayCastingKernel<16>(UniformTSDFVolumeCudaServer<16> server,
                          ImageCudaServer<Vector3f> image,
                          MonoPinholeCameraCuda camera,
                          TransformCuda transform_camera_to_world);
template
__global__
void RayCastingKernel<256>(UniformTSDFVolumeCudaServer<256> server,
                           ImageCudaServer<Vector3f> image,
                           MonoPinholeCameraCuda camera,
                           TransformCuda transform_camera_to_world);

template
__global__
void RayCastingKernel<512>(UniformTSDFVolumeCudaServer<512> server,
                           ImageCudaServer<Vector3f> image,
                           MonoPinholeCameraCuda camera,
                           TransformCuda transform_camera_to_world);

}


