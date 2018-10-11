//
// Created by wei on 10/10/18.
//

#pragma once

#include "UniformTSDFVolumeCuda.cuh"
#include <Geometry/ImageCuda.cuh>

namespace open3d {
template<size_t N>
__global__
void IntegrateKernel(UniformTSDFVolumeCudaServer<N> server,
                     ImageCudaServer<Vector1f> depth,
                     MonoPinholeCameraCuda camera,
                     TransformCuda transform_camera_to_world) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= N || y >= N || z >= N) return;

    /** Projective data association **/
    Vector3f X_w = server.volume_to_world(x, y, z);
    Vector3f X_c = transform_camera_to_world.Inverse() * X_w;
    Vector2f p_c = camera.Projection(X_c);

    /** TSDF **/
    if (!camera.IsValid(p_c)) return;
    float d = depth.get_interp(p_c(0), p_c(1))(0);

    float sdf = d - X_c(2);
    if (sdf <= -server.sdf_trunc_) return;
    sdf = fminf(sdf, server.sdf_trunc_);

    /** Weight average **/
    float &tsdf = server.tsdf(x, y, z);
    float &weight = server.weight(x, y, z);

    /** TODO: color **/
    tsdf = (tsdf * weight + sdf * 1.0f) / (weight + 1.0f);
    weight += 1.0f;
}

template<size_t N>
__global__
void RayCastingKernel(UniformTSDFVolumeCudaServer<N> server,
                      ImageCudaServer<Vector3f> image,
                      MonoPinholeCameraCuda camera,
                      TransformCuda transform_camera_to_world) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= image.width_ || y >= image.height_) return;
    image.get(x, y) = Vector3f(0);

    Vector3f ray_c = camera.InverseProjection(x, y, 1.0f).normalized();

    /** TODO: throw it into parameters **/
    const float t_min = 0.1f / ray_c(2);
    const float t_max = 3.0f / ray_c(2);

    const Vector3f camera_origin_w = transform_camera_to_world * Vector3f(0);
    const Vector3f ray_w = transform_camera_to_world.Rotate(ray_c);

    float t_prev = 0, tsdf_prev = 0, weight_prev = 0;

    /** Do NOT use #pragma unroll: it will make it slow **/
    for (float t = t_min; t < t_max; t += server.voxel_length_) {
        Vector3f X_w = camera_origin_w + t * ray_w;
        Vector3f X_v = server.world_to_volume(X_w);

        if (!server.InVolumef(X_v)) continue;

        float tsdf_curr = server.TSDFAt(X_v);
        float weight_curr = server.WeightAt(X_v);

        if (weight_curr > 0 && weight_prev > 0
            && tsdf_prev > 0 && tsdf_curr < 0) { /* Intersection */

            float t_intersect =
                (t * tsdf_prev - t_prev * tsdf_curr) / (tsdf_prev - tsdf_curr);

            Vector3f X_surface_v = server.world_to_volume(camera_origin_w + t_intersect * ray_w);
//            printf("(%f, %f), (%f, %f) -> (%f, %f)\n", t_prev, tsdf_prev, t,
//                   tsdf_curr, t_intersect, server.TSDFAt(X_surface_v));
            Vector3f normal = server.GradientAt(X_surface_v).normalized();
//            printf("%f %f %f -> %f %f %f\n", t_intersect, tsdf_prev,
//                tsdf_curr, X_surface_v(0), X_surface_v(1), X_surface_v(2));

            image.get(x, y) = normal;
            return;
        }

        tsdf_prev = tsdf_curr;
        weight_prev = weight_curr;
        t_prev = t;
    }
}
}