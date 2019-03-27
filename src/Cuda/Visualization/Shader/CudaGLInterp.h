//
// Created by wei on 11/5/18.
//

#pragma once

#include <Cuda/Common/UtilsCuda.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <cassert>

namespace open3d {
namespace {
/**
 * Replacement of BindData - direct transfer data to graphics part on GPU
 */
template<typename T>
bool CopyDataToCudaGraphicsResource(GLuint &opengl_buffer,
                                    cudaGraphicsResource_t &cuda_graphics_resource,
                                    T *cuda_vector, size_t cuda_vector_size) {

    CheckCuda(cudaGraphicsMapResources(1, &cuda_graphics_resource));

    /* Copy memory on-chip */
    void *mapped_ptr;
    size_t mapped_size;
    CheckCuda(cudaGraphicsResourceGetMappedPointer(&mapped_ptr, &mapped_size,
                                                   cuda_graphics_resource));

    const size_t byte_count = cuda_vector_size * sizeof(T);
    assert(byte_count <= mapped_size);
    CheckCuda(cudaMemcpy(mapped_ptr, cuda_vector,
                         byte_count,
                         cudaMemcpyDeviceToDevice));

    CheckCuda(cudaGraphicsUnmapResources(1, &cuda_graphics_resource, nullptr));

    return true;
}

template<typename T>
bool RegisterResource(cudaGraphicsResource_t &cuda_graphics_resource,
                      GLenum opengl_buffer_type,
                      GLuint &opengl_buffer,
                      T *cuda_vector,
                      size_t cuda_vector_size) {
    glGenBuffers(1, &opengl_buffer);
    glBindBuffer(opengl_buffer_type, opengl_buffer);
    glBufferData(opengl_buffer_type, sizeof(T) * cuda_vector_size,
                 nullptr, GL_STATIC_DRAW);
    CheckCuda(cudaGraphicsGLRegisterBuffer(&cuda_graphics_resource,
                                           opengl_buffer,
                                           cudaGraphicsMapFlagsReadOnly));

    CopyDataToCudaGraphicsResource(opengl_buffer, cuda_graphics_resource,
                                   cuda_vector, cuda_vector_size);
    return true;
}

bool UnregisterResource(cudaGraphicsResource_t &cuda_graphics_resource,
                        GLuint &opengl_buffer) {

    glDeleteBuffers(1, &opengl_buffer);
    CheckCuda(cudaGraphicsUnregisterResource(cuda_graphics_resource));

    return true;
}
}
}