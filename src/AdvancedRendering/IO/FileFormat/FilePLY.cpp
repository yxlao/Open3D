//
// Created by wei on 4/21/19.
//

// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <rply/rply.h>
#include <Open3D/Utility/Console.h>
#include <AdvancedRendering/Geometry/ExtendedTriangleMesh.h>
#include <AdvancedRendering/IO/ClassIO/ExtendedTriangleMeshIO.h>

namespace open3d {

namespace {
using namespace io;

namespace ply_trianglemesh_extended_reader {

struct PLYReaderState {
    geometry::ExtendedTriangleMesh *mesh_ptr;
    long vertex_index;
    long vertex_num;
    long normal_index;
    long normal_num;
    long color_index;
    long color_num;

    long uv_index;
    long uv_num;
    long material_index;
    long material_num;

    long triangle_index;
    long triangle_num;
};

int ReadVertexCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &index);
    if (state_ptr->vertex_index >= state_ptr->vertex_num) {
        return 0;
    }

    double value = ply_get_argument_value(argument);
    state_ptr->mesh_ptr->vertices_[state_ptr->vertex_index](index) = value;
    if (index == 2) {  // reading 'z'
        state_ptr->vertex_index++;
        utility::AdvanceConsoleProgress();
    }
    return 1;
}

int ReadNormalCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &index);
    if (state_ptr->normal_index >= state_ptr->normal_num) {
        return 0;
    }

    double value = ply_get_argument_value(argument);
    state_ptr->mesh_ptr->vertex_normals_[state_ptr->normal_index](index) =
        value;
    if (index == 2) {  // reading 'nz'
        state_ptr->normal_index++;
    }
    return 1;
}

int ReadColorCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &index);
    if (state_ptr->color_index >= state_ptr->color_num) {
        return 0;
    }

    double value = ply_get_argument_value(argument);
    state_ptr->mesh_ptr->vertex_colors_[state_ptr->color_index](index) =
        value / 255.0;
    if (index == 2) {  // reading 'blue'
        state_ptr->color_index++;
    }
    return 1;
}

int ReadUVCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &index);
    if (state_ptr->uv_index >= state_ptr->uv_num) {
        return 0;
    }

    double value = ply_get_argument_value(argument);
    state_ptr->mesh_ptr->vertex_uvs_[state_ptr->uv_index](index) = value;
    if (index == 1) {  // reading 'v'
        state_ptr->uv_index++;
    }
    return 1;
}

int ReadMaterialCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &index);
    if (state_ptr->material_index >= state_ptr->material_num) {
        return 0;
    }

    double value = ply_get_argument_value(argument);
    state_ptr->mesh_ptr->vertex_textures_[state_ptr->material_index](index) =
        value;
    if (index == 2) {  // reading 'ao'
        state_ptr->material_index++;
    }
    return 1;
}

int ReadFaceCallBack(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long dummy, length, index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &dummy);
    double value = ply_get_argument_value(argument);
    if (state_ptr->triangle_index >= state_ptr->triangle_num) {
        return 0;
    }

    ply_get_argument_property(argument, NULL, &length, &index);
    if ((index >= 0) && (index <= 2)) {
        state_ptr->mesh_ptr->triangles_[state_ptr->triangle_index](index) =
            static_cast<int>(value);
    }
    if (index == 2) {  // reading 'triangles_[n](2)'
        state_ptr->triangle_index++;
        utility::AdvanceConsoleProgress();
    }
    return 1;
}

}  // namespace ply_trianglemesh_reader
} // unnamed space

namespace io {

bool ReadExtendedTriangleMeshFromPLY(const std::string &filename,
                                     geometry::ExtendedTriangleMesh &mesh) {
    using namespace ply_trianglemesh_extended_reader;

    p_ply ply_file = ply_open(filename.c_str(), NULL, 0, NULL);
    if (!ply_file) {
        utility::PrintWarning("Read PLY failed: unable to open file: %s\n",
                              filename.c_str());
        return false;
    }
    if (!ply_read_header(ply_file)) {
        utility::PrintWarning("Read PLY failed: unable to parse header.\n");
        ply_close(ply_file);
        return false;
    }

    PLYReaderState state;
    state.mesh_ptr = &mesh;
    state.vertex_num = ply_set_read_cb(ply_file, "vertex", "x",
                                       ReadVertexCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "y", ReadVertexCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "z", ReadVertexCallback, &state, 2);

    state.normal_num = ply_set_read_cb(ply_file, "vertex", "nx",
                                       ReadNormalCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "ny", ReadNormalCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "nz", ReadNormalCallback, &state, 2);

    state.color_num = ply_set_read_cb(ply_file, "vertex", "red",
                                      ReadColorCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "green", ReadColorCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "blue", ReadColorCallback, &state, 2);

    state.uv_num = ply_set_read_cb(ply_file, "vertex", "u",
        ReadUVCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "v", ReadUVCallback, &state, 1);

    state.material_num = ply_set_read_cb(ply_file, "vertex", "roughness",
                                       ReadMaterialCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "metallic", ReadMaterialCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "ao", ReadMaterialCallback, &state, 2);

    if (state.vertex_num <= 0) {
        utility::PrintWarning("Read PLY failed: number of vertex <= 0.\n");
        ply_close(ply_file);
        return false;
    }

    state.triangle_num = ply_set_read_cb(ply_file, "face", "vertex_indices",
                                         ReadFaceCallBack, &state, 0);
    if (state.triangle_num == 0) {
        state.triangle_num = ply_set_read_cb(ply_file, "face", "vertex_index",
                                             ReadFaceCallBack, &state, 0);
    }

    state.vertex_index = 0;
    state.normal_index = 0;
    state.color_index = 0;
    state.uv_index = 0;
    state.material_index = 0;
    state.triangle_index = 0;

    mesh.Clear();
    mesh.vertices_.resize(state.vertex_num);
    mesh.vertex_normals_.resize(state.normal_num);
    mesh.vertex_colors_.resize(state.color_num);
    mesh.vertex_uvs_.resize(state.uv_num);
    mesh.vertex_textures_.resize(state.material_num);
    mesh.triangles_.resize(state.triangle_num);

    utility::ResetConsoleProgress(state.vertex_num + state.triangle_num,
                                  "Reading PLY: ");

    if (!ply_read(ply_file)) {
        utility::PrintWarning("Read PLY failed: unable to read file: %s\n",
                              filename.c_str());
        ply_close(ply_file);
        return false;
    }

    ply_close(ply_file);
    return true;
}

bool WriteExtendedTriangleMeshToPLY(const std::string &filename,
                                    const geometry::ExtendedTriangleMesh &mesh,
                                    bool write_ascii /* = false*/,
                                    bool compressed /* = false*/) {
    if (mesh.IsEmpty()) {
        utility::PrintWarning("Write PLY failed: mesh has 0 vertices.\n");
        return false;
    }

    p_ply ply_file = ply_create(filename.c_str(),
                                write_ascii ? PLY_ASCII : PLY_LITTLE_ENDIAN,
                                NULL, 0, NULL);
    if (!ply_file) {
        utility::PrintWarning("Write PLY failed: unable to open file: %s\n",
                              filename.c_str());
        return false;
    }
    ply_add_comment(ply_file, "Created by Open3D");
    ply_add_element(ply_file, "vertex",
                    static_cast<long>(mesh.vertices_.size()));
    ply_add_property(ply_file, "x", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    ply_add_property(ply_file, "y", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    ply_add_property(ply_file, "z", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    if (mesh.HasVertexNormals()) {
        ply_add_property(ply_file, "nx", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
        ply_add_property(ply_file, "ny", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
        ply_add_property(ply_file, "nz", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    }
    if (mesh.HasVertexColors()) {
        ply_add_property(ply_file, "red", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
        ply_add_property(ply_file, "green", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
        ply_add_property(ply_file, "blue", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
    }
    if (mesh.HasUVs()) {
        ply_add_property(ply_file, "u", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
        ply_add_property(ply_file, "v", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    }
    if (mesh.HasVertexTextures()) {
        ply_add_property(ply_file, "roughness", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
        ply_add_property(ply_file, "metallic", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
        ply_add_property(ply_file, "ao", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    }

    ply_add_element(ply_file, "face",
                    static_cast<long>(mesh.triangles_.size()));
    ply_add_property(ply_file, "vertex_indices", PLY_LIST, PLY_UCHAR, PLY_UINT);
    if (!ply_write_header(ply_file)) {
        utility::PrintWarning("Write PLY failed: unable to write header.\n");
        ply_close(ply_file);
        return false;
    }

    utility::ResetConsoleProgress(
        static_cast<int>(mesh.vertices_.size() + mesh.triangles_.size()),
        "Writing PLY: ");
    for (size_t i = 0; i < mesh.vertices_.size(); i++) {
        const auto &vertex = mesh.vertices_[i];
        ply_write(ply_file, vertex(0));
        ply_write(ply_file, vertex(1));
        ply_write(ply_file, vertex(2));
        if (mesh.HasVertexNormals()) {
            const auto &normal = mesh.vertex_normals_[i];
            ply_write(ply_file, normal(0));
            ply_write(ply_file, normal(1));
            ply_write(ply_file, normal(2));
        }
        if (mesh.HasVertexColors()) {
            const auto &color = mesh.vertex_colors_[i];
            ply_write(ply_file, color(0) * 255.0);
            ply_write(ply_file, color(1) * 255.0);
            ply_write(ply_file, color(2) * 255.0);
        }
        if (mesh.HasUVs()) {
            const auto &uv = mesh.vertex_uvs_[i];
            ply_write(ply_file, uv(0));
            ply_write(ply_file, uv(1));
        }
        if (mesh.HasVertexTextures()) {
            const auto &material = mesh.vertex_textures_[i];
            ply_write(ply_file, material(0));
            ply_write(ply_file, material(1));
            ply_write(ply_file, material(2));
        }
        utility::AdvanceConsoleProgress();
    }
    for (size_t i = 0; i < mesh.triangles_.size(); i++) {
        const auto &triangle = mesh.triangles_[i];
        ply_write(ply_file, 3);
        ply_write(ply_file, triangle(0));
        ply_write(ply_file, triangle(1));
        ply_write(ply_file, triangle(2));
        utility::AdvanceConsoleProgress();
    }

    ply_close(ply_file);
    return true;
}
}  // namespace io
}  // namespace open3d
