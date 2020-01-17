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

#include <vector>

#include "open3d_pybind/core/container.h"
#include "open3d_pybind/docstring.h"
#include "open3d_pybind/open3d_pybind.h"
#include "open3d_pybind/pybind_utils.h"

#include "Open3D/Core/Blob.h"
#include "Open3D/Core/CUDAUtils.h"
#include "Open3D/Core/Device.h"
#include "Open3D/Core/Dispatch.h"
#include "Open3D/Core/Dtype.h"
#include "Open3D/Core/SizeVector.h"
#include "Open3D/Core/Tensor.h"

using namespace open3d;

template <typename T>
void bind_templated_constructor(py::class_<Tensor>& tensor) {
    tensor.def(py::init([](const std::vector<T>& init_vals,
                           const SizeVector& shape, const Dtype& dtype,
                           const Device& device = Device("CPU:0")) {
                   return new Tensor(init_vals, shape, dtype, device);
               }),
               "init_vals"_a, "shape"_a, "dtype"_a, "device"_a);
}

/// Automatically casts type to T.
template <typename T>
static std::vector<T> ToFlatVector(
        py::array_t<T, py::array::c_style | py::array::forcecast> np_array) {
    py::buffer_info info = np_array.request();
    T* start = static_cast<T*>(info.ptr);
    return std::vector<T>(start, start + info.size);
}

void pybind_core_tensor(py::module& m) {
    py::class_<Tensor> tensor(
            m, "Tensor",
            "A Tensor is a view of a data Blob with shape, stride, data_ptr.");

    tensor.def(py::init([](py::array np_array, const Dtype& dtype,
                           const Device& device) {
        py::buffer_info info = np_array.request();
        SizeVector shape(info.shape.begin(), info.shape.end());
        Tensor t;
        DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
            t = Tensor(ToFlatVector<scalar_t>(np_array), shape, dtype, device);
        });
        return t;
    }));

    // Device transfer
    tensor.def("cuda",
               [](const Tensor& tensor, int64_t device_id = 0) {
                   if (!cuda::IsAvailable()) {
                       utility::LogError(
                               "CUDA is not available, cannot copy Tensor.");
                   }
                   if (device_id < 0 || device_id >= cuda::DeviceCount()) {
                       utility::LogError(
                               "Invalid device_id {}, must satisfy 0 <= "
                               "device_id < {}",
                               device_id, cuda::DeviceCount());
                   }
                   return tensor.Copy(
                           Device(Device::DeviceType::CUDA, device_id));
               })
            .def("cpu", [](const Tensor& tensor) {
                return tensor.Copy(Device(Device::DeviceType::CPU, 0));
            });

    // Buffer I/O for Numpy and DLPack(PyTorch)
    tensor.def("numpy",
               [](const Tensor& tensor) {
                   if (tensor.GetDevice().GetType() !=
                       Device::DeviceType::CPU) {
                       utility::LogError(
                               "Can only convert CPU Tensor to numpy. Copy "
                               "Tensor to CPU before converting to numpy.");
                   }
                   py::dtype py_dtype = py::dtype(
                           pybind_utils::DtypeToArrayFormat(tensor.GetDtype()));
                   py::array::ShapeContainer py_shape(tensor.GetShape());
                   SizeVector strides = tensor.GetStrides();
                   int64_t element_byte_size =
                           DtypeUtil::ByteSize(tensor.GetDtype());
                   for (auto& s : strides) {
                       s *= element_byte_size;
                   }
                   py::array::StridesContainer py_strides(strides);

                   // `base_tensor` is a shallow copy of `tensor`. `base_tensor`
                   // is on the heap and is owned by py::capsule
                   // `base_tensor_capsule`. The capsule is referenced as the
                   // "base" of the numpy tensor returned by o3d.Tensor.numpy().
                   // When the "base" goes out-of-scope (e.g. when all numpy
                   // tensors referencing the base have gone out-of-scope), the
                   // deleter is called to free the `base_tensor`.
                   //
                   // This behavior is important when the origianl `tensor` goes
                   // out-of-scope while we still want to keep the data alive.
                   // e.g.
                   //
                   // ```python
                   // def get_np_tensor():
                   //     o3d_t = o3d.Tensor(...)
                   //     return o3d_t.numpy()
                   //
                   // # Now, `o3d_t` is out-of-scope, but `np_t` still
                   // # references the base tensor which references the
                   // # underlying data of `o3d_t`. Thus np_t is still valid.
                   // # When np_t goes out-of-scope, the underlying data will be
                   // # finally freed.
                   // np_t = get_np_tensor()
                   // ```
                   //
                   // See:
                   // https://stackoverflow.com/questions/44659924/returning-numpy-arrays-via-pybind11
                   Tensor* base_tensor = new Tensor(tensor);

                   // See PyTorch's torch/csrc/Module.cpp
                   auto capsule_destructor = [](PyObject* data) {
                       Tensor* base_tensor = reinterpret_cast<Tensor*>(
                               PyCapsule_GetPointer(data, "open3d::Tensor"));
                       if (base_tensor) {
                           delete base_tensor;
                       } else {
                           PyErr_Clear();
                       }
                   };

                   py::capsule base_tensor_capsule(
                           base_tensor, "open3d::Tensor", capsule_destructor);

                   return py::array(py_dtype, py_shape, py_strides,
                                    tensor.GetDataPtr(), base_tensor_capsule);
               })
            .def_static(
                    "from_numpy",
                    [](py::array np_array) {
                        py::buffer_info info = np_array.request();

                        SizeVector shape(info.shape.begin(), info.shape.end());
                        SizeVector strides(info.strides.begin(),
                                           info.strides.end());
                        for (size_t i = 0; i < strides.size(); ++i) {
                            strides[i] /= info.itemsize;
                        }
                        Dtype dtype =
                                pybind_utils::ArrayFormatToDtype(info.format);
                        Device device("CPU:0");

                        // Blob expects an std::function<void(void*)> deleter, a
                        // dummy deleter is used here, since the memory is
                        // managed by numpy.
                        std::function<void(void*)> deleter = [](void*) -> void {
                        };
                        auto blob = std::make_shared<Blob>(device, info.ptr,
                                                           deleter);

                        return Tensor(shape, strides, info.ptr, dtype, device,
                                      blob);
                    })
            .def("to_dlpack",
                 [](const Tensor& tensor) {
                     DLManagedTensor* dl_managed_tensor = tensor.ToDLPack();
                     // See PyTorch's torch/csrc/Module.cpp
                     auto capsule_destructor = [](PyObject* data) {
                         DLManagedTensor* dl_managed_tensor =
                                 (DLManagedTensor*)PyCapsule_GetPointer(
                                         data, "dltensor");
                         if (dl_managed_tensor) {
                             // the dl_managed_tensor has not been consumed,
                             // call deleter ourselves
                             dl_managed_tensor->deleter(
                                     const_cast<DLManagedTensor*>(
                                             dl_managed_tensor));
                         } else {
                             // The dl_managed_tensor has been consumed
                             // PyCapsule_GetPointer has set an error indicator
                             PyErr_Clear();
                         }
                     };
                     return py::capsule(dl_managed_tensor, "dltensor",
                                        capsule_destructor);
                 })
            .def_static("from_dlpack",
                        [](py::capsule data) {
                            DLManagedTensor* dl_managed_tensor =
                                    static_cast<DLManagedTensor*>(data);
                            if (!dl_managed_tensor) {
                                utility::LogError(
                                        "from_dlpack must receive "
                                        "DLManagedTensor PyCapsule.");
                            }
                            // Make sure that the PyCapsule is not used again.
                            // See:
                            // torch/csrc/Module.cpp, and
                            // https://github.com/cupy/cupy/pull/1445/files#diff-ddf01ff512087ef616db57ecab88c6ae
                            Tensor t = Tensor::FromDLPack(dl_managed_tensor);
                            PyCapsule_SetName(data.ptr(), "used_dltensor");
                            return t;
                        })
            .def("__repr__",
                 [](const Tensor& tensor) { return tensor.ToString(); });
}
