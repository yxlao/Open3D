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

#include "open3d_pybind/core/container.h"
#include "open3d_pybind/docstring.h"
#include "open3d_pybind/open3d_pybind.h"

#include "Open3D/Core/TensorKey.h"

using namespace open3d;

// Trempoline classes for pybind11.
template <class TensorKeyBase = TensorKey>
class PyTensorKey : public TensorKeyBase {
public:
    using TensorKeyBase::TensorKeyBase;
};

template <class TensorIndexBase = TensorIndex>
class PyTensorIndex : public TensorIndex {
public:
    using TensorIndexBase::TensorIndexBase;
};

template <class TensorSliceBase = TensorSlice>
class PyTensorSlice : public TensorSlice {
public:
    using TensorSliceBase::TensorSliceBase;
};

void pybind_core_tensor_key(py::module &m) {
    py::class_<TensorKey, PyTensorKey<>, std::shared_ptr<TensorKey>> tensor_key(
            m, "TensorKey");

    py::class_<TensorIndex, PyTensorIndex<>, TensorKey,
               std::shared_ptr<TensorIndex>>
            tensor_index(m, "TensorIndex");
    tensor_index.def(
            py::init([](int64_t index) { return new TensorIndex(index); }));

    py::class_<TensorSlice, PyTensorSlice<>, TensorKey,
               std::shared_ptr<TensorSlice>>
            tensor_slice(m, "TensorSlice");
    tensor_slice.def(py::init([](int64_t start, int64_t stop, int64_t end) {
        return new TensorSlice(start, stop, end);
    }));

    tensor_slice.def_static("all", &TensorSlice::All);
}
