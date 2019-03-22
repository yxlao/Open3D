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

#include "Python/visualization/visualization.h"
#include "Python/visualization/visualization_trampoline.h"
#include "Python/docstring.h"

#include <Open3D/Visualization/Visualizer/ViewControl.h>
#include <Open3D/IO/ClassIO/IJsonConvertibleIO.h>
using namespace open3d;

void pybind_viewcontrol(py::module &m) {
    py::class_<visualization::ViewControl, PyViewControl<>,
               std::shared_ptr<visualization::ViewControl>>
            viewcontrol(m, "ViewControl");
    py::detail::bind_default_constructor<visualization::ViewControl>(
            viewcontrol);
    viewcontrol
            .def("__repr__",
                 [](const visualization::ViewControl &vc) {
                     return std::string("ViewControl");
                 })
            .def("convert_to_pinhole_camera_parameters",
                 [](visualization::ViewControl &vc) {
                     camera::PinholeCameraParameters parameter;
                     vc.ConvertToPinholeCameraParameters(parameter);
                     return parameter;
                 },
                 "Function to convert visualization::ViewControl to "
                 "camera::PinholeCameraParameters")
            .def("convert_from_pinhole_camera_parameters",
                 &visualization::ViewControl::
                         ConvertFromPinholeCameraParameters,
                 "parameter"_a)
            .def("scale", &visualization::ViewControl::Scale,
                 "Function to process scaling", "scale"_a)
            .def("rotate", &visualization::ViewControl::Rotate,
                 "Function to process rotation", "x"_a, "y"_a, "xo"_a = 0.0,
                 "yo"_a = 0.0)
            .def("translate", &visualization::ViewControl::Translate,
                 "Function to process translation", "x"_a, "y"_a, "xo"_a = 0.0,
                 "yo"_a = 0.0)
            .def("get_field_of_view",
                 &visualization::ViewControl::GetFieldOfView,
                 "Function to get field of view")
            .def("change_field_of_view",
                 &visualization::ViewControl::ChangeFieldOfView,
                 "Function to change field of view", "step"_a = 0.45);
}

void pybind_viewcontrol_method(py::module &m) {}
