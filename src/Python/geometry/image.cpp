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

#include "Open3D/Geometry/Image.h"
#include "Open3D/Geometry/RGBDImage.h"
#include "Python/docstring.h"
#include "Python/geometry/geometry.h"
#include "Python/geometry/geometry_trampoline.h"

using namespace open3d;

// Image functions have similar arguments, thus the arg docstrings may be shared
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"color", "The color image."},
                {"convert_rgb_to_intensity",
                 "Whether to convert RGB image to intensity image."},
                {"depth", "The depth image."},
                {"depth_scale",
                 "The ratio to scale depth values. The depth values will first "
                 "be scaled and then truncated."},
                {"depth_trunc",
                 "Depth values larger than ``depth_trunc`` gets truncated to "
                 "0. The depth values will first be scaled and then "
                 "truncated."},
                {"filter_type", "The filter type to be applied."},
                {"image", "The Image object."},
                {"image_pyramid", "The ImagePyramid object"},
                {"num_of_levels ", "Levels of the image pyramid"},
                {"with_gaussian_filter",
                 "When ``True``, image in the pyramid will first be filtered "
                 "by a 3x3 Gaussian kernel before downsampling."}};

void pybind_image(py::module &m) {
    py::class_<geometry::Image, PyGeometry2D<geometry::Image>,
               std::shared_ptr<geometry::Image>, geometry::Geometry2D>
            image(m, "Image", py::buffer_protocol(),
                  "The image class stores image with customizable width, "
                  "height, num of channels and bytes per channel.");
    py::detail::bind_default_constructor<geometry::Image>(image);
    py::detail::bind_copy_functions<geometry::Image>(image);
    image.def(py::init([](py::buffer b) {
             py::buffer_info info = b.request();
             int width, height, num_of_channels = 0, bytes_per_channel;
             if (info.format == py::format_descriptor<uint8_t>::format() ||
                 info.format == py::format_descriptor<int8_t>::format()) {
                 bytes_per_channel = 1;
             } else if (info.format ==
                                py::format_descriptor<uint16_t>::format() ||
                        info.format ==
                                py::format_descriptor<int16_t>::format()) {
                 bytes_per_channel = 2;
             } else if (info.format == py::format_descriptor<float>::format()) {
                 bytes_per_channel = 4;
             } else {
                 throw std::runtime_error(
                         "Image can only be initialized from buffer of uint8, "
                         "uint16, or float!");
             }
             if (info.strides[info.ndim - 1] != bytes_per_channel) {
                 throw std::runtime_error(
                         "Image can only be initialized from c-style buffer.");
             }
             if (info.ndim == 2) {
                 num_of_channels = 1;
             } else if (info.ndim == 3) {
                 num_of_channels = (int)info.shape[2];
             }
             height = (int)info.shape[0];
             width = (int)info.shape[1];
             auto img = new geometry::Image();
             img->PrepareImage(width, height, num_of_channels,
                               bytes_per_channel);
             memcpy(img->data_.data(), info.ptr, img->data_.size());
             return img;
         }))
            .def_buffer([](geometry::Image &img) -> py::buffer_info {
                std::string format;
                switch (img.bytes_per_channel_) {
                    case 1:
                        format = py::format_descriptor<uint8_t>::format();
                        break;
                    case 2:
                        format = py::format_descriptor<uint16_t>::format();
                        break;
                    case 4:
                        format = py::format_descriptor<float>::format();
                        break;
                    default:
                        throw std::runtime_error(
                                "Image has unrecognized bytes_per_channel.");
                        break;
                }
                if (img.num_of_channels_ == 1) {
                    return py::buffer_info(
                            img.data_.data(), img.bytes_per_channel_, format, 2,
                            {static_cast<unsigned long>(img.height_),
                             static_cast<unsigned long>(img.width_)},
                            {static_cast<unsigned long>(img.bytes_per_channel_ *
                                                        img.num_of_channels_ *
                                                        img.width_),
                             static_cast<unsigned long>(img.bytes_per_channel_ *
                                                        img.num_of_channels_)});
                } else {
                    return py::buffer_info(
                            img.data_.data(), img.bytes_per_channel_, format, 3,
                            {static_cast<unsigned long>(img.height_),
                             static_cast<unsigned long>(img.width_),
                             static_cast<unsigned long>(img.num_of_channels_)},
                            {static_cast<unsigned long>(img.bytes_per_channel_ *
                                                        img.num_of_channels_ *
                                                        img.width_),
                             static_cast<unsigned long>(img.bytes_per_channel_ *
                                                        img.num_of_channels_),
                             static_cast<unsigned long>(
                                     img.bytes_per_channel_)});
                }
            })
            .def("__repr__", [](const geometry::Image &img) {
                return std::string("Image of size ") +
                       std::to_string(img.width_) + std::string("x") +
                       std::to_string(img.height_) + ", with " +
                       std::to_string(img.num_of_channels_) +
                       std::string(
                               " channels.\nUse numpy.asarray to access buffer "
                               "data.");
            });

    py::class_<geometry::RGBDImage, std::shared_ptr<geometry::RGBDImage>>
            rgbd_image(m, "RGBDImage",
                       "RGBDImage is for a pair of registered color and depth "
                       "images, viewed from the same view, of the same "
                       "resolution. If you have other format, convert it "
                       "first.");
    py::detail::bind_default_constructor<geometry::RGBDImage>(rgbd_image);
    rgbd_image
            .def_readwrite("color", &geometry::RGBDImage::color_,
                           "open3d.geometry.Image: The color image.")
            .def_readwrite("depth", &geometry::RGBDImage::depth_,
                           "open3d.geometry.Image: The depth image.")
            .def("__repr__", [](const geometry::RGBDImage &rgbd_image) {
                return std::string("RGBDImage of size \n") +
                       std::string("Color image : ") +
                       std::to_string(rgbd_image.color_.width_) +
                       std::string("x") +
                       std::to_string(rgbd_image.color_.height_) + ", with " +
                       std::to_string(rgbd_image.color_.num_of_channels_) +
                       std::string(" channels.\n") +
                       std::string("Depth image : ") +
                       std::to_string(rgbd_image.depth_.width_) +
                       std::string("x") +
                       std::to_string(rgbd_image.depth_.height_) + ", with " +
                       std::to_string(rgbd_image.depth_.num_of_channels_) +
                       std::string(" channels.\n") +
                       std::string("Use numpy.asarray to access buffer data.");
            });
}

void pybind_image_methods(py::module &m) {
    py::enum_<geometry::Image::FilterType> image_filter_type(m,
                                                             "ImageFilterType");
    image_filter_type.value("Gaussian3", geometry::Image::FilterType::Gaussian3)
            .value("Gaussian5", geometry::Image::FilterType::Gaussian5)
            .value("Gaussian7", geometry::Image::FilterType::Gaussian7)
            .value("Sobel3dx", geometry::Image::FilterType::Sobel3Dx)
            .value("Sobel3dy", geometry::Image::FilterType::Sobel3Dy)
            .export_values();
    // Trick to write docs without listing the members in the enum class again.
    image_filter_type.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class for Image filter types.";
            }),
            py::none(), py::none(), "");

    m.def("filter_image",
          [](const geometry::Image &input,
             geometry::Image::FilterType filter_type) {
              if (input.num_of_channels_ != 1 ||
                  input.bytes_per_channel_ != 4) {
                  auto input_f = CreateFloatImageFromImage(input);
                  auto output = geometry::FilterImage(*input_f, filter_type);
                  return *output;
              } else {
                  auto output = geometry::FilterImage(input, filter_type);
                  return *output;
              }
          },
          "Function to filter Image", "image"_a, "filter_type"_a);
    docstring::FunctionDocInject(m, "filter_image",
                                 map_shared_argument_docstrings);

    m.def("create_image_pyramid",
          [](const geometry::Image &input, size_t num_of_levels,
             bool with_gaussian_filter) {
              if (input.num_of_channels_ != 1 ||
                  input.bytes_per_channel_ != 4) {
                  auto input_f = CreateFloatImageFromImage(input);
                  auto output = geometry::CreateImagePyramid(
                          *input_f, num_of_levels, with_gaussian_filter);
                  return output;
              } else {
                  auto output = geometry::CreateImagePyramid(
                          input, num_of_levels, with_gaussian_filter);
                  return output;
              }
          },
          "Function to create ImagePyramid", "image"_a, "num_of_levels"_a,
          "with_gaussian_filter"_a);
    docstring::FunctionDocInject(m, "create_image_pyramid",
                                 map_shared_argument_docstrings);

    m.def("filter_image_pyramid",
          [](const geometry::ImagePyramid &input,
             geometry::Image::FilterType filter_type) {
              auto output = geometry::FilterImagePyramid(input, filter_type);
              return output;
          },
          "Function to filter ImagePyramid", "image_pyramid"_a,
          "filter_type"_a);
    docstring::FunctionDocInject(m, "filter_image_pyramid",
                                 map_shared_argument_docstrings);

    m.def("create_rgbd_image_from_color_and_depth",
          &geometry::CreateRGBDImageFromColorAndDepth,
          "Function to make RGBDImage from color and depth image", "color"_a,
          "depth"_a, "depth_scale"_a = 1000.0, "depth_trunc"_a = 3.0,
          "convert_rgb_to_intensity"_a = true);
    docstring::FunctionDocInject(m, "create_rgbd_image_from_color_and_depth",
                                 map_shared_argument_docstrings);

    m.def("create_rgbd_image_from_redwood_format",
          &geometry::CreateRGBDImageFromRedwoodFormat,
          "Function to make RGBDImage (for Redwood format)", "color"_a,
          "depth"_a, "convert_rgb_to_intensity"_a = true);
    docstring::FunctionDocInject(m, "create_rgbd_image_from_redwood_format",
                                 map_shared_argument_docstrings);

    m.def("create_rgbd_image_from_tum_format",
          &geometry::CreateRGBDImageFromTUMFormat,
          "Function to make RGBDImage (for TUM format)", "color"_a, "depth"_a,
          "convert_rgb_to_intensity"_a = true);
    docstring::FunctionDocInject(m, "create_rgbd_image_from_tum_format",
                                 map_shared_argument_docstrings);

    m.def("create_rgbd_image_from_sun_format",
          &geometry::CreateRGBDImageFromSUNFormat,
          "Function to make RGBDImage (for SUN format)", "color"_a, "depth"_a,
          "convert_rgb_to_intensity"_a = true);
    docstring::FunctionDocInject(m, "create_rgbd_image_from_sun_format",
                                 map_shared_argument_docstrings);

    m.def("create_rgbd_image_from_nyu_format",
          &geometry::CreateRGBDImageFromNYUFormat,
          "Function to make RGBDImage (for NYU format)", "color"_a, "depth"_a,
          "convert_rgb_to_intensity"_a = true);
    docstring::FunctionDocInject(m, "create_rgbd_image_from_nyu_format",
                                 map_shared_argument_docstrings);
}
