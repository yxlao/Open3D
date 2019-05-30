//
// Created by wei on 4/27/19.
//

#include <stb_image/stb_image_write.h>
#include <stb_image/stb_image.h>
#include "ImageExt.h"

namespace open3d {
namespace geometry {

std::shared_ptr<Image> ConvertImageFromFloatImage(const Image &image) {
    auto uimage = std::make_shared<geometry::Image>();
    if (image.IsEmpty()) {
        return uimage;
    }

    uimage->PrepareImage(image.width_, image.height_,
                         image.num_of_channels_, 1);

    int num_pixels = image.height_ * image.width_;
    for (int i = 0; i < num_pixels; i++) {
        uint8_t *p = (uint8_t *) (uimage->data_.data()
            + i * uimage->num_of_channels_);

        float *pf = (float *) (image.data_.data()
            + i * image.num_of_channels_ * image.bytes_per_channel_);

        for (int k = 0; k < image.num_of_channels_; ++k) {
            p[k] = uint8_t(std::abs(pf[k]) * 255);
        }
    }

    return uimage;
}

std::shared_ptr<Image> FlipImageExt(const Image &input) {
    auto output = std::make_shared<Image>();
    output->PrepareImage(input.width_, input.height_,
        input.num_of_channels_, input.bytes_per_channel_);

    const int stride = input.width_ * input.num_of_channels_ * input.bytes_per_channel_;
    for (int y = 0; y < input.height_; y++) {
        auto input_ptr = input.data_.data() + y * stride;
        auto output_ptr = output->data_.data() + (input.height_ - 1 - y) * stride;
        memcpy(output_ptr, input_ptr, stride);
    }
    return output;
}
}

namespace io {
bool WriteImageToHDR(const std::string &filename, const geometry::Image &image) {
    if (image.IsEmpty()) {
        utility::PrintError("Invalid hdr image, abort.\n");
        return false;
    }

    stbi_flip_vertically_on_write(true);
    stbi_write_hdr(filename.c_str(),
                   image.width_, image.height_, image.num_of_channels_,
                   (float*) image.data_.data());
    return true;
}

std::shared_ptr<geometry::Image> ReadImageFromHDR(const std::string &filename) {
    auto image = std::make_shared<geometry::Image>();

    stbi_set_flip_vertically_on_load(true);
    int width, height, channel;
    float *data = stbi_loadf(filename.c_str(), &width, &height, &channel, 0);
    if (! data) {
        utility::PrintError("Unable to load hdr image, abort.\n");
        return image;
    }

    image->PrepareImage(width, height, channel, sizeof(float));
    image->data_.assign((unsigned char *) (&data[0]),
                        (unsigned char *) (&data[width * height * channel]));
    stbi_image_free(data);

    return image;
}
}
}