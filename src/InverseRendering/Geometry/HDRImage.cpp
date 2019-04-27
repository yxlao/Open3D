//
// Created by wei on 4/23/19.
//

#include "HDRImage.h"

namespace open3d {
namespace geometry {

bool HDRImage::ReadFromHDR(const std::string &filename, bool flip) {
    stbi_set_flip_vertically_on_load(flip);
    int width, height, channel;
    float *data = stbi_loadf(filename.c_str(), &width, &height, &channel, 0);
    if (! data) {
        utility::PrintError("Unable to load hdr image, abort.\n");
        return false;
    }

    if (! image_) {
        image_ = std::make_shared<Image>();
    }
    image_->PrepareImage(width, height, channel, sizeof(float));
    image_->data_.assign((unsigned char *) (&data[0]),
                         (unsigned char *) (&data[width * height * channel]));
    stbi_image_free(data);

    return true;
}

bool HDRImage::WriteToHDR(const std::string &filename, bool flip) {
    if (! image_) {
        utility::PrintError("Invalid hdr image, abort.\n");
        return false;
    }

    stbi_flip_vertically_on_write(flip);
    stbi_write_hdr(filename.c_str(),
                   image_->width_, image_->height_, image_->num_of_channels_,
                   (float*) image_->data_.data());
    return true;
}
}
}