//
// Created by wei on 4/23/19.
//

#include <Open3D/Open3D.h>
#include <Material/Physics/HDRImage.h>

using namespace open3d;

int main() {
    std::string path = "/media/wei/Data/data/pbr/env/Mans_Outside_2k.hdr";

    geometry::HDRImage hdr;
    hdr.ReadFromHDR(path);
    visualization::DrawGeometries({hdr.image_});

    float *data = (float *)hdr.image_->data_.data();
    int num_floats = hdr.image_->width_ * hdr.image_->height_ * hdr.image_->num_of_channels_;
    for (int i = 0; i < num_floats; ++i) {
        data[i] = 1.0f;
    }

    hdr.WriteToHDR("/media/wei/Data/data/pbr/env/White.hdr");

    return 0;
}