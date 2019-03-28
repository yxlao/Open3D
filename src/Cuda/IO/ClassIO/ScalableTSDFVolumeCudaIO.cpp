//
// Created by wei on 3/28/19.
//

#include "ScalableTSDFVolumeCudaIO.h"

namespace open3d {
namespace io {

template<size_t N>
bool ReadTSDFVolumeToBIN(const std::string &filename,
                         const cuda::ScalableTSDFVolumeCuda<N> &volume) {
    auto key_value = volume.DownloadVolumes();

    auto keys = key_value.first;
    auto values = key_value.second;

    std::vector<float> &tsdf = std::get<0>(values);
    std::vector<uchar> &weight = std::get<1>(values);
    std::vector<cuda::Vector3b> &color = std::get<2>(values);

    int NNN = N * N * N;
    assert(keys.size() == tsdf.size() * NNN);
    assert(tsdf.size() == weight.size());
    assert(tsdf.size() == color.size());

    FILE *fid = fopen(filename.c_str(), "wb");
    if (fid == NULL) {
        utility::PrintWarning("Write BIN failed: unable to open file: %s\n",
                              filename.c_str());
        return false;
    }

    int num_volumes = keys.size(), volume_size = N;
    if (fwrite(&num_volumes, sizeof(int), 1, fid) < 1) {
        utility::PrintWarning("Write BIN failed: unable to write num volumes\n");
        return false;
    }
    if (fwrite(&volume_size, sizeof(int), 1, fid) < 1) {
        utility::PrintWarning("Write BIN failed: unable to write volume size\n");
        return false;
    }

    if (fwrite(keys.data(), sizeof(cuda::Vector3i), keys.size(), fid) < keys.size()) {
        utility::PrintWarning("Write BIN failed: unable to write keys\n");
        return false;
    }
    if (fwrite(tsdf.data(), sizeof(float), tsdf.size(), fid) < tsdf.size()) {
        utility::PrintWarning("Write BIN failed: unable to write TSDF\n");
        return false;
    }
    if (fwrite(weight.data(), sizeof(uchar), weight.size(), fid) < weight.size()) {
        utility::PrintWarning("Write BIN failed: unable to write weight\n");
        return false;
    }
    if (fwrite(color.data(), sizeof(cuda::Vector3b), color.size(), fid) < color.size()) {
        utility::PrintWarning("Write BIN failed: unable to write color\n");
        return false;
    }

    fclose(fid);
    return true;
}

template<size_t N>
bool WriteTSDFVolumeFromBIN(const std::string &filename,
                            const cuda::ScalableTSDFVolumeCuda<N> &volume) {
    FILE *fid = fopen(filename.c_str(), "rb");
    if (fid == NULL) {
        utility::PrintWarning("Read BIN failed: unable to open file: %s\n",
                              filename.c_str());
        return false;
    }

    int num_volumes, volume_size;
    std::vector<cuda::Vector3i> keys;
    std::vector<float> tsdf;
    std::vector<uchar> weight;
    std::vector<cuda::Vector3b> color;

    if (fread(&num_volumes, sizeof(int), 1, fid) < 1) {
        utility::PrintWarning("Read BIN failed: unable to read num volumes\n");
        return false;
    }
    if (fread(&volume_size, sizeof(int), 1, fid) < 1) {
        utility::PrintWarning("Read BIN failed: unable to read volume size\n");
        return false;
    }
    assert(volume_size == N);
    int NNN = N * N * N;

    keys.resize(num_volumes);
    tsdf.resize(num_volumes * NNN);
    weight.resize(num_volumes * NNN);
    color.resize(num_volumes * NNN);

    if (fread(keys.data(), sizeof(cuda::Vector3i), keys.size(), fid) < keys.size()) {
        utility::PrintWarning("Read BIN failed: unable to read keys\n");
        return false;
    }
    if (fread(tsdf.data(), sizeof(float), tsdf.size(), fid) < tsdf.size()) {
        utility::PrintWarning("Read BIN failed: unable to read TSDF\n");
        return false;
    }
    if (fread(weight.data(), sizeof(uchar), weight.size(), fid) < weight.size()) {
        utility::PrintWarning("Read BIN failed: unable to read weight\n");
        return false;
    }
    if (fread(color.data(), sizeof(cuda::Vector3b), color.size(), fid) < color.size()) {
        utility::PrintWarning("Read BIN failed: unable to read color\n");
        return false;
    }

    fclose(fid);


    return true;
}

} // io
} // open3d