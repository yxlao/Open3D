//
// Created by wei on 3/28/19.
//

#include "ScalableTSDFVolumeCudaIO.h"

namespace open3d {
namespace io {

template<size_t N>
bool WriteTSDFVolumeToBIN(const std::string &filename,
                          cuda::ScalableTSDFVolumeCuda<N> &volume) {
    auto key_value = volume.DownloadVolumes();

    auto keys = key_value.first;
    auto values = key_value.second;
    assert(keys.size() == values.size());

    FILE *fid = fopen(filename.c_str(), "wb");
    if (fid == NULL) {
        utility::PrintWarning("Write BIN failed: unable to open file: %s\n",
                              filename.c_str());
        return false;
    }

    int num_volumes = keys.size(), volume_size = N;

    /** metadata **/
    if (fwrite(&num_volumes, sizeof(int), 1, fid) < 1) {
        utility::PrintWarning("Write BIN failed: unable to write num volumes\n");
        return false;
    }
    if (fwrite(&volume_size, sizeof(int), 1, fid) < 1) {
        utility::PrintWarning("Write BIN failed: unable to write volume size\n");
        return false;
    }

    /** keys **/
    if (fwrite(keys.data(), sizeof(cuda::Vector3i), keys.size(), fid) < keys.size()) {
        utility::PrintWarning("Write BIN failed: unable to write keys\n");
        return false;
    }

    /** subvolumes **/
    utility::PrintInfo("Writing %d subvolumes.\n", keys.size());
    for (auto &subvolume : values) {
        auto &tsdf = std::get<0>(subvolume);
        auto &weight = std::get<1>(subvolume);
        auto &color = std::get<2>(subvolume);

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
    }

    fclose(fid);
    return true;
}

template<size_t N>
bool ReadTSDFVolumeFromBIN(const std::string &filename,
                           cuda::ScalableTSDFVolumeCuda<N> &volume) {
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

    /** metadata **/
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
    tsdf.resize(NNN);
    weight.resize(NNN);
    color.resize(NNN);

    /** keys **/
    if (fread(keys.data(), sizeof(cuda::Vector3i), keys.size(), fid) < keys.size()) {
        utility::PrintWarning("Read BIN failed: unable to read keys\n");
        return false;
    }

    /** values **/
    int batch_size = 5000;
    int num_batches = (num_volumes + batch_size - 1) / batch_size;

    std::vector<int> failed_indices;
    std::vector<cuda::Vector3i> failed_keys;
    std::vector<std::tuple<std::vector<float>, std::vector<uchar>, std::vector<cuda::Vector3b>>>
        failed_subvolumes;


    for (int b = 0; b < num_batches; ++b) {
        std::vector<cuda::Vector3i> block_keys;
        std::vector<std::tuple<std::vector<float>, std::vector<uchar>, std::vector<cuda::Vector3b>>>
            block_subvolumes;

        int begin = b * batch_size;
        int end = std::min((b + 1) * batch_size, num_volumes);

        for (int i = begin; i < end; ++i) {
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

            block_keys.emplace_back(keys[i]);
            block_subvolumes.emplace_back(std::make_tuple(tsdf, weight, color));
        }

        failed_indices = volume.UploadVolume(block_keys, block_subvolumes);
        for (auto &i : failed_indices) {
            failed_keys.emplace_back(block_keys[i]);
            failed_subvolumes.emplace_back(std::move(block_subvolumes[i]));
        }
    }
    fclose(fid);

    while (failed_indices.size() > 0) {
        failed_indices = volume.UploadVolume(failed_keys, failed_subvolumes);

        std::vector<cuda::Vector3i> tmp_failed_keys;
        std::vector<std::tuple<std::vector<float>, std::vector<uchar>, std::vector<cuda::Vector3b>>>
            tmp_failed_subvolumes;

        for (auto &i : failed_indices) {
            tmp_failed_keys.emplace_back(failed_keys[i]);
            tmp_failed_subvolumes.emplace_back(std::move(failed_subvolumes[i]));
        }

        std::swap(failed_keys, tmp_failed_keys);
        std::swap(failed_subvolumes, tmp_failed_subvolumes);
    }

    return true;
}

template
bool WriteTSDFVolumeToBIN(const std::string &filename,
                          cuda::ScalableTSDFVolumeCuda<8> &volume);
template
bool ReadTSDFVolumeFromBIN(const std::string &filename,
                           cuda::ScalableTSDFVolumeCuda<8> &volume);

} // io
} // open3d