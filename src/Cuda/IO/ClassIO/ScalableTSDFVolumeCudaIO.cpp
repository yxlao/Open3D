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

    /** metadata **/
    int num_volumes = keys.size(), volume_size = N;
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
        auto &tsdf = subvolume.tsdf_;
        auto &weight = subvolume.weight_;
        auto &color = subvolume.color_;

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
                           cuda::ScalableTSDFVolumeCuda<N> &volume,
                           int batch_size) {
    FILE *fid = fopen(filename.c_str(), "rb");
    if (fid == NULL) {
        utility::PrintWarning("Read BIN failed: unable to open file: %s\n",
                              filename.c_str());
        return false;
    }

    /** metadata **/
    int num_volumes, volume_size;
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

    /** keys **/
    std::vector<cuda::Vector3i> keys(num_volumes);
    if (fread(keys.data(), sizeof(cuda::Vector3i), keys.size(), fid) < keys.size()) {
        utility::PrintWarning("Read BIN failed: unable to read keys\n");
        return false;
    }

    /** values **/
    std::vector<float> tsdf_buffer(NNN);
    std::vector<uchar> weight_buffer(NNN);
    std::vector<cuda::Vector3b> color_buffer(NNN);

    std::vector<int> failed_key_indices;
    std::vector<cuda::Vector3i> failed_keys;
    std::vector<cuda::ScalableTSDFVolumeCpuData> failed_subvolumes;

    /** Updating (key, value) pairs in parallel is tricky:
     * - thread lock can forbid some block to be allocated;
     * - split all the pairs them into batches can increase success rate;
     * - we should retry to insert stubborn failure pairs until they get in. **/
    if (batch_size <= 0) {
        batch_size = int(volume.hash_table_.bucket_count_ * 0.2f);
    }
    int num_batches = (num_volumes + batch_size - 1) / batch_size;
    for (int batch = 0; batch < num_batches; ++batch) {
        std::vector<cuda::Vector3i> batch_keys;
        std::vector<cuda::ScalableTSDFVolumeCpuData> batch_subvolumes;

        int begin = batch * batch_size;
        int end = std::min((batch + 1) * batch_size, num_volumes);

        for (int i = begin; i < end; ++i) {
            if (fread(tsdf_buffer.data(), sizeof(float), tsdf_buffer.size(),
                fid) < tsdf_buffer.size()) {
                utility::PrintWarning("Read BIN failed: unable to read TSDF\n");
                return false;
            }
            if (fread(weight_buffer.data(), sizeof(uchar), weight_buffer.size(),
                fid) < weight_buffer.size()) {
                utility::PrintWarning("Read BIN failed: unable to read weight\n");
                return false;
            }
            if (fread(color_buffer.data(), sizeof(cuda::Vector3b),
                color_buffer.size(), fid) < color_buffer.size()) {
                utility::PrintWarning("Read BIN failed: unable to read color\n");
                return false;
            }

            batch_keys.emplace_back(keys[i]);
            batch_subvolumes.emplace_back(cuda::ScalableTSDFVolumeCpuData(
                tsdf_buffer, weight_buffer, color_buffer));
        }

        volume.UploadVolumes(batch_keys, batch_subvolumes);
    }
    fclose(fid);

    return true;
}

template
bool WriteTSDFVolumeToBIN(const std::string &filename,
                          cuda::ScalableTSDFVolumeCuda<8> &volume);
template
bool ReadTSDFVolumeFromBIN(const std::string &filename,
                           cuda::ScalableTSDFVolumeCuda<8> &volume,
                           int batch_size);

} // io
} // open3d