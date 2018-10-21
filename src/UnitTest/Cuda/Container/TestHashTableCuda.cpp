//
// Created by wei on 9/24/18.
//

#include <Cuda/Container/ArrayCuda.h>
#include <Cuda/Container/LinkedListCuda.h>
#include <Cuda/Container/HashTableCuda.h>
#include <Cuda/Common/UtilsCuda.h>
#include <Core/Core.h>
#include <cuda.h>
#include <vector>
#include <unordered_map>
#include <cstdlib>
#include <random>
#include "UnitTest.h"


TEST(HashTableCuda, HashTableProfiling) {
    using namespace open3d;

    Timer timer;
    std::random_device rd;
    std::default_random_engine rd_engine(rd());

    HashTableCuda<Vector3i, int, SpatialHasher> table;
    table.Create(15, 300);

    int num_pairs = 300;
    std::vector<Vector3i> keys;
    std::vector<int> values;

    keys.resize(num_pairs);
    values.resize(num_pairs);

    std::uniform_int_distribution<> dist(0, num_pairs);
    std::unordered_map<Vector3i, int, SpatialHasher> pairs;
    for (int i = 0; i < num_pairs; ++i) {
        keys[i] =
            Vector3i(dist(rd_engine), dist(rd_engine), dist(rd_engine));
        values[i] = dist(rd_engine);
        pairs[keys[i]] = values[i];
    }

    /* give more tries */
    int num_iters = 60;
    int num_per_iter = num_pairs / num_iters;
    for (int i = 0; i < num_iters; ++i) {
        std::vector<Vector3i> subkeys(
            keys.begin() + i * num_per_iter,
            keys.begin() + (i + 1) * num_per_iter);
        std::vector<int> subvalues(
            values.begin() + i * num_per_iter,
            values.begin() + (i + 1) * num_per_iter);
        table.New(subkeys, subvalues);
        table.ResetLocks();
    }
    auto downloaded = table.Download();
    std::vector<Vector3i> downloaded_keys = std::get<0>(downloaded);
    std::vector<int> downloaded_values = std::get<1>(downloaded);
    for (int i = 0; i < downloaded_keys.size(); ++i) {
        EXPECT_EQ(pairs[downloaded_keys[i]], downloaded_values[i]);
    }
    PrintInfo("Uploading passed, %d / %d entries uploaded.\n",
              downloaded_keys.size(), keys.size());

    auto profile = table.Profile();
    std::vector<int> array_entry_count = std::get<0>(profile);
    std::vector<int> list_entry_count = std::get<1>(profile);
    PrintInfo("Profiling occupied array entries and linked list entries "
              "per bucket...\n");
    for (int i = 0; i < array_entry_count.size(); ++i) {
        PrintInfo("> %d: %d %d\n", i,
            array_entry_count[i], list_entry_count[i]);
    }

    for (int i = 0; i < num_iters; ++i) {
        std::vector<Vector3i> subkeys(
            keys.begin() + i * num_per_iter,
            keys.begin() + (i + 1) * num_per_iter);
        table.Delete(subkeys);
        table.ResetLocks();
    }
    downloaded = table.Download();
    downloaded_keys = std::get<0>(downloaded);
    downloaded_values = std::get<1>(downloaded);
    PrintInfo("Deletion passed, %d entries remains.\n",
              downloaded_keys.size());

    profile = table.Profile();
    array_entry_count = std::get<0>(profile);
    list_entry_count = std::get<1>(profile);
    PrintInfo("Profiling occupied array entries and linked list entries "
              "per bucket...\n");
    for (int i = 0; i < array_entry_count.size(); ++i) {
        PrintInfo("> %d: d %d\n", i,
            array_entry_count[i], list_entry_count[i]);
    }

    table.Release();
}

TEST(HashTableCuda, HashTableInsertionAndDelete) {
    using namespace open3d;

    Timer timer;
    std::random_device rd;
    std::default_random_engine rd_engine(rd());

    HashTableCuda<Vector3i, int, SpatialHasher> table;
    const int bucket_count = 400000;
    table.Create(bucket_count, 2000000);

    int num_pairs = 2000000;
    std::vector<Vector3i> keys;
    std::vector<int> values;

    keys.resize(num_pairs);
    values.resize(num_pairs);

    std::uniform_int_distribution<> dist(0, num_pairs);
    std::unordered_map<Vector3i, int, SpatialHasher> pairs;
    for (int i = 0; i < num_pairs; ++i) {
        keys[i] =
            Vector3i(dist(rd_engine), dist(rd_engine), dist(rd_engine));
        values[i] = dist(rd_engine);
        pairs[keys[i]] = values[i];
    }

    /* give more tries */
    int iters = 1000;
    int num_per_iter = num_pairs / iters;
    for (int i = 0; i < iters; ++i) {
        std::vector<Vector3i> subkeys(
            keys.begin() + i * num_per_iter,
            keys.begin() + (i + 1) * num_per_iter);
        std::vector<int> subvalues(
            values.begin() + i * num_per_iter,
            values.begin() + (i + 1) * num_per_iter);
        table.New(subkeys, subvalues);
        table.ResetLocks();
    }
    auto downloaded = table.Download();
    std::vector<Vector3i> downloaded_keys = std::get<0>(downloaded);
    std::vector<int> downloaded_values = std::get<1>(downloaded);
    PrintInfo("Uploading passed, %d / %d entries uploaded.\n",
              downloaded_keys.size(), keys.size());
    auto profile = table.Profile();
    std::vector<int> array_entry_count = std::get<0>(profile);
    std::vector<int> list_entry_count = std::get<1>(profile);
    PrintInfo("Profiling occupied array entries and linked list entries "
              "per bucket...\n");
    int array_entry_cnt = 0, list_entry_cnt = 0;
    for (int i = 0; i < (int) array_entry_count.size(); ++i) {
        array_entry_cnt += array_entry_count[i];
        list_entry_cnt += list_entry_count[i];
    }
    PrintInfo("Average %.2f entries per array, %.2f entries per linked "
              "list\n",
              array_entry_cnt / (float) bucket_count,
              list_entry_cnt / (float) bucket_count
    );

    for (int i = 0; i < iters; ++i) {
        std::vector<Vector3i> subkeys(
            keys.begin() + i * num_per_iter,
            keys.begin() + (i + 1) * num_per_iter);
        table.Delete(subkeys);
        table.ResetLocks();
    }
    downloaded = table.Download();
    downloaded_keys = std::get<0>(downloaded);
    downloaded_values = std::get<1>(downloaded);
    PrintInfo("Delete passed, %d entries remains.\n",
              downloaded_keys.size());

    profile = table.Profile();
    array_entry_count = std::get<0>(profile);
    list_entry_count = std::get<1>(profile);
    PrintInfo("Profiling occupied array entries and linked list entries "
              "per bucket...\n");
    array_entry_cnt = 0;
    list_entry_cnt = 0;
    for (int i = 0; i < (int) array_entry_count.size(); ++i) {
        array_entry_cnt += array_entry_count[i];
        list_entry_cnt += list_entry_count[i];
    }
    PrintInfo("Average %.2f entries per array, %.2f entries per linked "
              "list\n",
              array_entry_cnt / (float) bucket_count,
              list_entry_cnt / (float) bucket_count
    );
    table.Release();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    RUN_ALL_TESTS();

    return 0;
}
