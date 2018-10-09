//
// Created by wei on 9/24/18.
//

#include <Cuda/Container/ArrayCuda.h>
#include <Cuda/Container/LinkedListCuda.h>
#include <Cuda/Container/HashTableCuda.h>
#include <Core/Core.h>

#include <vector>
#include <unordered_map>
#include <cstdlib>
#include <random>
#include "UnitTest.h"

TEST(ContainerCuda, ArrayFill) {
    using namespace open3d;

    Timer timer;
    std::random_device rd;
    std::default_random_engine rd_engine(rd());

    timer.Start();

    ArrayCuda<int> array;
    const int kMaxCapacity = 1000000;
    const int kFilledValue = 1203;

    array.Create(kMaxCapacity);
    array.Fill(kFilledValue);
    std::vector<int> downloaded = array.DownloadAll();
    for (auto &val : downloaded) {
        EXPECT_EQ(val, kFilledValue);
    }
    timer.Stop();
    PrintInfo("> ArrayCuda.Fill() passed in %.2f seconds.\n",
              timer.GetDuration() * 0.001f);
    array.Release();
}

TEST(ContainerCuda, ArrayUploadAndDownload) {
    using namespace open3d;
    Timer timer;
    std::random_device rd;
    std::default_random_engine rd_engine(rd());

    timer.Start();

    ArrayCuda<int> array;
    const int kMaxCapacity = 1000000;
    array.Create(kMaxCapacity);

    std::vector<int> random_vec;
    random_vec.resize(kMaxCapacity / 2);
    std::uniform_int_distribution<> dist(0, kMaxCapacity);
    for (auto &val : random_vec) {
        val = dist(rd_engine);
    }

    array.Upload(random_vec);
    std::vector<int> downloaded = array.Download();
    EXPECT_EQ(random_vec.size(), downloaded.size());

    for (int i = 0; i < (int) random_vec.size(); ++i) {
        EXPECT_EQ(random_vec[i], downloaded[i]);
    }
    timer.Stop();
    PrintInfo("ArrayCuda.Upload() and ArrayCuda.Download() "
              "passed in %.2f seconds.\n",
              timer.GetDuration() * 0.001f);
    array.Release();
}

TEST(ContainerCuda, LinkedListInsertAndDownload) {
    using namespace open3d;

    Timer timer;
    std::random_device rd;
    std::default_random_engine rd_engine(rd());

    /**
     * Test LinkedList.Insert() and LinkedList.Download()
     **/

    timer.Start();

    MemoryHeapCuda<LinkedListNodeCuda<int>> memory_heap;
    const int kMaxCapacity = 1000000;

    memory_heap.Create(kMaxCapacity);

    LinkedListCuda<int> linked_list1, linked_list2;
    linked_list1.Create(kMaxCapacity / 2, memory_heap);
    linked_list2.Create(kMaxCapacity / 2, memory_heap);

    int num_samples = kMaxCapacity / 4;
    std::vector<int> values;
    values.resize(num_samples);
    std::unordered_map<int, int> value_cnt;
    std::uniform_int_distribution<> dist(0, num_samples);
    for (int i = 0; i < num_samples; ++i) {
        int val = dist(rd_engine);
        values[i] = val;
        value_cnt[val] += 1;
    }
    linked_list1.Insert(values);

    std::vector<int> downloaded_values = linked_list1.Download();
    EXPECT_EQ(downloaded_values.size(), num_samples);

    std::unordered_map<int, int> downloaded_value_cnt = value_cnt;
    int valid_values = 0;
    for (int i = 0; i < num_samples; ++i) {
        int val = downloaded_values[i];
        EXPECT_NE(value_cnt.find(val), value_cnt.end());
        downloaded_value_cnt[val]--;
        if (downloaded_value_cnt[val] == 0) {
            valid_values++;
        }
        EXPECT_GE(downloaded_value_cnt[val], 0);
    }
    EXPECT_EQ(valid_values, value_cnt.size());

    timer.Stop();
    PrintInfo("LinkedListCuda.Insert() and LinkedListCuda.Download() "
              "passed in %.2f seconds.\n",
              timer.GetDuration() * 0.001f);

    linked_list1.Release();
    linked_list2.Release();
    memory_heap.Release();
}

TEST(ContainerCuda, LinkedListInsertAndDelete) {
    using namespace open3d;

    Timer timer;
    std::random_device rd;
    std::default_random_engine rd_engine(rd());

    /* A more complex case */
    timer.Start();

    MemoryHeapCuda<LinkedListNodeCuda<int>> memory_heap;
    const int kMaxCapacity = 1000000;

    memory_heap.Create(kMaxCapacity);

    LinkedListCuda<int> linked_list1, linked_list2;
    linked_list1.Create(kMaxCapacity / 2, memory_heap);
    linked_list2.Create(kMaxCapacity / 2, memory_heap);

    std::vector<int> insert_values[2];
    std::vector<int> insert_and_delete_values[2];
    int num_samples = 1000;
    insert_values[0].resize(num_samples);
    insert_values[1].resize(num_samples);
    insert_and_delete_values[0].resize(num_samples);
    insert_and_delete_values[1].resize(num_samples);

    std::uniform_int_distribution<> dist(0, num_samples);
    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < 2; ++j) {
            insert_values[j][i] = dist(rd_engine);
            insert_and_delete_values[j][i] = dist(rd_engine);
        }
    }

    std::vector<int> downloaded_values;
    std::unordered_map<int, int> downloaded_value_cnt;
    std::unordered_map<int, int> value_cnt;

    linked_list1.Insert(insert_and_delete_values[0]);
    downloaded_values = linked_list1.Download();
    for (auto &val : insert_and_delete_values[0]) {
        value_cnt[val] += 1;
    }
    for (auto &val : downloaded_values) {
        downloaded_value_cnt[val] += 1;
    }
    EXPECT_EQ(value_cnt.size(), downloaded_value_cnt.size());

    for (auto &it : value_cnt) {
        EXPECT_EQ(downloaded_value_cnt[it.first], it.second);
    }
    PrintInfo("#1 Insertion passed\n");

    linked_list1.Insert(insert_values[0]);
    downloaded_values = linked_list1.Download();
    for (auto &val : insert_values[0]) {
        value_cnt[val] += 1;
    }
    downloaded_value_cnt.clear();
    for (auto &val : downloaded_values) {
        downloaded_value_cnt[val] += 1;
    }
    EXPECT_EQ(value_cnt.size(), downloaded_value_cnt.size());
    for (auto &it : value_cnt) {
        EXPECT_EQ(downloaded_value_cnt[it.first], it.second);
    }
    PrintInfo("#2 Insertion passed\n");

    linked_list1.Delete(insert_and_delete_values[0]);
    downloaded_values = linked_list1.Download();
    for (auto &val : insert_and_delete_values[0]) {
        value_cnt[val] -= 1;
        if (value_cnt[val] == 0) {
            value_cnt.erase(val);
        }
    }

    downloaded_value_cnt.clear();
    for (auto &val : downloaded_values) {
        downloaded_value_cnt[val] += 1;
    }
    EXPECT_EQ(value_cnt.size(), downloaded_value_cnt.size());
    for (auto &it : value_cnt) {
        EXPECT_EQ(downloaded_value_cnt[it.first], it.second);
    }
    PrintInfo("#3 Deletion passed\n");

    linked_list1.Insert(insert_and_delete_values[1]);
    linked_list1.Insert(insert_values[1]);
    downloaded_values = linked_list1.Download();
    for (auto &val : insert_and_delete_values[1]) {
        value_cnt[val] += 1;
    }
    for (auto &val : insert_values[1]) {
        value_cnt[val] += 1;
    }
    downloaded_value_cnt.clear();
    for (auto &val : downloaded_values) {
        downloaded_value_cnt[val] += 1;
    }
    EXPECT_EQ(value_cnt.size(), downloaded_value_cnt.size());
    for (auto &it : value_cnt) {
        EXPECT_EQ(downloaded_value_cnt[it.first], it.second);
    }
    PrintInfo("#4 Double Insertion passed\n");

    linked_list1.Delete(insert_and_delete_values[1]);
    downloaded_values = linked_list1.Download();
    for (auto &val : insert_and_delete_values[1]) {
        value_cnt[val] -= 1;
        if (value_cnt[val] == 0) {
            value_cnt.erase(val);
        }
    }
    downloaded_value_cnt.clear();
    for (auto &val : downloaded_values) {
        downloaded_value_cnt[val] += 1;
    }
    EXPECT_EQ(value_cnt.size(), downloaded_value_cnt.size());

    for (auto &it : value_cnt) {
        EXPECT_EQ(downloaded_value_cnt[it.first], it.second);
    }
    PrintInfo("#5 Deletion\n");

    timer.Stop();
    PrintInfo("LinkedListCuda.Insert() and LinkedListCuda.Download() "
              "complex cases passed in %.2f seconds.\n",
              timer.GetDuration() * 0.001f);
}

TEST(ContainerCuda, HashTableProfiling) {
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
    int num_iters = 30;
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

TEST(ContainerCuda, HashTableInsertionAndDelete) {
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
