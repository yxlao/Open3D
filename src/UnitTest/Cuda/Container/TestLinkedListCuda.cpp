//
// Created by wei on 10/17/18.
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

TEST(LinkedListCuda, LinkedListInsertAndDownload) {
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
    memory_heap.Resize(kMaxCapacity * 2);
    linked_list1.UpdateServer();
    linked_list2.UpdateServer();

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

TEST(LinkedListCuda, LinkedListInsertAndDelete) {
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

    linked_list2 = linked_list1;
    downloaded_values = linked_list2.Download();
    downloaded_value_cnt.clear();
    for (auto &val : downloaded_values) {
        downloaded_value_cnt[val] += 1;
    }
    EXPECT_EQ(value_cnt.size(), downloaded_value_cnt.size());
    for (auto &it : value_cnt) {
        EXPECT_EQ(downloaded_value_cnt[it.first], it.second);
    }
    PrintInfo("#5 Assignment passed\n");

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
    PrintInfo("#6 Deletion\n");

    timer.Stop();
    PrintInfo("LinkedListCuda.Insert() and LinkedListCuda.Download() "
              "complex cases passed in %.2f seconds.\n",
              timer.GetDuration() * 0.001f);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}