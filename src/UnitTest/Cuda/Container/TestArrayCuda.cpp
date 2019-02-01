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
#include <gtest/gtest.h>

using namespace open3d;
using namespace open3d::cuda;

TEST(ArrayCuda, ArrayFill) {
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
}

TEST(ArrayCuda, ArrayUploadAndDownload) {
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
}

TEST(ArrayCuda, ArrayResize) {
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
    array.Resize(kMaxCapacity * 2);

    std::vector<int> downloaded = array.Download();
    EXPECT_EQ(random_vec.size(), downloaded.size());

    for (int i = 0; i < (int) random_vec.size(); ++i) {
        EXPECT_EQ(random_vec[i], downloaded[i]);
    }
    timer.Stop();
    PrintInfo("ArrayCuda.Upload(), Resize(), and ArrayCuda.Download() "
              "passed in %.2f seconds.\n",
              timer.GetDuration() * 0.001f);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}