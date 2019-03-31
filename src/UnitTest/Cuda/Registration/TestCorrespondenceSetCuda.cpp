//
// Created by wei on 1/15/19.
//

#include <Cuda/Registration/CorrespondenceSetCuda.h>
#include <gtest/gtest.h>
#include <iostream>
using namespace open3d;
using namespace open3d::cuda;

TEST(CorrespondenceSetCuda, Comress) {
    CorrespondenceSetCuda correspondence_set;
    Eigen::Matrix<int, -1, -1> corres_cpu(3, 10000);
    for (int i = 0; i < corres_cpu.cols(); ++i) {
        corres_cpu(0, i) = (i & 1) ? -1 : i;
    }

    correspondence_set.SetCorrespondenceMatrix(corres_cpu);
    correspondence_set.Compress();

    std::vector<int> corres_indices =
        correspondence_set.indices_.Download();
    EXPECT_EQ(corres_indices.size(), corres_cpu.cols() >> 1);

    for (auto &index : corres_indices) {
        EXPECT_EQ(index & 1, 0);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}