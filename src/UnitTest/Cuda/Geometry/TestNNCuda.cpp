//
// Created by wei on 1/21/19.
//


#include <Cuda/Geometry/NNCuda.h>
#include <Eigen/Eigen>
#include <Core/Core.h>
#include <gtest/gtest.h>

using namespace open3d;
using namespace open3d::cuda;

TEST(NN, NNSearch) {
    const int size = 10000;
    const int feature_size = 33;
    Eigen::MatrixXd qr = Eigen::MatrixXd::Zero(feature_size, size);
    Eigen::MatrixXd reference = Eigen::MatrixXd::Zero(feature_size, size);

    for (int i = 0; i < size; ++i) {
        qr(0, i) = i;
        qr(feature_size - 1, i) = i;
        reference(0, i) = i;
        reference(feature_size - 1, i) = i;
    }

    NNCuda nn;
    Timer timer;
    timer.Start();
    nn.BruteForceNN(qr, reference);
    timer.Stop();
    PrintInfo("NNSearch takes: %f\n", timer.GetDuration());

    auto result = nn.nn_idx_.Download();
    for (int i = 0; i < size; ++i) {
        EXPECT_EQ(i, result(0, i));
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
