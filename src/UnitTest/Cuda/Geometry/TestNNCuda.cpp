//
// Created by wei on 1/21/19.
//


#include <Cuda/Geometry/NNCuda.h>
#include <Eigen/Eigen>
#include <Core/Core.h>
#include <gtest/gtest.h>

using namespace open3d;
using namespace open3d::cuda;

int main(int argc, char **argv) {
    const int size = 10000;
    const int feature_size = 33;
    Eigen::Matrix<float, -1, -1, Eigen::RowMajor> qr
    = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>::Zero(feature_size, size);
    Eigen::Matrix<float, -1, -1, Eigen::RowMajor> reference
    = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>::Zero(feature_size, size);

    for (int i = 0; i < size; ++i) {
        qr(0, i) = i;
        qr(feature_size - 1, i) = i;
        reference(0, i) = i;
        reference(feature_size - 1, i) = i;
    }

//    for (int i = 0; i < 10; ++i) {
    NNCuda nn;
    Timer timer;
    timer.Start();
    nn.NNSearch(qr, reference);
    timer.Stop();
    PrintInfo("NNSearch takes: %f\n", timer.GetDuration());

//    std::cout << nn.query_.Download() << std::endl;
//    std::cout << nn.reference_.Download() << std::endl;
//    std::cout << nn.distance_matrix_.Download() << std::endl;
//    }

    auto result = nn.nn_idx_.Download();
//    std::cout << result << std::endl;
    for (int i = 0; i < size; ++i) {
        if (i != result(0, i)) {
            std::cout << i << " " << result(i, 0) << std::endl;
        }
    }

//    ::testing::InitGoogleTest(&argc, argv);
//    return RUN_ALL_TESTS();
}
