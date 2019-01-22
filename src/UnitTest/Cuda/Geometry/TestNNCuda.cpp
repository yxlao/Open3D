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
    Eigen::Matrix<float, -1, -1, Eigen::RowMajor> query(size, 33);
    Eigen::Matrix<float, -1, -1, Eigen::RowMajor> reference(size, 33);

    for (int i = 0; i < size; ++i) {
        query(i, 0) = i;
        reference(i, 0) = i;
    }

//    for (int i = 0; i < 10; ++i) {
    NNCuda nn;
    Timer timer;
    timer.Start();
    nn.NNSearch(query, reference);
    timer.Stop();
    PrintInfo("NNSearch takes: %f\n", timer.GetDuration());
//    }

    auto result = nn.nn_idx_.Download();
    for (int i = 0; i < size; ++i) {
        if (i != result(i, 0)) {
            std::cout << i << " " << result(i, 0) << std::endl;
        }
    }

//    ::testing::InitGoogleTest(&argc, argv);
//    return RUN_ALL_TESTS();
}
