//
// Created by wei on 1/15/19.
//

// #include <Cuda/Registration/ColoredICPCuda.h>
#include <gtest/gtest.h>
#include <Eigen/Eigen>
#include <iostream>

TEST(ColoredICPCuda, StdVectorToEigen) {
    Eigen::Matrix<int, -1, -1, Eigen::RowMajor> matrix =
        Eigen::Matrix<int, -1, -1, Eigen::RowMajor>::Constant(8, 6, -1);

    std::vector<int> indices{1, 3, 4, 2, 5};
    Eigen::Map<Eigen::RowVectorXi> row_vector(indices.data(), indices.size());

    matrix.block(2, 0, 1, indices.size()) = row_vector;

    std::cout << matrix << std::endl;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}