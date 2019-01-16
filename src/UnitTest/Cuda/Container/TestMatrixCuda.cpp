//
// Created by wei on 1/14/19.
//

#include <Eigen/Eigen>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <Cuda/Container/Array2DCuda.h>

TEST(Eigen, RawData) {
    Eigen::Matrix<float, 3, 5, Eigen::RowMajor> matrix;
    matrix << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14;

    std::cout << matrix << std::endl;
    for (int i = 0; i < 15; ++i) {
        std::cout << *(matrix.data() + i) << " ";
    }
}

TEST(Eigen, UploadAndDownload) {
    std::random_device rd;
    std::default_random_engine engine(rd());
    std::uniform_int_distribution<int> uniform(100, 10000);
    int rows = uniform(engine);
    int cols = 30;

    std::cout << "rows: " << rows << std::endl;
    Eigen::Matrix<float, -1, -1, Eigen::RowMajor> matrix(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix(i, j) = i + j;
            EXPECT_EQ(*(matrix.data() + i * cols + j), i + j);
        }
    }

    open3d::cuda::Array2DCuda<float> matrix_cuda;
    matrix_cuda.Create(rows, cols);
    matrix_cuda.Upload(matrix);

    EXPECT_EQ(matrix_cuda.Download(), matrix);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}