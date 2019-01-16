//
// Created by wei on 1/15/19.
//

#include <Cuda/Common/LinearAlgebraCuda.h>
#include <gtest/gtest.h>

TEST(Eigen, Eigen) {
    Eigen::Matrix3d A;
    A << 4, 12, -16, 12, 37, -43, -16, -43, 98;
    Eigen::Vector3d b(1, 0, 1);

    for (int i = 0; i < 100; ++i) {
        Eigen::Matrix3d A = Eigen::Matrix3d::Zero();

        for (int s = 0; s < 20; ++s) {
            Eigen::Vector3d a = Eigen::Vector3d::Random();
            A += a * a.transpose();
        }
        b = Eigen::Vector3d::Random();
        std::cout << A.ldlt().solve(b).transpose() << std::endl;

        open3d::cuda::MatrixCuda<float, 3, 3> matrix;
        matrix.FromEigen(A);
        open3d::cuda::VectorCuda<float, 3> vector;
        vector.FromEigen(b);
        open3d::cuda::LDLT<float, 3, 3> ldlt(matrix);
        std::cout << ldlt.Solve(vector).ToEigen().transpose() << std::endl;

        std::cout << matrix.ldlt().Solve(vector).ToEigen().transpose()
                  << std::endl;
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}