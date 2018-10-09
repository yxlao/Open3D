//
// Created by wei on 10/3/18.
//

#include <Cuda/Geometry/TransformCuda.h>
#include <Eigen/Eigen>
#include <Core/Core.h>

#include "UnitTest.h"

TEST(TransformCuda, Transform) {
    using namespace open3d;

    for (int i = 0; i < 1000; ++i) {
        /* Generate random R & t */
        Eigen::Vector3f w = Eigen::Vector3f::Random();
        float theta = w.norm();
        w = w / theta;
        Eigen::Matrix3f w_tilde;
        w_tilde << 0, -w(2), w(1),
            w(2), 0, -w(0),
            -w(1), w(0), 0;
        Eigen::Matrix3f R = Eigen::Matrix3f::Identity() + sin(theta) * w_tilde +
            (1 - cos(theta)) * (w_tilde * w_tilde);
        Eigen::Vector3f t = Eigen::Vector3f::Random();
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        T.block<3, 3>(0, 0) = R;
        T.block<3, 1>(0, 3) = t;

        Eigen::Vector3f v = Eigen::Vector3f::Random();

        TransformCuda transform_cuda;
        transform_cuda.FromEigen(R, t);
        float
            matrix_norm = (T.inverse() - transform_cuda.inv().ToEigen()).norm();
        EXPECT_LE(matrix_norm, 1e-6);

        Vector3f v_cuda;
        v_cuda.FromEigen(v);
        Vector3f Tv_cuda = transform_cuda * v_cuda;
        float vector_norm = ((T * v.homogeneous()).hnormalized() -
            (transform_cuda * v_cuda).ToEigen()).norm();
        EXPECT_LE(vector_norm, 1e-6);
    }
    PrintInfo("Transform tests passed\n");

    Vector1f v;
    v(0) = 1.0f;
    v = 1 * v;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
