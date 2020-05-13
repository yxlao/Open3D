// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "Open3D/Core/Device.h"
#include "Open3D/Core/AdvancedIndexing.h"
#include "Open3D/Core/Dtype.h"
#include "Open3D/Core/Kernel/Kernel.h"
#include "Open3D/Core/MemoryManager.h"
#include "Open3D/Core/SizeVector.h"
#include "Open3D/Core/Tensor.h"
#include "Open3D/Utility/Helper.h"

#include "Core/CoreTest.h"
#include "TestUtility/UnitTest.h"

using namespace std;
using namespace open3d;

TEST(Device, DefaultConstructor) {
    Device ctx;
    EXPECT_EQ(ctx.GetType(), Device::DeviceType::CPU);
    EXPECT_EQ(ctx.GetID(), 0);
}

TEST(Device, CPUMustBeID0) {
    EXPECT_EQ(Device(Device::DeviceType::CPU, 0).GetID(), 0);
    EXPECT_THROW(Device(Device::DeviceType::CPU, 1), std::runtime_error);
}

TEST(Device, SpecifiedConstructor) {
    Device ctx(Device::DeviceType::CUDA, 1);
    EXPECT_EQ(ctx.GetType(), Device::DeviceType::CUDA);
    EXPECT_EQ(ctx.GetID(), 1);
}

TEST(Device, StringConstructor) {
    Device ctx("CUDA:1");
    EXPECT_EQ(ctx.GetType(), Device::DeviceType::CUDA);
    EXPECT_EQ(ctx.GetID(), 1);
}

TEST(Device, StringConstructorLower) {
    Device ctx("cuda:1");
    EXPECT_EQ(ctx.GetType(), Device::DeviceType::CUDA);
    EXPECT_EQ(ctx.GetID(), 1);
}

TEST(Device, TensorPtr) {
    Device device("CPU:0");
    Tensor a;
    a = Tensor::Ones({2, 3}, Dtype::Float32, device);
    a = Tensor(std::vector<float>({0, 1, 2, 3, 4, 5}), {2, 3}, Dtype::Float32,
               device);
    a = a.Contiguous();

    EXPECT_EQ(a.GetShape(), SizeVector({2, 3}));
    EXPECT_EQ(a.NumElements(), 6);

    std::cout << a.ToString() << std::endl;
    std::cout << a[0].ToString() << std::endl;

    // float* ptr = static_cast<float*>(a.GetDataPtr());
    // std::cout << ptr << std::endl;
    // std::cout << ptr[0] << std::endl;
    // std::cout << ptr[1] << std::endl;
    // std::cout << ptr[2] << std::endl;
}

class KNN {
public:
    KNN(const Tensor& tensor) {
        if (tensor.NumDims() != 2) {
            utility::LogError("must be N * 2");
        }
        dim_ = tensor.GetShape()[1];
    }

    int KNNSearch(const T& query,
                  int knn,
                  std::vector<int>& indices,
                  std::vector<float>& distance2);

    int RadiusSearch(const T& query,
                     float radius,
                     std::vector<int>& indices,
                     std::vector<float>& distance2);

private:
    int dim_ = 0;
};
