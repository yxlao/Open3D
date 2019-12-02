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

#include "Open3D/Container/IndexingEngine.h"
#include "Open3D/Container/Device.h"
#include "Open3D/Container/SizeVector.h"

#include "Container/ContainerTest.h"
#include "TestUtility/UnitTest.h"

using namespace std;
using namespace open3d;

class IndexingEnginePermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(IndexingEngine,
                         IndexingEnginePermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

class IndexingEnginePermuteDevicePairs : public PermuteDevicePairs {};
INSTANTIATE_TEST_SUITE_P(
        IndexingEngine,
        IndexingEnginePermuteDevicePairs,
        testing::ValuesIn(IndexingEnginePermuteDevicePairs::TestCases()));

class IndexingEnginePermuteSizesDefaultStridesAndDevices
    : public testing::TestWithParam<
              std::tuple<std::pair<SizeVector, SizeVector>, Device>> {};
INSTANTIATE_TEST_SUITE_P(
        IndexingEngine,
        IndexingEnginePermuteSizesDefaultStridesAndDevices,
        testing::Combine(
                testing::ValuesIn(PermuteSizesDefaultStrides::TestCases()),
                testing::ValuesIn(PermuteDevices::TestCases())));

TEST_P(IndexingEnginePermuteDevices, TensorRef) {
    Device device = GetParam();

    Tensor t({2, 1, 3}, Dtype::Float32, device);
    TensorRef tr(t);

    EXPECT_EQ(tr.ndims_, 3);
    EXPECT_EQ(tr.dtype_byte_size_, 4);
    EXPECT_EQ(tr.data_ptr_, t.GetDataPtr());
    EXPECT_EQ(SizeVector(tr.shape_, tr.shape_ + 3), SizeVector({2, 1, 3}));
    EXPECT_EQ(SizeVector(tr.strides_, tr.strides_ + 3), SizeVector({3, 3, 1}));
}

// TEST_P(IndexingEnginePermuteDevices, BroadcastRestride) {
//     Device device = GetParam();

//     Tensor input0({2, 1, 3}, Dtype::Float32, device);
//     Tensor input1({1, 3}, Dtype::Float32, device);
//     Tensor output({2, 2, 2, 3}, Dtype::Float32, device);
//     IndexingEngine indexer({input0, input1}, output);

//     EXPECT_EQ(indexer.NumInputs(), 2);
//     TensorRef input0_tr = indexer.GetInput(0);
//     TensorRef input1_tr = indexer.GetInput(1);
//     TensorRef output_tr = indexer.GetOutput();

//     // Check ndims
//     EXPECT_EQ(input0_tr.ndims_, 4);
//     EXPECT_EQ(input1_tr.ndims_, 4);

//     // Check shapes
//     EXPECT_EQ(SizeVector(input0_tr.shape_, input0_tr.shape_ + 4),
//               SizeVector({1, 2, 1, 3}));
//     EXPECT_EQ(SizeVector(input1_tr.shape_, input1_tr.shape_ + 4),
//               SizeVector({1, 1, 1, 3}));
//     EXPECT_EQ(SizeVector(output_tr.shape_, output_tr.shape_ + 4),
//               SizeVector({2, 2, 2, 3}));
// }
