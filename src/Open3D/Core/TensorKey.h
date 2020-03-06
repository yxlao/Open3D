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

#pragma once

#include "Open3D/Utility/Console.h"

namespace open3d {

class TensorKey {
public:
    TensorKey(){};
    virtual ~TensorKey() {}
};

class TensorIndex : public TensorKey {
public:
    TensorIndex(int64_t index) : index_(index) {}
    int64_t index_ = 0;
};

class TensorSlice : public TensorKey {
public:
    TensorSlice(int64_t start, int64_t stop, int64_t step)
        : TensorSlice(start, stop, step, false, false, false) {}

    TensorSlice(int64_t start,
                int64_t stop,
                int64_t step,
                bool start_is_none,
                bool stop_is_none,
                bool step_is_none)
        : start_(start),
          stop_(stop),
          step_(step),
          start_is_none_(start_is_none),
          stop_is_none_(stop_is_none),
          step_is_none_(step_is_none) {}

public:
    /// Slice non (i.e. keep all) in a dimension, i.e. t[:].
    /// Usage in C++: t.GetItem(TensorSlice.All());
    static TensorSlice None() { return TensorSlice(0, 0, 0, true, true, true); }

    /// When dim_size is know, convert the slice object such that
    /// start_is_none_ == stop_is_none_ == step_is_none_ == false
    /// E.g. if t.shape == (5,), t[:4]:
    ///      before compute: Slice(None, 4, None)
    ///      after compute : Slice(   0, 4,    1)
    TensorSlice UpdateWithDimSize(int64_t dim_size) const {
        return TensorSlice(start_is_none_ ? 0 : start_,
                           stop_is_none_ ? dim_size : stop_,
                           step_is_none_ ? 1 : step_);
    }

    int64_t start_ = 0;
    int64_t stop_ = 0;
    int64_t step_ = 0;
    bool start_is_none_ = false;
    bool stop_is_none_ = false;
    bool step_is_none_ = false;
};

};  // namespace open3d
