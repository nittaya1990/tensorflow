/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_KERNELS_DEBUG_OP_H_
#define TENSORFLOW_KERNELS_DEBUG_OP_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

// Identity op for debugging
class DebugIdentityOp : public OpKernel {
 public:
  explicit DebugIdentityOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("tensor_name", &tensor_name_));
  }

  void Compute(OpKernelContext* context) override {
    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      context->set_output(0, context->input(0));
    }
  }

  bool IsExpensive() override { return false; }

 private:
  string tensor_name_;
};

// NaN value counter op for debugging
template <typename T>
class DebugNanCountOp : public OpKernel {
 public:
  explicit DebugNanCountOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("tensor_name", &tensor_name_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const TensorShape& input_shape = input.shape();
    const T* input_flat = input.template flat<T>().data();

    int64 output = 0;
    for (int64 i = 0; i < input_shape.num_elements(); ++i) {
      if (Eigen::numext::isnan(input_flat[i])) {
        output++;
      }
    }

    // Use DT_INT64/int64 to be consistent with TensorShape::num_elements()
    DataType dtype = DataType::DT_INT64;
    TensorShape shape({1});

    Allocator* allocator = tensorflow::cpu_allocator();
    Tensor output_tensor(allocator, dtype, shape);
    output_tensor.vec<int64>()(0) = output;

    context->set_output(0, output_tensor);
  }

  bool IsExpensive() override { return false; }

 private:
  string tensor_name_;
};

// TODO(cais): Add DebugInfinityCount
// TODO(cais): Add DebugZeroCount

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_IDENTITY_OP_H_
