/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/array_ops.cc.
#include "tensorflow/core/kernels/debug_ops.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {

// Register debug identity (non-ref and ref) ops.
// For the ref op, also, register on CPU.
#define REGISTER_DEBUG_IDENTITY(type)                                     \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("DebugIdentity").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      DebugIdentityOp<type>);

#define REGISTER_DEBUG_REF_IDENTITY(type)                                    \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("DebugRefIdentity").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      DebugIdentityOp<type>);

TF_CALL_ALL_TYPES(REGISTER_DEBUG_IDENTITY);
TF_CALL_ALL_TYPES(REGISTER_DEBUG_REF_IDENTITY);

#if GOOGLE_CUDA
#define REGISTER_GPU_DEBUG_REF_IDENTITY(type)             \
  REGISTER_KERNEL_BUILDER(Name("DebugRefIdentity")        \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("debug_signal") \
                              .TypeConstraint<type>("T"), \
                          DebugIdentityOp<type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_DEBUG_REF_IDENTITY);
#endif

// Register debug NaN-counter (non-ref and ref) ops.
// For the ref op, also, register on CPU.
#define REGISTER_DEBUG_NAN_COUNT(type)                                    \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("DebugNanCount").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      DebugNanCountOp<type>);

#define REGISTER_DEBUG_REF_NAN_COUNT(type)                                   \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("DebugRefNanCount").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      DebugNanCountOp<type>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_DEBUG_NAN_COUNT);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_DEBUG_REF_NAN_COUNT);

#if GOOGLE_CUDA
#define REGISTER_GPU_DEBUG_REF_NAN_COUNT(type)            \
  REGISTER_KERNEL_BUILDER(Name("DebugRefNanCount")        \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("debug_signal") \
                              .TypeConstraint<type>("T"), \
                          DebugNanCountOp<type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_DEBUG_REF_NAN_COUNT);
#endif

}  // namespace tensorflow
