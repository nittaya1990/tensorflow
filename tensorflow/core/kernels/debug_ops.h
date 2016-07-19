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

#include <fstream>

#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/events_writer.h"
#include "tensorflow/core/util/event.pb.h"

// TODO(cais): Check for unused includes

namespace tensorflow {

// Helper class for debug ops.
class DebugOpsHelper {
 public:
  static void RecursiveCreateDir(Env* env, const string& dir) {
    string parent_dir = GetFileDir(dir);

    // TODO(cais): what if parent_dir is actually a file (not a dir)?
    if (!parent_dir.empty() && !env->FileExists(parent_dir)) {
      RecursiveCreateDir(env, parent_dir);  // Recursive call
    }

    env->CreateDir(dir);
  }

  static string GetFileDir(const string& filename) {
     // TODO(cais): Support other platforms such as Windows?
    size_t last_delim_idx = filename.rfind("/");
    if (last_delim_idx != string::npos && last_delim_idx != 0) {
      return filename.substr(0, last_delim_idx);
    } else {
      return string("");
    }
  }

  static void WriteTensorToFile(const Tensor& tensor, const string& filename) {
    Env* env(Env::Default());

    // Create the directory if necessary.
    string filedir = DebugOpsHelper::GetFileDir(filename);
    if (!filedir.empty() && !env->FileExists(filedir)) {
      DebugOpsHelper::RecursiveCreateDir(env, filedir);
    }

    // Encapsulate the tensor value inside a Summary proto, and then an Event
    // proto.
    Event event;
    event.set_wall_time(Env::Default()->NowMicros());
    Summary::Value* summ_val = event.mutable_summary()->add_value();

    // TODO(cais): Confusing node name with tensor name may cause problems?
    summ_val->set_node_name(tensor_name_);
    if (tensor.dtype() == DT_STRING) {
      tensor.AsProtoField(summ_val->mutable_tensor());
    } else {
      tensor.AsProtoTensorContent(summ_val->mutable_tensor());
    }

    string event_str;
    event.AppendToString(&event_str);

    std::fstream ofs(filename, std::ios::out | std::ios::binary);
    event.SerializeToOstream(&ofs);
  }
};

// Copy op for debugging.
// Performs CPU-to-CPU or GPU-to-GPU deep-copying of tensor, depending on the
// device on which the tensor is allocated.
class CopyOp : public OpKernel {
 public:
  explicit CopyOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("tensor_name", &tensor_name_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& src_tensor = context->input(0);

    DeviceContext* device_ctxt = context->op_device_context();
    Device* device = static_cast<Device*>(context->device());

    // Determine if the input tensor is not on CPU (e.g., on GPU).
    bool off_host_input = device->device_type() == DEVICE_GPU &&
                          !context->input_alloc_attr(0).on_host();

    Tensor* copied_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, src_tensor.shape(),
                                                     &copied_tensor));
    if (off_host_input) {
      // Input is not on host: deep-copy it from GPU to the same GPU.
      Notification done_copy;
      GPUUtil::CopyGPUTensorToSameGPU(
          device, device_ctxt, &src_tensor, copied_tensor,
          [&done_copy](const Status& s) { done_copy.Notify(); });
      done_copy.WaitForNotification();
    } else {
      // The input tensor is on the host (CPU): deep-copy from CPU to CPU.
      *copied_tensor = tensor::DeepCopy(src_tensor);
    }
  }

  bool IsExpensive() override { return false; }

 private:
  string tensor_name_;
};

// Identity op for debugging.
//   Output slot 0 carries the debug signal and is always allocated on the
//   host (CPU) as a non-Ref tensor. In the case of DebugIdentityOp,
//   the debug signal is equal to the input tensor.
class DebugIdentityOp : public OpKernel {
 public:
  explicit DebugIdentityOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("tensor_name", &tensor_name_));
    OP_REQUIRES_OK(context, context->GetAttr("debug_url", &debug_url_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);

    if (!debug_url_.empty()) {
      const string kProtocolPrefixFile = "file:/";

      // Require support protocols.
      OP_REQUIRES(context, debug_url_.find(kProtocolPrefixFile) == 0,
                  errors::InvalidArgument(strings::StrCat("Unsupported debug URL protocol in ", debug_url_)));

      // TODO(cais): Create directory if it does not exist.
      string file_path(debug_url_);
      file_path.replace(0, kProtocolPrefixFile.size(), "");

      DebugOpsHelper::WriteTensorToFile(input, file_path);
    }

    context->set_output(0, context->input(0));
  }

  bool IsExpensive() override { return false; }

 private:
  string tensor_name_;
  string debug_url_;
};

// NaN-counter op for debugging.
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

    // Count NaNs.
    // Use DT_INT64/int64 to be consistent with TensorShape::num_elements().
    int64 nan_count = 0;
    for (int64 i = 0; i < input_shape.num_elements(); ++i) {
      if (Eigen::numext::isnan(input_flat[i])) {
        nan_count++;
      }
    }

    TensorShape shape({1});

    Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output_tensor));
    output_tensor->vec<int64>()(0) = nan_count;
  }

  bool IsExpensive() override { return false; }

 private:
  string tensor_name_;
};

// TODO(cais): Add DebugInfinityCount
// TODO(cais): Add DebugZeroCount

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_IDENTITY_OP_H_
