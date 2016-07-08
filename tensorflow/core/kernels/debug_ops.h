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

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/core/notification.h"

namespace tensorflow {

// Identity op for debugging.
//
// The op produces two outputs:
//   Output slot 0 carries the pass through tensor. If the input tensor is
//   reference-type, the pass-through output will also be a reference.
//   Output slot 1 carries the debug signal and is always allocated on the
//   host (CPU) and is not reference-type. In the case of DebugIdentityOp,
//   the debug signal is equal to the input tensor. Other debug ops may
//   inherit from DebugIdentityOp and generate non-trivial transformation
//   of the input tensor.
template <typename T>
class DebugIdentityOp : public OpKernel {
 public:
  explicit DebugIdentityOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("tensor_name", &tensor_name_));
    OP_REQUIRES_OK(context, context->GetAttr("deep_copy", &deep_copy_));
  }

  ~DebugIdentityOp() override {
    if (tensor_copy_) delete tensor_copy_;
  }

  void Compute(OpKernelContext* context) override {
    std::cout << "In DebugIdentityOp::Compute: tensor_name = " << tensor_name_
              << std::endl;  // DEBUG

    // Input to the GetDebugSignal method that produces the debug signal on
    // the second output slot (output1).
    const Tensor* debug_input;
    std::unique_ptr<Tensor> host_tensor;

    const bool is_input_ref = IsRefType(context->input_dtype(0));
    const Tensor& src_tensor = is_input_ref ? context->mutable_input(0, false)
        : context->input(0);

    DeviceContext* device_ctxt = context->op_device_context();
    Device* device = static_cast<Device*>(context->device());

    // Determine if the input tensor is not on CPU (e.g., on GPU).
    bool off_host_input = device->name().find("gpu:") != string::npos &&
                          !context->input_alloc_attr(0).on_host();

    if (off_host_input) {
      // Input is not on host (e.g., on GPU), copy it to host before
      // generating the debug signal.
      host_tensor.reset(new Tensor(tensorflow::cpu_allocator(),
				   src_tensor.dtype(), src_tensor.shape()));

      Notification done_copy_to_cpu;
      device_ctxt->CopyDeviceTensorToCPU(
          &src_tensor, "TensorCopy", device, host_tensor.get(),
	  [&done_copy_to_cpu](const Status& s) {
	    done_copy_to_cpu.Notify();
	  });
      done_copy_to_cpu.WaitForNotification();
      std::cout << "CopyDeviceTensorToCPU() done: host_tensor = "
                << host_tensor->DebugString() << std::endl
                << std::flush;  // DEBUG
      debug_input = host_tensor.get();
    } else {
      // The input tensor is on the host (CPU). No copying is necessary.
      debug_input = &src_tensor;
    }

    if (is_input_ref) {
      // If deep_copy == true, we will forward the copy as reference to the
      // first output (i.e., the pass-through).
      // If deep_copy == false, we will forward the original reference.
      // Produce the first output (the pass-through).
      if (deep_copy_) {
	if (tensor_copy_ == nullptr) {
	  // Tensor dtype and shape should not change. So initializing once
	  // is sufficient.
	  Allocator* allocator =
              device->GetAllocator(context->output_alloc_attr(0));
	  tensor_copy_ =
	      new Tensor(allocator, src_tensor.dtype(), src_tensor.shape());
	}

        if (off_host_input) {
          // Copy the host copy of the GPU tensor back to GPU
          Notification done_copy_to_device;
          device_ctxt->CopyCPUTensorToDevice(
              host_tensor.get(), device, tensor_copy_,
              [&done_copy_to_device](const Status& s) {
                done_copy_to_device.Notify();
              });
          done_copy_to_device.WaitForNotification();
        } else {
	  // Make a deep copy of the tensor on CPU
	  *tensor_copy_ = tensor::DeepCopy(src_tensor);
	}

        context->set_output_ref(0, &mu_, tensor_copy_);
      } else {
        context->forward_ref_input_to_ref_output(0, 0);
      }
    } else {
      // Input is not reference-type. The tensor output cannot change after it
      // is produced from the watched node. So there is no need to copy it
      // even if deep_copy_ is true.
      context->set_output(0, context->input(0));
    }

    std::cout << "  tensor_name = " << tensor_name_ << ": debug_input = " << debug_input
              << std::endl;
    std::cout << "  tensor_name = " << tensor_name_ << ": debug_input->DebugString() = "
              << debug_input->DebugString() << std::endl;

    // Produce the second output (debug signal), which is identical to the input
    // in the case of this kernel.
    context->set_output(1, GetDebugSignal(*debug_input));
  }

  bool IsExpensive() override { return false; }

 protected:
  // Method that converts the input (i.e., watched) tensor into a debug
  // signal. The base implementation is the simplest case: identity mapping.
  // Subclasses may override this method to implement other types of mapping.
  virtual const Tensor GetDebugSignal(const Tensor& tensor) { return tensor; }

 private:
  string tensor_name_;
  bool deep_copy_;

  // Deep-copied tensor from input. This is used if deep_copy is true and if
  // the input is a reference-type tensor, in which case this OpKernel has
  // ownership of the deep-copied tensor.
  Tensor* tensor_copy_ = nullptr;
  mutex mu_;
};

// NaN value counter op for debugging
template <typename T>
class DebugNanCountOp : public DebugIdentityOp<T> {
 public:
  explicit DebugNanCountOp(OpKernelConstruction* context)
      : DebugIdentityOp<T>(context) {}

 protected:
  const Tensor GetDebugSignal(const Tensor& tensor) override {
    const TensorShape& input_shape = tensor.shape();
    const T* input_flat = tensor.template flat<T>().data();

    // Count NaNs
    int64 nan_count = 0;
    for (int64 i = 0; i < input_shape.num_elements(); ++i) {
      if (Eigen::numext::isnan(input_flat[i])) {
        nan_count++;
      }
    }

    // Use DT_INT64/int64 to be consistent with TensorShape::num_elements()
    DataType dtype = DataType::DT_INT64;
    TensorShape shape({1});

    Tensor debug_signal(tensorflow::cpu_allocator(), dtype, shape);
    debug_signal.vec<int64>()(0) = nan_count;

    return debug_signal;
  }
};

// TODO(cais): Add DebugInfinityCount
// TODO(cais): Add DebugZeroCount

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_IDENTITY_OP_H_
