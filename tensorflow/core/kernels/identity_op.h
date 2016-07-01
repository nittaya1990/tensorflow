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

#ifndef TENSORFLOW_KERNELS_IDENTITY_OP_H_
#define TENSORFLOW_KERNELS_IDENTITY_OP_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/env.h"  // tfdb-modgraph
#include "tensorflow/core/platform/macros.h"  // tfdb-modgraph
#include "tensorflow/core/platform/types.h"  // tfdb-modgraph
#include "tensorflow/core/lib/io/record_writer.h"  // tfdb-modgraph
#include "tensorflow/core/lib/core/status.h"  // tfdb-modgraph
#include "tensorflow/core/lib/core/threadpool.h"  // tfdb-modgraph
#include "tensorflow/core/platform/mutex.h"  // tfdb-modgraph


namespace tensorflow {

class IdentityOp : public OpKernel {
 public:
  explicit IdentityOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      context->set_output(0, context->input(0));
    }
  }

  bool IsExpensive() override { return false; }
};

// tfdb-modgraph
class DebugOp : public OpKernel {
 public:
  explicit DebugOp(OpKernelConstruction* context) 
      : OpKernel(context), env_(Env::Default()) {
    OP_REQUIRES_OK(context, context->GetAttr("tensor_name", &tensor_name_));

    string thread_pool_name = strings::StrCat("debug_thread_", tensor_name_);
    thread_pool_.reset(new thread::ThreadPool(
        context->env(), thread_pool_name, 1
      ));
  }

  void Compute(OpKernelContext* context) override {
    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      std::cout << "||| " << tensor_name_ << ": "
                << context->input(0).SummarizeValue(3) << std::endl << std::flush; // For debugging

      TensorProto tensor_proto;
      context->input(0).AsProtoTensorContent(&tensor_proto);

      // Write to file
      std::unique_ptr<WritableFile> recordio_file_;
      std::unique_ptr<io::RecordWriter> recordio_writer_;
      string filename_ = strings::StrCat("/tmp/debugger_output_", tensor_name_);
      Status s = env_->NewWritableFile(filename_, &recordio_file_);

      if (!s.ok()) {
        LOG(ERROR) << "Could not open debug dump file: " << filename_ << ": " << s;
      } else {
        recordio_writer_.reset(new io::RecordWriter(recordio_file_.get()));
        if (recordio_writer_.get() == NULL) {
          LOG(ERROR) << "Could not create record writer";
        } else {
          string tensor_proto_str;
          tensor_proto.SerializeToString(&tensor_proto_str);
          recordio_writer_->WriteRecord(tensor_proto_str);
        }
      }

      // Clean up
      if (recordio_file_.get() != nullptr) {
        recordio_file_->Close();
      }
      recordio_writer_.reset(NULL);
      recordio_file_.reset(NULL);
      
      context->set_output(0, context->input(0));
    }
  }

  bool IsExpensive() override { return false; }
 private:
  static mutex mu_;
  string tensor_name_;

  Env* env_;
  std::unique_ptr<thread::ThreadPool> thread_pool_;
};

// Initialize static mutex for DebugOp
// mutex DebugOp::mu_;

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_IDENTITY_OP_H_
