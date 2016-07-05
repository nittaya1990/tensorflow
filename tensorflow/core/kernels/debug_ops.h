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
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

// Identity op for debugging
class DebugIdentityOp : public OpKernel {
 public:
  explicit DebugIdentityOp(OpKernelConstruction* context) 
      : OpKernel(context), env_(Env::Default()) {
    OP_REQUIRES_OK(context, context->GetAttr("tensor_name", &tensor_name_));
  }

  void Compute(OpKernelContext* context) override {
    // std::cout << "DebugIdentityOp: tensor_name = " << tensor_name_ << std::endl;  // DEBUG
    
    if (IsRefType(context->input_dtype(0))) {
      std::cout << "TODO: Ref input" << std::endl; // DEBUG
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      const Tensor& input_tensor = context->input(0);
      
      string filename = strings::StrCat("/tmp/tfdb1/", tensor_name_);
      
      string filedir = GetFileDir(filename);
      if (!filedir.empty() && !env_->FileExists(filedir)) {
        RecursiveCreateDir(filedir);
      }
      
      // std::cout << "  filename = " << filename << std::endl;  // DEBUG
      // std::cout << "  tensor = " << input_tensor.DebugString() << std::endl;  // DEBUG
      WriteTensorToFile(input_tensor, filename);

      context->set_output(0, input_tensor);

    }
  }

  bool IsExpensive() override { return false; }

 private:
  void WriteTensorToFile(const Tensor& tensor, const string& filename) {
    // First, serialize tensor to string
    TensorProto tensor_proto;
    string tensor_str;

    tensor.AsProtoField(&tensor_proto);
    // tensor.AsProtoTensorContent(&tensor_proto);
    tensor_proto.AppendToString(&tensor_str);

    // std::cout << "Original tensor \"" << tensor_name_
              // << "\" = " << tensor.DebugString() << std::endl;  // DEBUG

    // Second, open the RecordIO file and write the seralization 
    Status s = env_->NewWritableFile(filename, &recordio_file_);
    if (!s.ok()) {
      std::cerr << "Could not open events file: " << filename << ": " << s << std::endl;
    }
    recordio_writer_.reset(new io::RecordWriter(recordio_file_.get()));
    if (recordio_writer_.get() == NULL) {
      std::cerr << "Could not create record writer" << std::endl;
    }

    recordio_writer_->WriteRecord(tensor_str);
    recordio_file_->Flush();

    // Last, clean up
    recordio_writer_.reset(NULL);
    recordio_file_.reset(NULL);

    // // Test reading
    // std::unique_ptr<RandomAccessFile> ra_file;
    // env_->NewRandomAccessFile(filename, &ra_file);
    // io::RecordReader record_reader(ra_file.get());

    // string readout;
    // uint64 offset = 0;
    // record_reader.ReadRecord(&offset, &readout);

    // // DEBUG: Parse tensor_str back to Tensor
    // TensorProto tensor_proto_2;
    // tensor_proto_2.ParseFromString(readout);
    // Tensor tensor2; 
    // tensor2.FromProto(tensor_proto_2);

    // std::cout << "Readout tensor \"" << tensor_name_
    //           << "\" = " << tensor2.DebugString() << std::endl;  // DEBUG
  }

  void RecursiveCreateDir(const string& dir) {
    string parent_dir = GetFileDir(dir);
    if (!parent_dir.empty() && !env_->FileExists(parent_dir)) {
      RecursiveCreateDir(parent_dir);  // Recursive call
    }

    env_->CreateDir(dir);
  }

  string GetFileDir(const string& filename) {
    size_t last_delim_idx = filename.rfind("/");  // TODO(cais): 
    if (last_delim_idx != string::npos && last_delim_idx != 0) {
      return filename.substr(0, last_delim_idx);
    } else {
      return string("");
    }
  }

  string tensor_name_;

  Env* env_;
  std::unique_ptr<WritableFile> recordio_file_;
  std::unique_ptr<io::RecordWriter> recordio_writer_;
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
