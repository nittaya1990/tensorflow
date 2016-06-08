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

#ifndef TENSORFLOW_DEBUG_DEBUG_SESSION_H_
#define TENSORFLOW_DEBUG_DEBUG_SESSION_H_

#include <unordered_map>

#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/core/common_runtime/executor.h"

namespace tensorflow {

class DebugSession : public DirectSession {
 public:
  DebugSession(const SessionOptions& options, const DeviceMgr* device_mgr);
  ~DebugSession() override {
    ClearHostTensors();
  }

  // Callback for node completion. The value of the output tensor is not
  // necessarily available when this callback is invoked. It may need to be
  // asynchronously copied from device (e.g., GPU) to host, hence the need
  // for the NodeValueCallback below.
  typedef std::function<void(const string& node_name, const int output_slot,
                             const bool is_ref)>
      NodeCompletionCallback;
  void SetNodeCompletionCallback(NodeCompletionCallback callback);

  // Callback for node value. This is invoked when the value of a node's
  // output tensor is available on the host, possibly after copying from
  // a device (e.g., GPU).
  typedef std::function<void(const string& node_name, const int output_slot,
                             const Tensor& tensor_value, const bool is_ref)>
      NodeValueCallback;
  void SetNodeValueCallback(NodeValueCallback callback);

  Status Run(const std::vector<std::pair<string, Tensor> >& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs) override;

  void SetOptimizeGraph(const bool optimize_graph) {
    DirectSession::SetOptimizeGraph(optimize_graph);
  }

 private:
  NodeCompletionCallback comp_cb_ = nullptr;
  NodeValueCallback val_cb_ = nullptr;

  mutex mu_;
  std::unordered_map<string, const Tensor*> host_tensors_ GUARDED_BY(mu_);

  typedef std::function<void(const Tensor* dst_tensor)> CopyDoneCallback;

  void CopyTensor(const string& node_name, const int output_slot,
                  const Tensor* src_tensor, OpKernelContext* ctx,
                  CopyDoneCallback copy_done_cb);

  void ClearHostTensors();
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_DEBUG_DEBUG_SESSION_H_
