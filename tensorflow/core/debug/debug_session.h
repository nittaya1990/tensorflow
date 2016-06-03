/* Copyright 2016 TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/core/common_runtime/executor.h"

namespace tensorflow {

class DebugSession : public DirectSession {
 public:
  DebugSession(const SessionOptions& options, const DeviceMgr* device_mgr);
  ~DebugSession() override {}

  // Callback for node completion. The value of the output tensor is not
  // necessarily available when this callback is invoked. It may need to be
  // asynchronously copied from device (e.g., GPU) to host, hence the need
  // for the NodeValueCallback below.
  typedef std::function<void(const string& node_name,
                             const int64& completion_timestamp,
                             const bool is_ref)>
      NodeCompletionCallback;
  void SetNodeCompletionCallback(NodeCompletionCallback callback);

  // Callback for node value. This is invoked when the value of a node's
  // output tensor is available on the host, possibly after copying from
  // a device (e.g., GPU).
  typedef std::function<void(const string& node_name,
                             const Tensor& tensor_value,
                             const bool is_ref)> NodeValueCallback;
  void SetNodeValueCallback(NodeValueCallback callback);

 private:
  // Custom node output callback
  Executor::Args::NodeOutputCallback custom_node_output_cbk_;

  NodeCompletionCallback comp_cbk_ = nullptr;
  NodeValueCallback val_cbk_ = nullptr;
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_DEBUG_DEBUG_SESSION_H_
