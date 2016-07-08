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

#ifndef TENSORFLOW_DEBUG_NODE_INSERTER_H_
#define TENSORFLOW_DEBUG_NODE_INSERTER_H_

#include <unordered_map>
#include <vector>

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

class DebugNodeInserter {
 public:
  DebugNodeInserter(
      const protobuf::RepeatedPtrField<DebugTensorWatch>& watches);
  virtual ~DebugNodeInserter() {}

  // EXPERIMENTAL: Insert special debug ops (e.g., DebugIdentity) to graph for
  // debugging. Currently, such ops need to take exactly one input and has the
  // string attribute "tensor_name" to indicate what tensor it watches.
  //
  // For exapmle, before the node insertion, the graph may look like:
  //
  //   A:0 -----------1----------- B
  //
  // wherein the output slot 0 of node A feeds as input to node B, through
  // edge 1.
  //
  // After the node insertion, the graph becomes:
  //
  //   A:0 -----------1----------> B
  //        |                      ^
  //        |                      |
  //         ----2---> D ----3-----
  //
  // where D is the inserted debug node specified in the DebugNodeInserter's
  // constructor argument, 2 is an edge that sends A:0 to D, and 3 is a
  // control edge from D to B.
  //
  // DebugIdentity is the simplest debugging paradigm, in which the debug
  // signal (i.e., output from D) equals the tensor itself. More sophisticated
  // debug ops can be used to transform the tensor into other debug signals
  // (e.g., count of zeros, count of NaNs).
  Status InsertNodes(Graph* graph);

  // Get canonical name of the debug node.
  static const string GetDebugNodeName(const string& node_name,
                                       const int output_slot,
                                       const int debug_op_num,
                                       const string& debug_op_name);

 private:
  // A map from tensor name (e.g., "node_a:0") to list of debug op names
  // (e.g., {"DebugIdentity", "DebugNanCount"})
  std::unordered_map<string, std::vector<string>> tensor_watches_;
  // A map from tensor name to bool value indicating whether the debug op
  // should deep-copy the input tensor.
  std::unordered_map<string, bool> do_deep_copy_;
};
}  // namespace tensorflow

#endif  // TENSORFLOW_DEBUG_NODE_INSERTER_H_
