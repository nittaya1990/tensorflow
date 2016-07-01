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

#include "tensorflow/core/debug/debug_graph_utils.h"

#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

DebugNodeInserter::DebugNodeInserter(
    const protobuf::RepeatedPtrField<DebugTensorWatch>& watches)
    : tensor_watches_() {
  // Cache the proto content for fast lookup later
  for (const DebugTensorWatch& watch : watches) {
    if (watch.output_slot() < 0) {
      // The semantics of output_slot == -1 is that the node is watched only
      // for completion, but not for output tensor values (see
      // NodeCompletionCallback in debug_gateway.h).
      continue;
    }
    if (watch.debug_ops().empty()) {
      // The semantics of debug_ops being an empty list is that the tensor
      // value _itself_ is watched, i.e., watched without any transformation,
      // not even identity transformation.
      continue;
    }

    string tensor_name =
        strings::StrCat(watch.node_name(), ":", watch.output_slot());

    std::vector<string> debug_ops;
    for (const string& debug_op : watch.debug_ops()) {
      debug_ops.push_back(debug_op);
    }

    tensor_watches_[tensor_name] = debug_ops;
  }
}

Status DebugNodeInserter::InsertNodes(Graph* graph) {
  // Record existing edges in the graph.
  std::vector<const Edge*> existing_edges;
  for (const Edge* edge : graph->edges()) {
    existing_edges.push_back(edge);
  }

  // Iterate through the edges, look for edges that match the tensor watch
  // list.
  for (const Edge* edge : existing_edges) {
    Node* src_node = edge->src();
    Node* dst_node = edge->dst();

    string tensor_name =
        strings::StrCat(src_node->name(), ":", edge->src_output());
    if (tensor_watches_.find(tensor_name) == tensor_watches_.end()) {
      // Add debug nodes only for edges with matching source node and source
      // output slot.
      continue;
    }

    const DataType src_dt = src_node->output_type(edge->src_output());

    for (int i = 0; i < tensor_watches_[tensor_name].size(); ++i) {
      // Determine the name of the debug node
      string debug_node_name =
          GetDebugNodeName(src_node->name(), edge->src_output(), i,
                           tensor_watches_[tensor_name][i]);

      // Create debug node
      auto builder =
          NodeDefBuilder(debug_node_name, tensor_watches_[tensor_name][i])
              .Input(src_node->name(), edge->src_output(), src_dt)
              .Attr("tensor_name", tensor_name);

      NodeDef node_def;
      const KernelDef* kdef;
      Node* debug_node;

      if (!builder.Finalize(&node_def).ok()) {
        return Status(
            error::FAILED_PRECONDITION,
            strings::StrCat("Failed to create node definition ",
                            "for debug op ", tensor_watches_[tensor_name][i]));
      }
      if (!FindKernelDef(DEVICE_CPU, node_def, &kdef, nullptr).ok()) {
        return Status(
            error::FAILED_PRECONDITION,
            strings::StrCat("Failed to find kernel definition ",
                            "for debug op ", tensor_watches_[tensor_name][i]));
      }
      if (!NodeBuilder(builder).Finalize(graph, &debug_node).ok()) {
        return Status(error::FAILED_PRECONDITION,
                      strings::StrCat("Failed to create debug node ",
                                      tensor_watches_[tensor_name][i]));
      }

      // Add new edges.
      // The 1st edge is from the watched node to the debug op.
      // The 2nd edge is a control edge from the debug op to the destination
      // node.
      graph->AddEdge(src_node, edge->src_output(), debug_node, 0);
      graph->AddEdge(debug_node, Graph::kControlSlot, dst_node,
                     Graph::kControlSlot);
    }
  }

  return Status::OK();
}

const string DebugNodeInserter::GetDebugNodeName(const string& node_name,
                                                 const int output_slot,
                                                 const int debug_op_num,
                                                 const string& debug_op_name) {
  // For example, if the watched node is named "node1" and the debug op that
  // watches the output slot of node1 is of the type "DebugNanCount", the
  // debug node will be called: __dbg_node1_0_0_DebugNanCount.
  return strings::StrCat("__dbg_", node_name, "_", output_slot, "_",
                         debug_op_num, "_", debug_op_name);
}

}  // namespace tensorflow
