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
    : tensor_watches_(), do_deep_copy_() {
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
    do_deep_copy_[tensor_name] = watch.deep_copy();
  }
}

Status DebugNodeInserter::InsertNodes(Graph* graph) {
  // 1. Record existing edges in the graph.
  std::vector<const Edge*> existing_edges;
  for (const Edge* edge : graph->edges()) {
    existing_edges.push_back(edge);
  }

  // A map from tensor names to edges to be removed
  std::unordered_map<string, std::vector<const Edge*>> edges_to_remove;
  // A map from tensor names to newly added debug nodes
  std::unordered_map<string, Node*> added_debug_nodes;

  // 2. Iterate through the edges, look for edges that match the tensor watch
  // list.
  for (const Edge* edge : existing_edges) {
    Node* src_node = edge->src();
    Node* dst_node = edge->dst();

    const string tensor_name =
        strings::StrCat(src_node->name(), ":", edge->src_output());
    if (tensor_watches_.find(tensor_name) == tensor_watches_.end()) {
      // Add debug nodes only for edges with matching source node and source
      // output slot.
      continue;
    }

    if (edges_to_remove.find(tensor_name) == edges_to_remove.end()) {
      // It is the first time edges with this source tensor is encountered:
      // we will:
      //   1) Mark this edge as to be removed
      //   2) Create a new (debug) node
      //   3) Add a new edge, from the source tensor to the debug node
      //   4) Add a new edge, from the debug node to the destination node
      std::vector<const Edge*> node_edges_to_remove;
      node_edges_to_remove.push_back(edge);
      edges_to_remove[tensor_name] = node_edges_to_remove;

      const DataType src_dt = src_node->output_type(edge->src_output());

      // Create debug node(s). Obey the ordering: the first debug op connects
      // with the watched tensor; the second debug op connects with the first
      // debug op; and so forth.
      Node* curr_src_node = src_node;
      int curr_src_output = edge->src_output();
      for (int i = 0; i < tensor_watches_[tensor_name].size(); ++i) {
        string debug_node_name =
            GetDebugNodeName(src_node->name(), edge->src_output(), i,
                             tensor_watches_[tensor_name][i]);

        // Logic for reference- vs. non-reference-type tensors
        string op_name = tensor_watches_[tensor_name][i];

        auto builder = NodeDefBuilder(debug_node_name, op_name)
                           .Input(src_node->name(), edge->src_output(), src_dt)
                           .Attr("tensor_name", tensor_name)
                           .Attr("deep_copy", do_deep_copy_[tensor_name]);

        NodeDef node_def;
        const KernelDef* kdef;
        Node* debug_node;

        if (!builder.Finalize(&node_def).ok()) {
          return Status(error::FAILED_PRECONDITION,
                        strings::StrCat("Failed to create node definition ",
                                        "for debug op ", op_name,
                                        " on watched tensor ", tensor_name));
        }
        if (!FindKernelDef(DEVICE_CPU, node_def, &kdef, nullptr).ok()) {
          return Status(error::FAILED_PRECONDITION,
                        strings::StrCat("Failed to find kernel definition ",
                                        "for debug op ", op_name,
                                        " on watched tensor ", tensor_name));
        }
        if (!NodeBuilder(builder).Finalize(graph, &debug_node).ok()) {
          return Status(error::FAILED_PRECONDITION,
                        strings::StrCat("Failed to create debug node ", op_name,
                                        " on watched tensor ", tensor_name));
        }

        // Record the debug node for later use
        added_debug_nodes[tensor_name] = debug_node;

        // Add new edges.
        // The 1st edge is from the watched (source) tensor to the debug op
        // (in the case of the first debug op), or from the previous debug op
        // to the current one (in the case of the non-first debug op).
        graph->AddEdge(curr_src_node, curr_src_output, debug_node, 0);
        curr_src_node = debug_node;
        curr_src_output = 0;

        // This edge is a from the output_slot 0 of the debug op to the
        // destination node (in the case of the last debug op).
        if (i == tensor_watches_[tensor_name].size() - 1) {
          graph->AddEdge(debug_node, 0, dst_node, edge->dst_input());
        }
      }
    } else {
      // This not the first time an edge with this source tensor is seen.
      // We will:
      //   1) Mark this edge for removal
      //   2) Create an edge from the output_slot 0 of the debug op to the
      //      destination node.
      edges_to_remove[tensor_name].push_back(edge);

      Node* debug_node = added_debug_nodes[tensor_name];

      // Add new edge: from output_slot 0 of the debug op to the destination
      // node.
      graph->AddEdge(debug_node, 0, dst_node, edge->dst_input());
    }
  }

  // Remove all edges marked for removal
  for (auto it : edges_to_remove) {
    std::vector<const Edge*> edges = it.second;

    for (const Edge* edge : edges) {
      graph->RemoveEdge(edge);
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
