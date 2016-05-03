/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/debug_session.h"

#include <atomic>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/common_runtime/pending_counts.h"
#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/common_runtime/simple_placer.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/edgeset.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_partition.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/graph/validate.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

namespace tensorflow {

// namespace {

DebugExecutorImpl::DebugExecutorImpl(const LocalExecutorParams& p,
                                     const Graph* g)
    : debugger_notification(), node_value_store(), node_ref_store(),
      params_(p), graph_(g), initial_pending_counts_(graph_->num_node_ids()),
      thread_pool_(), break_at_node(), injected_tensors() {
  CHECK(p.create_kernel != nullptr);
  CHECK(p.delete_kernel != nullptr);

  debugger_notification.reset(new MultiUseNotification());
  thread_pool_.reset(new thread::ThreadPool(Env::Default(), "Debugger", 1));
}

DebugExecutorImpl::~DebugExecutorImpl() {
  for (int i = 0; i < graph_->num_node_ids(); i++) {
    params_.delete_kernel(nodes_[i].kernel);
  }
  delete[] nodes_;
  delete graph_;
}

// Helper functions
const Node* DebugExecutorImpl::NodeName2Node(const string& node_name) {
  const Node* the_node = nullptr;
  for (const Node* n : graph_->nodes()) {
    if (n->name() == node_name) {
      the_node = n;
      break;
    }
  }

  return the_node;
}

bool DebugExecutorImpl::NodeName2NodeKernelIsExpensive(const string& node_name) {
  const Node* the_node = NodeName2Node(node_name);
  return nodes_[the_node->id()].kernel_is_expensive;
}

// DEBUG helper function
void DebugPrintQueue(const string& title, const std::deque<string>& queue) {
  std::cout << title << ": [";
  for (const string& item : queue) {
    std::cout << item << ", ";
  }
 std::cout << "]" << std::endl;
}

// Simulation methods for calculating node order
void DebugExecutorImpl::SimProcess(const string& node_name) {
  std::deque<string> ready_queue;
  std::deque<string> inline_ready_queue;

  inline_ready_queue.push_back(node_name);
  while (!inline_ready_queue.empty()) {
    const string curr_node = inline_ready_queue.front();
    inline_ready_queue.pop_front();
    done_nodes.insert(curr_node);

    SimPropagateOutputs(curr_node, &ready_queue);

    node_order.push_back(curr_node);
    // DebugPrintQueue("node_order", node_order);

    SimNodeDone(curr_node, ready_queue, &inline_ready_queue);
  }
}

void DebugExecutorImpl::SimPropagateOutputs(const string& node_name,
                                            std::deque<string>* ready_queue) {
  // Simulates both PropagateOutputs and ActivateNodes

  // DEBUG
  std::cout << "- In SimPropagateOutputs: node_name = " << node_name << std::endl;

  ready_queue->clear();

  const Node* the_node = NodeName2Node(node_name);

  for (const Edge* e : the_node->out_edges()) {
    const Node* dst_node = e->dst();

    // Check if all the input nodes are satisfied
    bool all_inputs_ready = true;
    for (const Edge* edge : dst_node->in_edges()) {
      const string& input_node_name = edge->src()->name();
      // if (std::find(done_nodes.begin(), done_nodes.end(), input_node_name) ==
      //     done_nodes.end()) {
      if (done_nodes.count(input_node_name) == 0) {
        all_inputs_ready = false;
        break;
      }
    }

    if (all_inputs_ready) {
      std::cout << "out_edge: " << the_node->name() << " --> "
                << dst_node->name()
                << "; has " << dst_node->in_edges().size() << " input(s); "
                << "all_inputs_ready = " << all_inputs_ready
                << "; Pushing node " << dst_node->name() << " to ready_queue" << std::endl;  // DEBUG
      ready_queue->push_back(dst_node->name());
    } else {
      std::cout << "out_edge: " << the_node->name() << " --> "
                << dst_node->name()
                << "; has " << dst_node->in_edges().size() << " input(s); "
                << "all_inputs_ready = " << all_inputs_ready 
                << "; Node not ready yet." << std::endl;  // DEBUG
    }

    // getchar();
  }
}

void DebugExecutorImpl::SimNodeDone(const string& node_name,
                                    const std::deque<string>& ready_queue,
                                    std::deque<string>* inline_ready_queue) {
  std::cout << "In SimNodeDone: node_name = " << node_name << std::endl;  // DEBUG

  DebugPrintQueue("ready_queue", ready_queue);
  DebugPrintQueue("inline_ready_queue", *inline_ready_queue);

  // getchar();
  SimScheduleReady(ready_queue, inline_ready_queue);
}

void DebugExecutorImpl::SimScheduleReady(const std::deque<string>& ready_queue,
                                         std::deque<string>* inline_ready_queue) {
  if (ready_queue.empty()) {
    std::cout << "return from SimScheduleReady()" << std::endl;  // DEBUG
    return;
  }

  if (inline_ready_queue == nullptr) {
    // TODO(cais): Simulate
    //     runner_(std::bind(&ME::Process, this, tagged_node, scheduled_usec));
  }

  string curr_expensive_node("");

  for (const string& node_name : ready_queue) {
    bool kernel_is_expensive = NodeName2NodeKernelIsExpensive(node_name);
    // DEBUG
    std::cout << "DEBUG SimScheduleReady: node_name = " << node_name
              << "; kernel_is_expensive = " << kernel_is_expensive << std::endl;

    if (!kernel_is_expensive) { // Assume is_dead = false
      // std::cout << "SimScheduleReady: Pushing inexpensive node "
                // << node_name << std::endl; // DEBUG
      inline_ready_queue->push_back(node_name);
    } else {
      if (!curr_expensive_node.empty()) {
        SimProcess(curr_expensive_node);
      }
      curr_expensive_node = node_name;
    }
  }

  if (!curr_expensive_node.empty()) {
    if (inline_ready_queue->empty()) {
      std::cout << "%% Tail recursion optimization: Pushing expensive node "
                << curr_expensive_node << std::endl; // DEBUG
      inline_ready_queue->push_back(curr_expensive_node);
    } else {
      std::cout << "%% Calling runner_ SimProcess() C, node name = "
                << curr_expensive_node << std::endl; // DEBUG
      SimProcess(curr_expensive_node);
    }
  }

}

void DebugExecutorImpl::SimCalcNodeOrder() {
  // tfdb(cais): Precompute node execution order
  // DEBUG
  // std::cout << "### Precomputing node execution order ###" << std::endl;
  node_order.clear();

  // Calculate node order through simulation methods
  string init_node;
  for (const Node* n : graph_->nodes()) {
    if (n->in_edges().size() == 0) {
      // DEBUG
      init_node = n->name();
      break;
    }
  }

  // DEBUG
  std::cout << "Calling SimProcess with init_node = " << init_node << std::endl;
  // getchar();

  SimProcess(init_node);
}

void DebugExecutorImpl::NonSimCalcNodeOrder() {
  node_order.clear();

  std::deque<string> node_queue;
  std::unordered_set<string> visited_nodes;
  std::unordered_set<string> done_nodes;

  for (const Node* n : graph_->nodes()) {
    if (n->in_edges().size() == 0) {
      // DEBUG
      // std::cout << "Pushing to node_queue: " << n->name() << std::endl;
      node_queue.push_back(n->name());
      visited_nodes.insert(n->name());
    }
  }

  while (!node_queue.empty()) {
  // Pop all the ready nodes from the queue
  while (!node_queue.empty()) {
    const string processed_node = node_queue.front();

    // DEBUG
    // std::cout << "Popping from node_queue: " << processed_node << std::endl;
    node_queue.pop_front();
    node_order.push_back(processed_node);
    visited_nodes.insert(processed_node);
    done_nodes.insert(processed_node);
  }

  for (const Node* n : graph_->nodes()) {
    // Skip visited nodes
    if (visited_nodes.count(n->name()) > 0) {
      continue;
    }

    // Check if all the input nodes are satisfie
    bool all_inputs_ready = true;
    for (const Edge* edge : n->in_edges()) {
      const string& input_node_name = edge->src()->name();
      if (done_nodes.count(input_node_name) == 0) {
        all_inputs_ready = false;
        break;
      }
    }

      if (all_inputs_ready) {
        // DEBUG
        // std::cout << "Pushing to node_queue: " << n->name() << std::endl;
        node_queue.push_back(n->name());
        visited_nodes.insert(n->name());
      }
    }
  }
}


Status DebugExecutorImpl::Initialize() {
  const int num_nodes = graph_->num_node_ids();
  delete[] nodes_;
  nodes_ = new NodeItem[num_nodes];

  Status s;
  total_input_tensors_ = 0;
  total_output_tensors_ = 0;

  InitializePending(graph_, &initial_pending_counts_);

  // Cache this value so we make this virtual function call once, rather
  // that O(# steps * # nodes per step) times.
  device_record_tensor_accesses_ =
      params_.device->RequiresRecordingAccessedTensors();
  
  bool found_nontrivial_control_edges = false;

  // Preprocess every node in the graph to create an instance of op
  // kernel for each node;
  for (const Node* n : graph_->nodes()) {
    const int id = n->id();

    // See if this node is a root node, and if so, add to root_nodes_
    const int num_in_edges = n->in_edges().size();
    if (num_in_edges == 0) {
      root_nodes_.push_back(n);
    }

    // Determine if control edges exist
    for (const Edge* edge : n->in_edges()) {
      if (edge->IsControlEdge()) {
        //DEBUG
        const string& input_node_name = edge->src()->name();

        if (n->name() != "_SINK" && input_node_name != "_SOURCE") {
          std::cout << "Found control edge " << input_node_name
                    << " --> " << n->name() << std::endl;
          found_nontrivial_control_edges = true;
        }
      }
    }

    NodeItem* item = &nodes_[id];
    item->node = n;
    item->num_inputs = n->num_inputs();
    item->num_outputs = n->num_outputs();

    for (int i = 0; i < std::min(4, item->num_inputs); i++) {
      item->inlined_input_type[i] = n->input_type(i);
    }
    for (int i = 0; i < std::min(4, item->num_outputs); i++) {
      item->inlined_output_type[i] = n->output_type(i);
    }

    item->input_start = total_input_tensors_;
    total_input_tensors_ += n->num_inputs();

    item->output_attr_start = total_output_tensors_;
    total_output_tensors_ += n->num_outputs();

    s = params_.create_kernel(n->def(), &item->kernel);
    if (!s.ok()) {
      item->kernel = nullptr;
      s = AttachDef(s, n->def());
      LOG(ERROR) << "Executor failed to create kernel. " << s;
      break;
    }
    CHECK(item->kernel);
    item->kernel_is_expensive = item->kernel->IsExpensive();
    item->kernel_is_async = (item->kernel->AsAsync() != nullptr);
    item->is_merge = IsMerge(n);

    // Initialize static information about the frames in the graph.
    if (IsEnter(n)) {
      string frame_name;
      s = GetNodeAttr(n->def(), "frame_name", &frame_name);
      if (!s.ok()) return s;
      ++frame_input_count_[frame_name];
    }
  }


  // DEBUG
  std::cout << "found_nontrivial_control_edges = "
            << found_nontrivial_control_edges << std::endl;
  // tfdb: Pre-calculate node order
  // TODO(cais): Unified approach to deal with control edges
  if (found_nontrivial_control_edges) {
    NonSimCalcNodeOrder();
  } else {
    SimCalcNodeOrder();
  }


  if (!s.ok()) return s;
  return SetAllocAttrs();
}

// tfdb(cais)
DebuggerResponse DebugExecutorImpl::HandleDebuggerMessage(
  const DebuggerRequest& debugger_request) {
  static const string STEP("step");
  static const string STEP_PREFIX("step ");
  static const string PRINT_PREFIX("print ");
  static const string CONTINUE_PREFIX("continue ");
  static const string WHERE("where");
  static const string INJECT_VALUE_PREFIX("inject_value ");

  DebuggerResponse response;
  response.command = debugger_request.command;  // Record command in response

  // Determind completed nodes and remaining nodes
  std::vector<string> completed_nodes = GetCompletedNodes();
  std::vector<string> not_completed_nodes = GetNotCompletedNodes();

  response.completed_nodes = completed_nodes;
  response.remaining_nodes = not_completed_nodes;
  // std::cout << "response.completed_nodes.size() = "
  //           << response.completed_nodes.size() << std::endl;  // DEBUG
  // std::cout << "response.remaining_nodes.size() = "
  //           << response.remaining_nodes.size() << std::endl;  // DEBUG

  // In response, provide info about whether this debug round is complete
  if (not_completed_nodes.empty()) {
    response.is_completed = true;
  }

  if (debugger_request.command.find(STEP) == 0) {
    // Step once or multiple times

    if (debugger_request.command == STEP) {
      // Step once

      debugger_notification->NotifyOnce();
    } else if (debugger_request.command.find(STEP_PREFIX) == 0) {
      // Step multiple times

      int n_steps = 0;
      bool convert_okay = strings::safe_strto32(
          debugger_request.command.substr(STEP_PREFIX.size()).c_str(),
                                          &n_steps);

      if (convert_okay && n_steps > 0) {
        debugger_notification->Notify(n_steps);
      } else {
        std::cerr << "Syntax error in step command: \""
                  << debugger_request.command << "\"" << std::endl;  // DEBUG
      }
    } else {
      std::cerr << "Syntax error in step command: \""
                << debugger_request.command << "\"" << std::endl;  // DEBUG
    }
  } else if (debugger_request.command.find(PRINT_PREFIX) == 0) {
    // Print the tensor value on a node

    const string& node_name =
        debugger_request.command.substr(PRINT_PREFIX.size());

    for (const Node* n : graph_->nodes()) {
      if (node_name == n->name()) {
        if (node_value_store.count(node_name) == 1) {
          const Tensor& node_val = node_value_store.at(node_name);

          // std::cout << "Found node \"" << node_name
          //           << "\": IsInitialized() = " << node_val.IsInitialized()
          //           << "; value = " << node_val.DebugString() << std::endl;
          // DEBUG

          response.output_tensor = node_val;
          response.has_output_tensor = true;
          // DEBUG
          // std::cout << "Set has_output_tensor to true" << std::endl;

          break;
        } else if (node_ref_store.count(node_name) == 1) {
          const Tensor* node_ref = node_ref_store.at(node_name);

          // DEBUG
          // std::cout << "Found node \"" << node_name
          //           << " through stored reference: " << node_ref << std::endl;

          response.output_tensor = *node_ref;
          response.has_output_tensor = true;

          break;
        }
      }
    }

  } else if (debugger_request.command.find(CONTINUE_PREFIX) == 0) {
    // Continue execution

    const string& node_name =
        debugger_request.command.substr(CONTINUE_PREFIX.size());

    // See if the node is already completed
    bool already_completed = false;
    for (const string& completed_node : completed_nodes) {
      if (completed_node == node_name) {
        already_completed = true;
        break;
      }
    }

    if (already_completed) {
      // DEBUG
      // std::cerr << "ERROR: Node \"" << node_name
      //           << "\" is already completed" << std::endl;
    } else {
      size_t steps_to_go = 0;
      bool found_node = false;

      for (const string& remaining_node : not_completed_nodes) {
        steps_to_go++;
        if (remaining_node == node_name) {
          found_node = true;
          break;
        }
      }

      if (!found_node) {
        // DEBUG
        // std::cerr << "ERROR: Node \"" << node_name
        //           << "\" cannot be found" << std::endl;  // DEBUG
      } else {
        // std::cout << "Steps to go: " << steps_to_go << std::endl;
        debugger_notification->Notify(steps_to_go);
      }
    }

  } else if (debugger_request.command == WHERE) {
    // Get current debugger location: No special action required here

  } else if (debugger_request.command.find(INJECT_VALUE_PREFIX) == 0) {
    // Inject value to the current (just completed) node

    const string& node_name =
        debugger_request.command.substr(INJECT_VALUE_PREFIX.size());

    // std::cout << "inject_value to node \"" << node_name << "\": "
    //           << debugger_request.input_tensor.DebugString()
    //           << "; source address = " << &(debugger_request.input_tensor)
    //           << std::endl;

    if (!node_name.empty()) {
      executor_state->InjectNodeValue(debugger_request.input_tensor);
    } else {
      // DEBUG
      std::cerr << "Invalid node name for inject_value" << std::endl;
    }
  } else if (debugger_request.command.empty()) {
    // NOOP

  } else {
    std::cerr << "Unrecognized command: \""
              << debugger_request.command << "\"" << std::endl;
  }

  return response;
}

void DebugExecutorImpl::RunAsync(const Args& args, DoneCallback done) {
  executor_state = new DebugExecutorState(args, this);

  executor_state->RunAsync(done);

  // std::cout
  //     << "Exiting RunAsync: marking debugger_notification as completed"
  //     << std::endl;
  debugger_notification->MarkAsCompleted();
}

Status DebugExecutorImpl::SetAllocAttrs() {
  Status s;
  Device* device = params_.device;
  DeviceNameUtils::ParsedName local_dev_name = device->parsed_name();

  output_attrs_.resize(total_output_tensors_);
  for (const Node* n : graph_->nodes()) {
    NodeItem* item = &nodes_[n->id()];
    const int base_index = item->output_attr_start;
    // Examine the out edges of each node looking for special use
    // cases that may affect memory allocation attributes.
    for (auto e : n->out_edges()) {
      const int index = e->src_output();
      AllocatorAttributes attr;
      s = InferAllocAttr(n, e->dst(), local_dev_name, &attr);
      if (!s.ok()) return s;
      if (attr.value != 0) {
        if (!e->IsControlEdge()) {
          output_attrs_[base_index + index].Merge(attr);
        }
      }
    }

    for (int out = 0; out < n->num_outputs(); out++) {
      OpKernel* op_kernel = item->kernel;
      DCHECK_LT(out, op_kernel->output_memory_types().size());
      bool on_host = op_kernel->output_memory_types()[out] == HOST_MEMORY;
      AllocatorAttributes h;
      h.set_on_host(on_host);
      output_attrs_[base_index + out].Merge(h);
    }
  }
  return s;
}

Status DebugExecutorImpl::InferAllocAttr(
    const Node* n, const Node* dst,
    const DeviceNameUtils::ParsedName& local_dev_name,
    AllocatorAttributes* attr) {
  Status s;
  // Note that it's possible for *n to be a Recv and *dst to be a Send,
  // so these two cases are not mutually exclusive.
  if (IsRecv(n)) {
    string src_name;
    s = GetNodeAttr(n->def(), "send_device", &src_name);
    if (!s.ok()) return s;
    DeviceNameUtils::ParsedName parsed_src_name;
    if (!DeviceNameUtils::ParseFullName(src_name, &parsed_src_name)) {
      s = errors::Internal("Bad send_device attr '", src_name, "' in node ",
                           n->name());
      return s;
    }
    if (!DeviceNameUtils::IsSameAddressSpace(parsed_src_name, local_dev_name)) {
      // Value is going to be the sink of an RPC.
      attr->set_nic_compatible(true);
      VLOG(2) << "node " << n->name() << " is the sink of an RPC in";
    } else if (local_dev_name.type == "CPU" && parsed_src_name.type == "GPU") {
      // Value is going to be the sink of a local DMA from GPU to CPU.
      attr->set_gpu_compatible(true);
      VLOG(2) << "node " << n->name() << " is the sink of a gpu->cpu copy";
    } else {
      VLOG(2) << "default alloc case local type " << local_dev_name.type
              << " remote type " << parsed_src_name.type;
    }
  }
  if (IsSend(dst)) {
    string dst_name;
    s = GetNodeAttr(dst->def(), "recv_device", &dst_name);
    if (!s.ok()) return s;
    DeviceNameUtils::ParsedName parsed_dst_name;
    if (!DeviceNameUtils::ParseFullName(dst_name, &parsed_dst_name)) {
      s = errors::Internal("Bad recv_device attr '", dst_name, "' in node ",
                           n->name());
      return s;
    }
    if (!DeviceNameUtils::IsSameAddressSpace(parsed_dst_name, local_dev_name)) {
      // Value is going to be the source of an RPC.
      attr->set_nic_compatible(true);
      VLOG(2) << "node " << n->name() << " is the source of an RPC out";
    } else if (local_dev_name.type == "CPU" && parsed_dst_name.type == "GPU") {
      // Value is going to be the source of a local DMA from CPU to GPU.
      attr->set_gpu_compatible(true);
      VLOG(2) << "node " << n->name() << " is the source of a cpu->gpu copy";
    } else {
      VLOG(2) << "default alloc case local type " << local_dev_name.type
              << " remote type " << parsed_dst_name.type;
    }
  } else if (dst->type_string() == "ToFloat") {
    for (auto e : dst->out_edges()) {
      s = InferAllocAttr(n, e->dst(), local_dev_name, attr);
      if (!s.ok()) return s;
    }
  }
  return s;
}

DebugExecutorState::DebugExecutorState(const Executor::Args& args,
                                       DebugExecutorImpl* impl)
    : step_id_(args.step_id),
      rendezvous_(args.rendezvous),
      session_state_(args.session_state),
      tensor_store_(args.tensor_store),
      stats_collector_(args.stats_collector),
      slice_reader_cache_(new checkpoint::TensorSliceReaderCacheWrapper),
      call_frame_(args.call_frame),
      impl_(impl),
      cancellation_manager_(args.cancellation_manager),
      runner_(args.runner),
      num_outstanding_ops_(0) {
  // We start the entire execution in iteration 0 of the root frame
  // so let us create the root frame and the state for iteration 0.
  // Initialize the frame.
  root_frame_ = new FrameState;
  root_frame_->frame_name = "_root";  // assume to be unique
  root_frame_->frame_id = 0;          // must be 0
  root_frame_->num_pending_inputs = 0;
  root_frame_->num_outstanding_iterations = 1;
  root_frame_->max_parallel_iterations = 1;  // enough for root frame
  root_frame_->iterations.resize(root_frame_->max_parallel_iterations);

  VLOG(2) << "Create frame: " << root_frame_->frame_name;

  // Initialize the iteration.
  IterationState* iter_state = new IterationState(impl);
  root_frame_->iterations[0] = iter_state;

  // Initialize the executor state.
  outstanding_frames_.insert({root_frame_->frame_name, root_frame_});
}

DebugExecutorState::~DebugExecutorState() {
  for (auto name_frame : outstanding_frames_) {
    delete name_frame.second;
  }

  for (auto it : device_context_map_) {
    it.second->Unref();
  }

  delete slice_reader_cache_;
}

void DebugExecutorImpl::InitializePending(const Graph* graph,
                                     PendingCounts* counts) {
  for (int id = 0; id < graph->num_node_ids(); id++) {
    // Make sure everything is initialized
    counts->set_initial_count(id, 0, 0);
  }
  for (const Node* n : graph->nodes()) {
    const int id = n->id();
    const int num_in_edges = n->in_edges().size();
    int initial_count;
    if (IsMerge(n)) {
      // merge waits all control inputs so we initialize the pending
      // count to be the number of control edges.
      int32 num_control_edges = 0;
      for (const Edge* edge : n->in_edges()) {
        if (edge->IsControlEdge()) {
          num_control_edges++;
        }
      }
      // Use bit 0 to indicate if we are waiting for a ready live data input.
      initial_count = 1 + (num_control_edges << 1);
    } else {
      initial_count = num_in_edges;
    }
    counts->set_initial_count(id, initial_count, num_in_edges);
  }
}

// tfdb(cais)
std::vector<string> DebugExecutorImpl::GetCompletedNodes() {
  std::vector<string> completed_nodes;

  if (!break_at_node.empty()) {
    for (const string& node_name : node_order) {
      completed_nodes.push_back(node_name);

      if (node_name == break_at_node) {
        break;
      }
    }
  }

  return completed_nodes;
}

std::vector<string> DebugExecutorImpl::GetNotCompletedNodes() {
  std::vector<string> not_completed_nodes;

  // First locate the current break point
  std::deque<string>::const_iterator it = node_order.cbegin();
  if (!break_at_node.empty()) {
    while (it != node_order.cend()) {
      if (*it == break_at_node) {
        it++;
        break;
      } else {
        it++;
      }
    }
  }

  while (it != node_order.cend()) {
      not_completed_nodes.push_back(*it);

      it++;
    }

  return not_completed_nodes;
}

void DebugExecutorState::RunAsync(Executor::DoneCallback done) {
  // tfdb(cais): Create new thread for debugging control: Keyboard for now
  const Graph* graph = impl_->graph_;
  // std::cout << "In RunAsync: graph->num_nodes() = "
  //           << graph->num_nodes() << std::endl;  // DEBUG

  impl_->node_value_store.clear();

  TaggedNodeSeq ready;

  // Ask the device to fill in the device context map.
  Device* device = impl_->params_.device;
  Status fill_status = device->FillContextMap(graph, &device_context_map_);
  if (!fill_status.ok()) {
    done(fill_status);
    return;
  }

  // Initialize the ready queue.
  for (const Node* n : impl_->root_nodes_) {
    // std::cout << "Pushing root node " << n->name()
    //           << " to ready queue" << std::endl;  // DEBUG
    DCHECK_EQ(n->in_edges().size(), 0);
    ready.push_back(TaggedNode{n, root_frame_, 0, false});
  }

  if (ready.empty()) {
    // std::cout << "Ready queue is empty()" << std::endl;  // DEBUG
    done(Status::OK());
  } else {
    num_outstanding_ops_ = ready.size();
    // std::cout << "Ready queue is NOT empty(); num_outstanding_ops_ = "
    //           << num_outstanding_ops_ << std::endl;  // DEBUG
    root_frame_->iterations[0]->outstanding_ops = ready.size();
    done_cb_ = done;
    // Schedule to run all the ready ops in thread pool.
    ScheduleReady(ready, nullptr);
  }
}

// namespace {

// Helpers to make a copy of 'p' and makes a copy of the input type
// vector and the device context vector.
//
// NOTE: We need to make a copy of p.input for asynchronous kernel
// because OpKernelContext methods like input_type(i) needs the param
// points to valid input type vector. It's not an issue for sync
// kernels because the type vector is kept on the stack.
OpKernelContext::Params* CopyParams(const OpKernelContext::Params& p) {
  OpKernelContext::Params* ret = new OpKernelContext::Params;
  *ret = p;
  // Ensure the copy of Params will make a new eigen GPU device if
  // necessary.
  ret->eigen_gpu_device = nullptr;
  ret->inputs = new TensorValueVec(*p.inputs);
  ret->input_device_contexts = new DeviceContextVec(*p.input_device_contexts);
  ret->input_alloc_attrs = new AllocatorAttributeVec(*p.input_alloc_attrs);
  return ret;
}

// Helpers to delete 'p' and copies made by CopyParams.
void DeleteParams(OpKernelContext::Params* p) {
  // No need to delete p->eigen_gpu_device since that is deleted in
  // p's destructor
  delete p->inputs;
  delete p->input_device_contexts;
  delete p->input_alloc_attrs;
  delete p;
}

// }  // end namespace

void DebugExecutorState::Process(TaggedNode tagged_node,
                                 int64 scheduled_usec) {
  // std::cout << "In Process: tagged_node = "
  //           << tagged_node.node->name() << std::endl;

  const NodeItem* nodes = impl_->nodes_;
  TaggedNodeSeq ready;
  std::deque<TaggedNode> inline_ready;

  // Parameters passed to OpKernel::Compute.
  TensorValueVec inputs;
  DeviceContextVec input_device_contexts;
  AllocatorAttributeVec input_alloc_attrs;

  OpKernelContext::Params params;
  params.step_id = step_id_;
  Device* device = impl_->params_.device;
  params.device = device;
  // track allocations if and only if we are collecting statistics
  params.track_allocations = (stats_collector_ != nullptr);
  params.rendezvous = rendezvous_;
  params.session_state = session_state_;
  params.tensor_store = tensor_store_;
  params.cancellation_manager = cancellation_manager_;
  params.call_frame = call_frame_;
  params.function_library = impl_->params_.function_library;
  params.resource_manager = device->resource_manager();
  params.step_resource_manager = &step_resource_manager_;
  params.slice_reader_cache = slice_reader_cache_;
  params.inputs = &inputs;
  params.input_device_contexts = &input_device_contexts;
  params.input_alloc_attrs = &input_alloc_attrs;

  Status s;
  NodeExecStats* stats = nullptr;
  EntryVector outputs;
  bool completed = false;
  inline_ready.push_back(tagged_node);
  while (!inline_ready.empty()) {
    tagged_node = inline_ready.front();
    inline_ready.pop_front();
    const Node* node = tagged_node.node;
    FrameState* input_frame = tagged_node.input_frame;
    int64 input_iter = tagged_node.input_iter;
    const int id = node->id();
    const NodeItem& item = nodes[id];

    // TODO(misard) Replace with a finer-grain enabling flag once we
    // add better optional debugging support.
    if (VLOG_IS_ON(1)) {
      mutex_lock l(mu_);

      IterationState* iter_state = input_frame->GetIteration(input_iter);
      iter_state->mark_started(id);
    }

    // Set the device_context for this node id, if it exists.
    auto dc_it = device_context_map_.find(id);
    if (dc_it != device_context_map_.end()) {
      params.op_device_context = dc_it->second;
    }

    if (stats_collector_) {
      stats = new NodeExecStats;
      stats->set_node_name(node->name());
      nodestats::SetScheduled(stats, scheduled_usec);
      nodestats::SetAllStart(stats);
    }

    VLOG(1) << "Process node: " << id << " step " << params.step_id << " "
            << SummarizeNodeDef(node->def());

    Entry* input_tensors = GetInputTensors(input_frame, input_iter);
    Entry* first_input = input_tensors + item.input_start;
    outputs.clear();
    outputs.resize(item.num_outputs);

    TensorReferenceVector accessed_tensors;
    DeviceContext* device_context = nullptr;
    // Only execute this node if it is not dead or it is a send/recv
    // transfer node. For transfer nodes, we need to propagate the "dead"
    // bit even when the node is dead.
    bool launched_asynchronously = false;
    if (!tagged_node.is_dead || IsTransferNode(node)) {
      // Prepares inputs.
      bool is_input_dead = false;
      s = PrepareInputs(item, first_input, &inputs, &input_device_contexts,
                        &input_alloc_attrs, &is_input_dead);
      if (!s.ok()) {
        // Clear the inputs to maintain the invariant that completed
        // nodes have no valid input tensors.
        int num_inputs = item.num_inputs;
        for (int i = 0; i < num_inputs; ++i) {
          (first_input + i)->val = *kEmptyTensor;
        }
        // TODO(misard) Replace with a finer-grain enabling flag once we
        // add better optional debugging support.
        if (VLOG_IS_ON(1)) {
          mutex_lock l(mu_);
          IterationState* iter_state = input_frame->GetIteration(input_iter);
          iter_state->mark_completed(id);
        }
        // Continue to process the nodes in 'inline_ready'.
        completed = NodeDone(s, item.node, ready, stats, &inline_ready);
        continue;
      }

      // Set up compute params.
      OpKernel* op_kernel = item.kernel;
      params.op_kernel = op_kernel;
      params.frame_iter = FrameAndIter(input_frame->frame_id, input_iter);
      params.is_input_dead = is_input_dead;
      params.output_attr_array =
          gtl::vector_as_array(&impl_->output_attrs_) + item.output_attr_start;

      if (item.kernel_is_async) {
        // std::cout << "  kernel is async" << std::endl; // DEBUG
        // Asynchronous computes.
        AsyncOpKernel* async = item.kernel->AsAsync();
        DCHECK(async != nullptr);
        launched_asynchronously = true;
        auto pcopy = CopyParams(params);
        auto ctx = new OpKernelContext(pcopy, item.num_outputs);
        auto done = [this, tagged_node, item, first_input, ctx, stats, pcopy,
                     device]() {
          VLOG(2) << this << " Async kernel done: "
                  << SummarizeNodeDef(item.node->def());
          if (stats_collector_) nodestats::SetOpEnd(stats);
          EntryVector outputs;
          Status s = ProcessOutputs(item, ctx, &outputs, stats);
          if (stats_collector_) nodestats::SetMemory(stats, ctx);
          // Clears inputs.
          int num_inputs = item.num_inputs;
          for (int i = 0; i < num_inputs; ++i) {
            (first_input + i)->val = *kEmptyTensor;
          }
          // TODO(misard) Replace with a finer-grain enabling flag once we
          // add better optional debugging support.
          if (VLOG_IS_ON(1)) {
            mutex_lock l(mu_);
            tagged_node.input_frame->GetIteration(tagged_node.input_iter)
                ->mark_completed(tagged_node.node->id());
          }
          TaggedNodeSeq ready;
          if (s.ok()) {
            PropagateOutputs(tagged_node, outputs, &ready);
          }
          outputs.clear();
          if (s.ok() && pcopy->device->RequiresRecordingAccessedTensors()) {
            // Get the list of all tensors accessed during the execution
            TensorReferenceVector accessed;
            ctx->retrieve_accessed_tensors(&accessed);
            if (stats_collector_)
              nodestats::SetReferencedTensors(stats, accessed);
            // callee takes ownership of the vector
            device->ConsumeListOfAccessedTensors(ctx->op_device_context(),
                                                 accessed);
          }
          bool completed = NodeDone(s, item.node, ready, stats, nullptr);
          delete ctx;
          DeleteParams(pcopy);
          if (completed) Finish();
        };
        if (stats_collector_) nodestats::SetOpStart(stats);
        device->ComputeAsync(async, ctx, done);
      } else {
        // std::cout << "  kernel is sync: "
        //           << op_kernel->name() << std::endl; // DEBUG
        // Synchronous computes.
        OpKernelContext ctx(&params, item.num_outputs);
        if (stats_collector_) nodestats::SetOpStart(stats);

        device->Compute(CHECK_NOTNULL(op_kernel), &ctx);
        // The final node in the step is always a Sink node. Block
        // this Op from completing until the device has finished all
        // queued operations. For devices like GPUs that continue to
        // execute Ops after their Compute methods have completed,
        // this ensures that control is not returned to the user until
        // the step (and its side-effects) has actually completed.
        if (node->IsSink() && ctx.status().ok()) {
          ctx.SetStatus(device->Sync());
        }
        if (stats_collector_) nodestats::SetOpEnd(stats);

        s = ProcessOutputs(item, &ctx, &outputs, stats);
        if (s.ok() && impl_->device_record_tensor_accesses_) {
          // Get the list of all tensors accessed during the execution
          ctx.retrieve_accessed_tensors(&accessed_tensors);
          device_context = ctx.op_device_context();
        }
        if (stats_collector_) nodestats::SetMemory(stats, &ctx);
      }
    }

    if (!launched_asynchronously) {
      // std::cout << "  Process: Launched synchronously" << std::endl; // DEBUG
      // Clears inputs.
      const int num_inputs = item.num_inputs;
      for (int i = 0; i < num_inputs; ++i) {
        (first_input + i)->val = *kEmptyTensor;
      }
      // TODO(misard) Replace with a finer-grain enabling flag once we
      // add better optional debugging support.
      if (VLOG_IS_ON(1)) {
        mutex_lock l(mu_);
        IterationState* iter_state = input_frame->GetIteration(input_iter);
        iter_state->mark_completed(id);
      }
      // Propagates outputs.
      if (s.ok()) {
        // std::cout << "  Process: Calling PropagateOutputs(): "
        //           << "ready.size() = " << ready.size() << std::endl;
        PropagateOutputs(tagged_node, outputs, &ready);
        // std::cout << "  Process: DONE calling PropagateOutputs(): "
        //           << "ready.size() = " << ready.size() << std::endl;
      }
      outputs.clear();
      if (!accessed_tensors.empty()) {
        if (stats_collector_)
          nodestats::SetReferencedTensors(stats, accessed_tensors);
        // device_context is set above in synchronous computes
        device->ConsumeListOfAccessedTensors(device_context,
                                             accessed_tensors);
      }
      if (stats_collector_) {
        scheduled_usec = nodestats::NowInUsec();
      }
      // Postprocess.
      completed = NodeDone(s, item.node, ready, stats, &inline_ready);
    }
  }  // while !inline_ready.empty()

  // This thread of computation is done if completed = true.
  if (completed) Finish();
}

Status DebugExecutorState::PrepareInputs(
    const NodeItem& item,
    Entry* first_input,
    TensorValueVec* inputs,
    DeviceContextVec* input_device_contexts,
    AllocatorAttributeVec* input_alloc_attrs,
    bool* is_input_dead) {
  const Node* node = item.node;

  inputs->clear();
  inputs->resize(item.num_inputs);
  input_device_contexts->clear();
  input_device_contexts->resize(item.num_inputs);
  input_alloc_attrs->clear();
  input_alloc_attrs->resize(item.num_inputs);

  *is_input_dead = false;

  bool is_merge = item.is_merge;
  for (int i = 0; i < item.num_inputs; ++i) {
    const bool expect_ref = IsRefType(item.input_type(i));
    Entry* entry = first_input + i;
    (*input_device_contexts)[i] = entry->device_context;
    (*input_alloc_attrs)[i] = entry->alloc_attr;

    // i-th input.
    TensorValue* inp = &(*inputs)[i];

    // Only merge and transfer nodes can have no-value inputs.
    if (!entry->has_value) {
      if (!is_merge) {
        DCHECK(IsTransferNode(node));
        inp->tensor = &entry->val;
        *is_input_dead = true;
      }
      continue;
    }
    if (entry->ref == nullptr) {
      if (expect_ref) {
        return AttachDef(
            errors::InvalidArgument(i, "-th input expects a ref type"),
            item.kernel->def());
      }
      inp->tensor = &entry->val;
    } else {
      if (!entry->ref->IsInitialized() && !IsInitializationOp(item.node)) {
        return AttachDef(
            errors::FailedPrecondition("Attempting to use uninitialized value ",
                                       item.kernel->def().input(i)),
            item.kernel->def());
      }
      if (expect_ref) {
        inp->mutex_if_ref = entry->ref_mu;
        inp->tensor = entry->ref;
      } else {
        // Automatically deref the tensor ref when the op expects a
        // tensor but is given a ref to a tensor.  Need to deref it
        // under the mutex.
        {
          mutex_lock l(*(entry->ref_mu));
          entry->val = *entry->ref;
        }
        inp->tensor = &entry->val;
      }
    }
  }
  return Status::OK();
}

Status DebugExecutorState::ProcessOutputs(const NodeItem& item,
                                          OpKernelContext* ctx,
                                          EntryVector* outputs,
                                          NodeExecStats* stats) {
  const Node* node = item.node;
  outputs->clear();
  outputs->resize(item.num_outputs);

  Status s = ctx->status();
  if (!s.ok()) {
    s = AttachDef(s, item.kernel->def());
    // TODO(misard) Replace with a finer-grain enabling flag once we
    // add better optional debugging support.
    if (VLOG_IS_ON(1)) {
      LOG(WARNING) << this << " Compute status: " << s;
      DumpState();
    }
    return s;
  }

  // Get the device_context for this node id, if it exists.
  DeviceContext* device_context = nullptr;
  auto dc_it = device_context_map_.find(node->id());
  if (dc_it != device_context_map_.end()) {
    device_context = dc_it->second;
  }

  for (int i = 0; i < item.num_outputs; ++i) {
    TensorValue val = ctx->release_output(i);
    if (*ctx->is_output_dead() || val.tensor == nullptr) {
      // Unless it's a Switch or a Recv, the node must produce a
      // tensor value at i-th output.
      if (!IsSwitch(node) && !IsRecv(node)) {
        s.Update(errors::Internal("Missing ", i, "-th output from ",
                                  SummarizeNodeDef(node->def())));
      }
    } else {
      Entry* out = &((*outputs)[i]);
      out->has_value = true;

      // This value is filled in below if LogMemory::IsEnabled.
      Tensor value_to_log;

      // Set the device context of the output entry.
      out->device_context = device_context;

      // Set the allocator attributes of the output entry.
      out->alloc_attr = ctx->output_alloc_attr(i);

      // Sanity check of output tensor types.
      DataType dtype = val->dtype();
      if (val.is_ref()) dtype = MakeRefType(dtype);
      if (dtype == item.output_type(i)) {
        if (val.is_ref()) {
          out->ref = val.tensor;
          out->ref_mu = val.mutex_if_ref;
          if (LogMemory::IsEnabled()) {
            // Dereference the tensor under the lock.
            mutex_lock l(*out->ref_mu);
            value_to_log = *out->ref;
          }
        } else {
          out->val = *val.tensor;
          if (LogMemory::IsEnabled()) {
            value_to_log = out->val;
          }
        }
        if (stats_collector_ && val.tensor->IsInitialized()) {
          nodestats::SetOutput(stats, i, val.tensor);
        }
      } else {
        s.Update(errors::Internal("Output ", i, " of type ",
                                  DataTypeString(dtype),
                                  " does not match declared output type ",
                                  DataTypeString(item.output_type(i)),
                                  " for node ",
                                  SummarizeNodeDef(node->def())));
      }
      if (LogMemory::IsEnabled()) {
        LogMemory::RecordTensorOutput(ctx->op_kernel().name(), ctx->step_id(),
                                      i, value_to_log);
      }
    }
    if (!val.is_ref()) {
      // If OpKernelContext returns outputs via pass-by-value, we
      // don't need this trouble.
      delete val.tensor;
    }
  }
  return s;
}

void DebugExecutorState::PropagateOutputs(const TaggedNode& tagged_node,
                                          const EntryVector& outputs,
                                          TaggedNodeSeq* ready) {
  FrameState* input_frame = tagged_node.input_frame;
  int64 input_iter = tagged_node.input_iter;

  // Propagates outputs along out edges, and puts newly ready nodes
  // into the ready queue.
  ready->clear();
  {
    FrameState* output_frame = input_frame;
    int64 output_iter = input_iter;

    mutex_lock l(mu_);
    // Sets the output_frame and output_iter of node.
    bool maybe_completed = SetOutputFrameIter(
        tagged_node, outputs, &output_frame, &output_iter, ready);
    if (output_frame != nullptr) {
      // Continue to process the out nodes:

      // Store output_frame, output_iter and outputs
      stored_node = tagged_node.node;
      stored_output_frame = output_frame;
      stored_output_iter = output_iter;
      stored_outputs = outputs;

      // std::cout << "--- Calling ActivateNode() with stored_output_frame: "
      //           << stored_output_frame->frame_name
      //           << "; output_iter = " << stored_output_iter
      //           << "; outputs.size() = " << stored_outputs.size()
      //           << std::endl;  // DEBUG
      ActivateNode(tagged_node.node, tagged_node.is_dead, output_frame,
                   output_iter, outputs, ready);
    }

    // At this point, this node is completely done.
    input_frame->GetIteration(input_iter)->outstanding_ops--;
    CleanupFramesIterations(input_frame, input_iter, ready);

    // The execution of a node such as Enter may cause the completion of
    // output_frame:output_iter, so perform cleanup if
    // output_frame:output_iter
    // is indeed completed.
    if (maybe_completed) {
      CleanupFramesIterations(output_frame, output_iter, ready);
    }
  }
}

// tfdb(cais)
void DebugExecutorState::InjectNodeValue(Tensor value) {
  // std::cout << "=== In InjectNodeValue()" << std::endl
  //           << "      Tensor address = " << &value << std::endl
  //           << "      Tensor value = " << value.DebugString()
  //           << std::endl
  //           << "      stored_node->name() = " << stored_node->name()
  //           << std::endl
  //           << "      stored_output_frame->iteration_count = "
  //           << stored_output_frame->iteration_count
  //           << "; stored_output_frame->frame_name = "
  //           << stored_output_frame->frame_name << std::endl
  //           << "      stored_output_iter = " << stored_output_iter
  //           << std::endl
  //           << "      stored_outputs.size() = " << stored_outputs.size()
  //           << std::endl;

  const NodeItem* nodes = impl_->nodes_;
  IterationState* output_iter_state =
      stored_output_frame->GetIteration(stored_output_iter);

  for (const Edge* e : stored_node->out_edges()) {
    const Node* dst_node = e->dst();
    const int dst_id = dst_node->id();
    const int src_slot = e->src_output();
    // std::cout << "      src_slot = " << src_slot << std::endl;  // DEBUG

    bool dst_need_input = !e->IsControlEdge();

    if (dst_need_input) {
      const NodeItem& dst_item = nodes[dst_id];
      const int dst_slot = e->dst_input();
      Entry* input_tensors = output_iter_state->input_tensors;
      int dst_loc = dst_item.input_start + dst_slot;

      if (stored_outputs[src_slot].ref != nullptr) {
        // std::cout << "      Calling operator= on original pointer: "
        //           << stored_outputs[src_slot].ref << " --> "
        //           << &value << std::endl;
        *(stored_outputs[src_slot].ref) = value;
      } else {  // TODO(cais): Is this logic correct?
        // std::cout << "      Injecting values" << std::endl;
        Entry injected_entry;
        injected_entry.val = value;
        injected_entry.ref = &value;
        input_tensors[dst_loc] = injected_entry;
      }

      // TODO(cais): For cases other than the simplest one, the following
      //             fields also need to be updated.
      // injected_entry.ref = stored_outputs[src_slot].ref;
      // injected_entry.ref_mu = stored_outputs[src_slot].ref_mu;
      // injected_entry.has_value = stored_outputs[src_slot].has_value;
      // injected_entry.alloc_attr = stored_outputs[src_slot].alloc_attr;
      // injected_entry.device_context =
      //     stored_outputs[src_slot].device_context;
    }
  }
}

void DebugExecutorState::ActivateNode(const Node* node, const bool is_dead,
                                 FrameState* output_frame, int64 output_iter,
                                 const EntryVector& outputs,
                                 TaggedNodeSeq* ready) {
  const NodeItem* nodes = impl_->nodes_;
  IterationState* output_iter_state = output_frame->GetIteration(output_iter);
  for (const Edge* e : node->out_edges()) {
    const Node* dst_node = e->dst();
    const int dst_id = dst_node->id();
    const int src_slot = e->src_output();

    // tfdb(cais): Record output
    const Node* output_src_node = e->src();
    const string& output_src_node_name = output_src_node->name();

    // std::cout << "  ActivateNode: " << node->name()
    //           << " --> " << dst_node->name()
    //           << "; dst_id = " << dst_id
    //           << "; src_slot = " << src_slot << std::endl;

    bool dst_dead = false;
    bool dst_ready = false;
    // True iff this input for dst is needed. We only set this input for
    // dst if this flag is true. This is needed to make the thread safety
    // analysis happy.
    bool dst_need_input = !e->IsControlEdge();
    if (IsMerge(dst_node)) {
      // std::cout << "  ActivateNode:   IsMerge is true" << std::endl;
      // A merge node is ready if all control inputs have arrived and either
      // a) a live data input becomes available or b) all data inputs are
      // dead.
      // For Merge, pending's LSB is set iff a live data input has arrived.
      if (e->IsControlEdge()) {
        output_iter_state->decrement_pending(dst_id, 2);
        int count = output_iter_state->pending(dst_id);
        dst_dead =
            (output_iter_state->dead_count(dst_id) == dst_node->num_inputs());
        dst_ready = (count == 0) || ((count == 1) && dst_dead);
      } else {
        if (outputs[src_slot].has_value) {
          // This is a live data input.
          int count = output_iter_state->pending(dst_id);
          output_iter_state->mark_live(dst_id);
          // Only the first live edge sets the input and (potentially)
          // triggers execution. The low bit of count is set if and
          // only if no live input has been used yet (mark_live clears
          // it). The node should be started if and only if this is
          // the first live input and there are no pending control
          // edges, i.e. count == 1.
          dst_ready = (count == 1);
          dst_need_input = ((count & 0x1) == 1);
        } else {
          // This is a dead data input.
          output_iter_state->increment_dead_count(dst_id);
          dst_dead =
              (output_iter_state->dead_count(dst_id) == dst_node->num_inputs());
          dst_ready = (output_iter_state->pending(dst_id) == 1) && dst_dead;
          dst_need_input = false;
        }
      }
    } else {
      // std::cout << "  ActivateNode:   IsMerge is false" << std::endl;
      // if (outputs[src_slot].has_value) {
      // // tfdb(cais): Print value
      // DEBUG
      // std::cout << "  ActivateNode:   outputs[src_slot] has value: "
      //           << outputs[src_slot].val.DebugString() << std::endl;
      // }
      // A non-merge node is ready if all its inputs are ready. We wait
      // for all inputs to come in even if we know the node is dead. This
      // ensures that all input tensors get cleaned up.
      if (is_dead || (!e->IsControlEdge() && !outputs[src_slot].has_value)) {
        output_iter_state->increment_dead_count(dst_id);
      }
      dst_dead = output_iter_state->dead_count(dst_id) > 0;
      dst_ready = (output_iter_state->decrement_pending(dst_id, 1) == 0);
    }

    if (dst_need_input) {
      const NodeItem& dst_item = nodes[dst_id];
      const int dst_slot = e->dst_input();
      Entry* input_tensors = output_iter_state->input_tensors;
      int dst_loc = dst_item.input_start + dst_slot;

      // tfdb(cais)
      if (outputs[src_slot].val.IsInitialized()) {
        // Store a copy of the output value
        // std::cout << "  *** Storing output value copy from node "
        //           << output_src_node_name << std::endl;
        Tensor tensor_val_copy(outputs[src_slot].val);

        impl_->node_value_store.insert({output_src_node_name, tensor_val_copy});
      } else if (outputs[src_slot].ref != nullptr) {
        // std::cout << "outputs[src_slot].ref = "
        //           << outputs[src_slot].ref << std::endl;  // DEBUG

        impl_->node_ref_store.insert(
            {output_src_node_name, outputs[src_slot].ref});
      }


      input_tensors[dst_loc] = outputs[src_slot];
    }

    // Add dst to the ready queue if it's ready
    if (dst_ready) {
      dst_dead = dst_dead && !IsControlTrigger(dst_node);
      // std::cout << "  ActivateNode:   *** Pushing node to ready: "
      //           << dst_node->name() << std::endl;
      ready->push_back(
          TaggedNode(dst_node, output_frame, output_iter, dst_dead));
      output_iter_state->outstanding_ops++;
    }
  }
}

void DebugExecutorState::ActivateNexts(FrameState* frame, int64 iter,
                                  TaggedNodeSeq* ready) {
  // Propagate the deferred NextIteration nodes to the new iteration.
  for (auto& node_entry : frame->next_iter_roots) {
    const Node* node = node_entry.first;
    const Entry& entry = node_entry.second;
    const bool is_dead = !entry.has_value;
    ActivateNode(node, is_dead, frame, iter, {entry}, ready);
  }
  frame->next_iter_roots.clear();
}

void DebugExecutorState::ActivateLoopInvs(FrameState* frame, int64 iter,
                                     TaggedNodeSeq* ready) {
  // Propagate loop invariants to the new iteration.
  for (auto& node_entry : frame->inv_values) {
    const Node* node = node_entry.first;
    const Entry& entry = node_entry.second;
    const bool is_dead = !entry.has_value;
    ActivateNode(node, is_dead, frame, iter, {entry}, ready);
  }
}

void DebugExecutorState::AddLoopInv(FrameState* frame, const Node* node,
                               const Entry& entry, TaggedNodeSeq* ready) {
  // Store this value.
  frame->inv_values.push_back({node, entry});

  // Make this value available to all iterations.
  bool is_dead = !entry.has_value;
  for (int i = 1; i <= frame->iteration_count; ++i) {
    ActivateNode(node, is_dead, frame, i, {entry}, ready);
  }
}

bool DebugExecutorState::NodeDone(const Status& s, const Node* node,
                             const TaggedNodeSeq& ready, NodeExecStats* stats,
                             std::deque<TaggedNode>* inline_ready) {
  const string& node_name = node->name();
  // std::cout << "In NodeDone(): node = " << node_name << " (Step ";
  impl_->break_at_node = node_name;

  size_t step_idx = 0;
  while (step_idx < impl_->node_order.size()) {
    if (impl_->node_order[step_idx] == node_name) {
      break;
    }
    step_idx++;
  }

  // if (step_idx >= impl_->node_order.size()) {
  //   std::cout << "?";
  // } else {
  //   std::cout << step_idx + 1;
  // }

  // DEBUG
  // std::cout << " / " << impl_->node_order.size() << ")" << std::endl;

  if (stats_collector_) {
    nodestats::SetAllEnd(stats);
    stats_collector_->UpdateCostModel(stats, impl_->graph_, node);
    if (!SetTimelineLabel(node, stats)) {
      // Only record non-transfer nodes.
      stats_collector_->Save(impl_->params_.device->name(), stats);
    } else {
      delete stats;
    }
  }

  Rendezvous* captured_rendezvous = nullptr;  // Will be set on error.
  if (!s.ok()) {
    // Some error happened. This thread of computation is done.
    mutex_lock l(mu_);
    if (status_.ok()) {
      captured_rendezvous = rendezvous_;
      if (captured_rendezvous) captured_rendezvous->Ref();
      status_ = s;
    }
  }
  if (captured_rendezvous) {
    // If we captured the rendezvous_ pointer, we are in an error condition.
    // Use captured_rendezvous, in case "this" is deleted by another thread.
    TRACEPRINTF("StartAbort: %s", s.ToString().c_str());
    captured_rendezvous->StartAbort(s);
    captured_rendezvous->Unref();
  }

  bool completed = false;
  int ready_size = ready.size();
  if (ready_size == 0 || !s.ok()) {
    completed = (num_outstanding_ops_.fetch_sub(1) == 1);
  } else if (ready_size > 1) {
    num_outstanding_ops_.fetch_add(ready_size - 1, std::memory_order_relaxed);
  }

  // Schedule the ready nodes in 'ready'.
  if (s.ok()) {
    // DEBUG
    // std::cout << "  NodeDone() calling ScheduleReady():" << std::endl;
    // std::cout << "    ready.size() = " << ready.size() << ": [";
    // for (const TaggedNode& t_node : ready) {
    //   std::cout << t_node.node->name() << ", ";
    // }
    // std::cout << "]" << std::endl; // DEBUG

    // std::cout << "    inline_ready->size() = "
    //           << inline_ready->size() << ": [";
    // for (TaggedNode& t_node : *inline_ready) {
    //   std::cout << t_node.node->name() << ", ";
    // }
    // std::cout << "]" << std::endl; // DEBUG

    // tfdb(cais): Wait to proceed
    impl_->debugger_notification->WaitForNotification();

    ScheduleReady(ready, inline_ready);
  }
  return completed;
}

void DebugExecutorState::ProcessInline(
    const std::deque<TaggedNode>& inline_ready) {
  if (inline_ready.empty()) return;
  int64 scheduled_usec = 0;
  if (stats_collector_) {
    scheduled_usec = nodestats::NowInUsec();
  }
  for (auto& tagged_node : inline_ready) {
    Process(tagged_node, scheduled_usec);
  }
}

void DebugExecutorState::ScheduleReady(const TaggedNodeSeq& ready,
                                  std::deque<TaggedNode>* inline_ready) {
  if (ready.empty()) return;

  int64 scheduled_usec = 0;
  if (stats_collector_) {
    scheduled_usec = nodestats::NowInUsec();
  }
  // std::cout << "In ScheduleReady(); scheduled_usec = "
  //           << scheduled_usec << std::endl; // DEBUG

  if (inline_ready == nullptr) {
    // Schedule to run all the ready ops in thread pool.
    for (auto& tagged_node : ready) {
      // std::cout
      //     << "Calling runner_ with Process() A (inline_ready = nullptr): "
      //     << "node: " << tagged_node.node->name() << std::endl; // DEBUG
      runner_(std::bind(&ME::Process, this, tagged_node, scheduled_usec));
    }
    return;
  }

  const NodeItem* nodes = impl_->nodes_;
  const TaggedNode* curr_expensive_node = nullptr;
  for (auto& tagged_node : ready) {
    const NodeItem& item = nodes[tagged_node.node->id()];
    if (tagged_node.is_dead || !item.kernel_is_expensive) {
      // Inline this inexpensive node.
      // std::cout << "Pushing inexpensive node "
      //           << tagged_node.node->name() << std::endl; // DEBUG
      inline_ready->push_back(tagged_node);
    } else {
      if (curr_expensive_node) {
        // Dispatch to another thread since there is plenty of work to
        // do for this thread.
        // std::cout << "Calling runner_ with Process() B"
        //           << std::endl; // DEBUG
        runner_(std::bind(&ME::Process, this, *curr_expensive_node,
                          scheduled_usec));
      }
      curr_expensive_node = &tagged_node;
    }
  }
  if (curr_expensive_node) {
    if (inline_ready->empty()) {
      // Tail recursion optimization
      // std::cout << "Tail recursion: Pushing expensive node "
                // << curr_expensive_node->node->name() << std::endl; // DEBUG
      inline_ready->push_back(*curr_expensive_node);
    } else {
      // There are inline nodes to run already. We dispatch this expensive
      // node to other thread.
      // std::cout << "Calling runner_ with Process() C" << std::endl; // DEBUG
      runner_(
          std::bind(&ME::Process, this, *curr_expensive_node, scheduled_usec));
    }
  }
}

void DebugExecutorState::DumpCompletedNodeState(const int node_id,
                                           const Entry* input_vector) {
  const NodeItem& node_item = impl_->nodes_[node_id];
  const Node& node = *node_item.node;
  LOG(WARNING) << "    Completed Node: " << node.DebugString();
  const int input_base = node_item.input_start;
  for (int i = 0; i < node.num_inputs(); ++i) {
    const Entry& input = input_vector[input_base + i];
    CHECK(!input.val.IsInitialized());
  }
}

void DebugExecutorState::DumpPendingNodeState(
    const int node_id, const Entry* input_vector,
    const bool show_nodes_with_no_ready_inputs) {
  const NodeItem& node_item = impl_->nodes_[node_id];
  const Node& node = *node_item.node;
  const int input_base = node_item.input_start;
  if (!show_nodes_with_no_ready_inputs) {
    bool has_ready_input = false;
    for (int i = 0; i < node.num_inputs(); ++i) {
      const Entry& input = input_vector[input_base + i];
      const Tensor* tensor;
      if (input.ref == nullptr) {
        tensor = &input.val;
      } else {
        tensor = input.ref;
      }
      if (tensor->IsInitialized()) {
        has_ready_input = true;
        break;
      }
    }
    if (!has_ready_input) {
      return;
    }
  }
  LOG(WARNING) << "    Pending Node: " << node.DebugString();
  for (int i = 0; i < node.num_inputs(); ++i) {
    const Entry& input = input_vector[input_base + i];
    const Tensor* tensor;
    if (input.ref == nullptr) {
      tensor = &input.val;
    } else {
      tensor = input.ref;
    }
    if (tensor->IsInitialized()) {
      LOG(WARNING) << "      Input " << i << ": "
                   << strings::StrCat(
                          "Tensor<type: ", DataTypeString(tensor->dtype()),
                          " shape: ", tensor->shape().DebugString(), ">");
    } else {
      LOG(WARNING) << "      Input " << i << ": not present";
    }
  }
}

void DebugExecutorState::DumpActiveNodeState(const int node_id,
                                        const Entry* input_vector) {
  const NodeItem& node_item = impl_->nodes_[node_id];
  const Node& node = *node_item.node;
  LOG(WARNING) << "    Active Node: " << node.DebugString();
  const int input_base = node_item.input_start;
  for (int i = 0; i < node.num_inputs(); ++i) {
    const Entry& input = input_vector[input_base + i];
    const Tensor* tensor;
    if (input.ref == nullptr) {
      tensor = &input.val;
    } else {
      tensor = input.ref;
    }
    if (tensor->IsInitialized()) {
      LOG(WARNING) << "      Input " << i << ": "
                   << strings::StrCat(
                          "Tensor<type: ", DataTypeString(tensor->dtype()),
                          " shape: ", tensor->shape().DebugString(), ">");
    } else {
      LOG(WARNING) << "      Input " << i << ": not present";
    }
  }
}

void DebugExecutorState::DumpIterationState(IterationState* iteration) {
  // Dump any waiting nodes that are holding on to tensors.
  for (int i = 0; i < impl_->graph_->num_node_ids(); ++i) {
    if (iteration->node_state(i) == PendingCounts::PENDING_NOTREADY ||
        iteration->node_state(i) == PendingCounts::PENDING_READY) {
      DumpPendingNodeState(i, iteration->input_tensors, false);
    }
  }
  // Then the active nodes.
  for (int i = 0; i < impl_->graph_->num_node_ids(); ++i) {
    if (iteration->node_state(i) == PendingCounts::STARTED) {
      DumpActiveNodeState(i, iteration->input_tensors);
    }
  }
  // Show all input tensors in use.
  size_t total_bytes = 0;
  for (int i = 0; i < impl_->total_input_tensors_; ++i) {
    const Entry& input = iteration->input_tensors[i];
    const Tensor* tensor;
    if (input.ref == nullptr) {
      tensor = &input.val;
    } else {
      tensor = input.ref;
    }
    if (tensor->IsInitialized()) {
      LOG(WARNING) << "    Input " << i << ": "
                   << strings::StrCat("Tensor<type: ",
                                      DataTypeString(tensor->dtype()),
                                      " shape: ", tensor->shape().DebugString(),
                                      ", bytes: ", tensor->TotalBytes(),
                                      ", hash: ", tensor->BufferHash(), ">");
      total_bytes += tensor->TotalBytes();
    }
  }
  LOG(WARNING) << "    Total bytes " << total_bytes;
}

void DebugExecutorState::DumpState() {
  mutex_lock l(mu_);
  if (!dumped_on_error_) {
    LOG(WARNING) << "Dumping state";
    for (auto& frame : outstanding_frames_) {
      LOG(WARNING) << frame.first;
      FrameState* frame_state = frame.second;
      for (IterationState* iteration : frame_state->iterations) {
        LOG(WARNING) << "  Iteration:";
        DumpIterationState(iteration);
      }
    }
    dumped_on_error_ = true;
  }
}

void DebugExecutorState::Finish() {
  mu_.lock();
  auto status = status_;
  auto done_cb = done_cb_;
  auto runner = runner_;
  mu_.unlock();
  delete this;
  CHECK(done_cb != nullptr);
  runner([done_cb, status]() { done_cb(status); });
}

bool DebugExecutorState::IsFrameDone(FrameState* frame) {
  return (frame->num_pending_inputs == 0 &&
          frame->num_outstanding_iterations == 0);
}

bool DebugExecutorState::IsIterationDone(FrameState* frame, int64 iter) {
  IterationState* iter_state = frame->GetIteration(iter);
  if (iter_state->outstanding_ops == 0 &&
      iter_state->outstanding_frame_count == 0) {
    if (iter == 0) {
      // The enclosing frame has no pending input.
      return frame->num_pending_inputs == 0;
    } else {
      // The preceding iteration is deleted (and therefore done).
      return (frame->GetIteration(iter - 1) == nullptr);
    }
  }
  return false;
}

void DebugExecutorState::FindOrCreateChildFrame(FrameState* frame, int64 iter,
                                           const Node* node,
                                           FrameState** child) {
  // Get the child frame name.
  string enter_name;
  Status s = GetNodeAttr(node->def(), "frame_name", &enter_name);
  CHECK(s.ok()) << s;
  const string child_name = MakeFrameName(frame, iter, enter_name);

  auto it = outstanding_frames_.find(child_name);
  if (it != outstanding_frames_.end()) {
    *child = it->second;
  } else {
    // Need to create a new frame instance.
    VLOG(2) << "Create frame: " << child_name;

    FrameState* temp = new FrameState;
    temp->frame_name = child_name;
    temp->frame_id = Hash64(child_name);
    temp->parent_frame = frame;
    temp->parent_iter = iter;
    s = GetNodeAttr(node->def(), "parallel_iterations",
                    &temp->max_parallel_iterations);
    CHECK(s.ok()) << s;
    // 'iterations' is a fixed-length circular buffer.
    temp->iterations.resize(temp->max_parallel_iterations + 1);
    IterationState* iter_state = new IterationState(impl_);
    temp->iterations[0] = iter_state;

    auto frame_pending = impl_->frame_input_count_.find(enter_name);
    DCHECK(frame_pending != impl_->frame_input_count_.end());
    temp->num_pending_inputs = frame_pending->second;
    temp->num_outstanding_iterations = 1;
    *child = temp;

    frame->GetIteration(iter)->outstanding_frame_count++;
    outstanding_frames_[child_name] = temp;
  }
}

void DebugExecutorState::IncrementIteration(FrameState* frame,
                                       TaggedNodeSeq* ready) {
  frame->iteration_count++;
  int64 next_iter = frame->iteration_count;

  VLOG(2) << "Create iteration: [" << frame->frame_name << ", " << next_iter
          << "]";

  IterationState* iter_state = new IterationState(impl_);
  frame->SetIteration(next_iter, iter_state);
  frame->num_outstanding_iterations++;
  frame->dead_exits.clear();

  // Activate the successors of the deferred roots in the new iteration.
  ActivateNexts(frame, next_iter, ready);

  // Activate the loop invariants in the new iteration.
  ActivateLoopInvs(frame, next_iter, ready);
}

bool DebugExecutorState::SetOutputFrameIter(const TaggedNode& tagged_node,
                                       const EntryVector& outputs,
                                       FrameState** output_frame,
                                       int64* output_iter,
                                       TaggedNodeSeq* ready) {
  const Node* node = tagged_node.node;
  FrameState* input_frame = tagged_node.input_frame;
  int64 input_iter = tagged_node.input_iter;
  bool is_dead = tagged_node.is_dead;
  bool is_enter = IsEnter(node);

  if (is_enter) {
    FindOrCreateChildFrame(input_frame, input_iter, node, output_frame);
    // Propagate if this is a loop invariant.
    bool is_constant;
    Status s = GetNodeAttr(node->def(), "is_constant", &is_constant);
    CHECK(s.ok()) << s;
    if (is_constant) {
      AddLoopInv(*output_frame, node, outputs[0], ready);
    }
    --(*output_frame)->num_pending_inputs;
    *output_iter = 0;
  } else if (IsExit(node)) {
    if (is_dead) {
      // Stop and remember this node if it is a dead exit.
      if (input_iter == input_frame->iteration_count) {
        input_frame->dead_exits.push_back(node);
      }
      *output_frame = nullptr;
    } else {
      *output_frame = input_frame->parent_frame;
      *output_iter = input_frame->parent_iter;
    }
  } else if (IsNextIteration(node)) {
    if (is_dead) {
      // Stop the deadness propagation
      *output_frame = nullptr;
    } else {
      if (input_iter == input_frame->iteration_count &&
          input_frame->num_outstanding_iterations ==
              input_frame->max_parallel_iterations) {
        // Reached the maximum for parallel iterations.
        input_frame->next_iter_roots.push_back({node, outputs[0]});
        *output_frame = nullptr;
      } else {
        // If this is a new iteration, start it.
        if (input_iter == input_frame->iteration_count) {
          IncrementIteration(input_frame, ready);
        }
        *output_iter = input_iter + 1;
      }
    }
  }
  return is_enter;
}

void DebugExecutorState::CleanupFramesIterations(FrameState* frame, int64 iter,
                                            TaggedNodeSeq* ready) {
  int64 curr_iter = iter;
  while (curr_iter <= frame->iteration_count &&
         IsIterationDone(frame, curr_iter)) {
    // Delete the iteration curr_iter
    VLOG(2) << "Delete iteration [" << frame->frame_name << ", " << curr_iter
            << "].";

    delete frame->GetIteration(curr_iter);
    frame->SetIteration(curr_iter, nullptr);
    --frame->num_outstanding_iterations;
    ++curr_iter;

    // If there is a deferred iteration, start it.
    if (frame->next_iter_roots.size() > 0) {
      IncrementIteration(frame, ready);
    }
  }

  if (IsFrameDone(frame)) {
    FrameState* parent_frame = frame->parent_frame;
    int64 parent_iter = frame->parent_iter;

    // Propagate all the dead exits to the parent frame.
    for (const Node* node : frame->dead_exits) {
      auto parent_iter_state = parent_frame->GetIteration(parent_iter);
      for (const Edge* e : node->out_edges()) {
        const Node* dst_node = e->dst();
        const int dst_id = dst_node->id();
        const NodeItem* dst_item = &(impl_->nodes_[dst_id]);

        bool dst_dead = true;
        bool dst_ready = false;
        // We know this is a dead input to dst
        if (dst_item->is_merge) {
          if (e->IsControlEdge()) {
            parent_iter_state->decrement_pending(dst_id, 2);
            int count = parent_iter_state->pending(dst_id);
            dst_dead = (parent_iter_state->dead_count(dst_id) ==
                        dst_node->num_inputs());
            dst_ready = (count == 0) || ((count == 1) && dst_dead);
          } else {
            parent_iter_state->increment_dead_count(dst_id);
            dst_dead = (parent_iter_state->dead_count(dst_id) ==
                        dst_node->num_inputs());
            dst_ready = (parent_iter_state->pending(dst_id) == 1) && dst_dead;
          }
        } else {
          parent_iter_state->increment_dead_count(dst_id);
          dst_ready = (parent_iter_state->decrement_pending(dst_id, 1) == 0);
        }
        if (dst_ready) {
          ready->push_back(
              TaggedNode(dst_node, parent_frame, parent_iter, dst_dead));
          parent_iter_state->outstanding_ops++;
        }
      }
    }

    // Delete the frame
    const string& frame_name = frame->frame_name;
    VLOG(2) << "Delete frame " << frame_name;
    outstanding_frames_.erase(frame_name);
    delete frame;

    // Cleanup recursively
    if (parent_frame != nullptr) {
      parent_frame->GetIteration(parent_iter)->outstanding_frame_count--;
      CleanupFramesIterations(parent_frame, parent_iter, ready);
    }
  }
}





// TODO(vrv): Figure out how to unify the many different functions
// that generate RendezvousKey, since many of them have to be
// consistent with each other.
string GetRendezvousKey(const string& tensor_name,
                        const DeviceAttributes& device_info,
                        const FrameAndIter& frame_iter) {
  return strings::StrCat(device_info.name(), ";",
                         strings::FpToString(device_info.incarnation()), ";",
                         device_info.name(), ";", tensor_name, ";",
                         frame_iter.frame_id, ":", frame_iter.iter_id);
}

// }  // end namespace

std::atomic_int_fast64_t DebugSession::step_id_counter_(1);

// NOTE: On Android with a single device, there is never
// a risk of an OpKernel blocking indefinitely:
//
// 1) No operations do I/O that depends on other simultaneous kernels,
//
// 2) Recv nodes always complete immediately: The inputs are sent into
//    the local rendezvous before we start the executor, so the
//    corresponding recvs will not block.
//
// Based on these assumptions, we can use the same thread pool for
// both "non-blocking" and "blocking" OpKernels on Android.
//
// This may change down the road when we add support for multiple
// devices that run concurrently, in which case we will need to
// revisit this decision.
void DebugSession::SchedClosure(std::function<void()> c) {
  // std::cout << "In SchedClosure" << std::endl; // DEBUG

// TODO(sanjay): Get rid of __ANDROID__ path
#ifdef __ANDROID__
  // On Android, there is no implementation of ThreadPool that takes
  // std::function, only Closure, which we cannot easily convert.
  //
  // Instead, we just run the function in-line, which is currently
  // safe given the reasoning above.
  c();
#else
  c();  // tfdb(cais)
#endif  // __ANDROID__
}

DebugSession::DebugSession(const SessionOptions& options,
                             const DeviceMgr* device_mgr)
    : debug_executor(nullptr),
      options_(options),
      device_mgr_(device_mgr),
      cancellation_manager_(new CancellationManager()),
      operation_timeout_in_ms_(options_.config.operation_timeout_in_ms()) {
  // NOTE(mrry): We do not need to use a unique string for the session
  // handle, because DebugSession owns its devices. This may change
  // in future versions.
  session_handle_ = "debug";
  int devices_added = 0;
  if (options.config.log_device_placement()) {
    const string mapping_str = device_mgr_->DeviceMappingString();
    if (mapping_str.empty()) {
      printf("Device mapping: no known devices.\n");
    } else {
      printf("Device mapping:\n%s", mapping_str.c_str());
    }
    LOG(INFO) << "Device mapping:\n" << mapping_str;
  }
  for (auto d : device_mgr_->ListDevices()) {
    // std::cout << "Adding device: " <<  d->name() << std::endl; // DEBUG
    devices_.push_back(d);
    device_set_.AddDevice(d);
    d->op_segment()->AddHold(session_handle_);

    // The first device added is special: it is the 'client device' (a
    // CPU device) from which we feed and fetch Tensors.
    if (devices_added == 0) {
      device_set_.set_client_device(d);
    }
    ++devices_added;
  }
}

DebugSession::~DebugSession() {
  for (auto& it : partial_runs_) {
    delete it.second;
  }
  for (auto it : executors_) {
    delete it.second;
  }
  for (auto d : device_mgr_->ListDevices()) {
    d->op_segment()->RemoveHold(session_handle_);
  }
  delete cancellation_manager_;

  for (auto it : cost_models_) {
    delete it.second;
  }
}

Status DebugSession::Create(const GraphDef& graph) {
  mutex_lock l(graph_def_lock_);
  if (graph_created_) {
    return errors::AlreadyExists(
        "A Graph has already been created for this session.");
  }
  return ExtendLocked(graph);
}

Status DebugSession::Extend(const GraphDef& graph) {
  mutex_lock l(graph_def_lock_);
  return ExtendLocked(graph);
}

Status DebugSession::ExtendLocked(const GraphDef& graph) {
  // Merge versions
  if (graph_def_.has_versions()) {
    if (graph_def_.versions().producer() != graph.versions().producer()) {
      return errors::InvalidArgument(
          "Can't extend GraphDef at version ", graph_def_.versions().producer(),
          " with graph at version ", graph.versions().producer());
    }
    VersionDef* versions = graph_def_.mutable_versions();
    versions->set_min_consumer(
        std::max(versions->min_consumer(), graph.versions().min_consumer()));
    if (graph.versions().bad_consumers_size()) {
      // Add new bad_consumers that aren't already marked bad.
      //
      // Note: This implementation is quadratic time if there are many calls to
      // ExtendLocked with many bad consumers.  Since this is unlikely, and
      // fixing it would require data structures outside of this routine,
      // quadratic time it is.
      auto* bad_consumers = versions->mutable_bad_consumers();
      const std::unordered_set<int> existing(bad_consumers->begin(),
                                             bad_consumers->end());
      for (const int v : graph.versions().bad_consumers()) {
        if (existing.find(v) == existing.end()) {
          bad_consumers->Add(v);
        }
      }
    }
  } else {
    graph_def_.mutable_versions()->CopyFrom(graph.versions());
  }

  const int node_size_before_merge = graph_def_.node_size();
  graph_def_.MergeFrom(graph);

  FunctionLibraryDefinition fdefs(graph_def_.library());
  // Add default attributes to all new nodes in the graph.
  Status s =
      AddDefaultAttrsToGraphDef(&graph_def_, fdefs, node_size_before_merge);
  if (!s.ok()) {
    // One of the nodes was invalid, return the state of graph_def_
    // to what it was before this function.
    const int nodes_added = graph_def_.node_size() - node_size_before_merge;
    graph_def_.mutable_node()->DeleteSubrange(node_size_before_merge,
                                              nodes_added);
    return s;
  }

  if (graph_def_.versions().producer() >= 5) {
    // Validate the graph: we assume that merging two valid graphs
    // should maintain graph validity.
    TF_RETURN_IF_ERROR(graph::ValidateGraphDef(graph_def_, fdefs));
  }

  graph_created_ = true;  // In case this is first call
  return Status::OK();
}

// TODO(yuanbyu): Simplify by treating Run() as "PRunSetup(); PRun()".
Status DebugSession::Run(const NamedTensorList& inputs,
                          const std::vector<string>& output_names,
                          const std::vector<string>& target_nodes,
                          std::vector<Tensor>* outputs) {
  RunMetadata run_metadata;
  return Run(RunOptions(), inputs, output_names, target_nodes, outputs,
             &run_metadata);
}

Status DebugSession::Run(const RunOptions& run_options,
                          const NamedTensorList& inputs,
                          const std::vector<string>& output_names,
                          const std::vector<string>& target_nodes,
                          std::vector<Tensor>* outputs,
                          RunMetadata* run_metadata) {
  // std::cout << "In Run() 6-arg" << std::endl;  // DEBUG
  {
    mutex_lock l(graph_def_lock_);
    if (!graph_created_) {
      return errors::InvalidArgument(
          "Session was not created with a graph before Run()!");
    }
  }

  // Extract the inputs names for this run of the session.
  std::vector<string> input_tensor_names;
  input_tensor_names.reserve(inputs.size());
  for (const auto& it : inputs) {
    input_tensor_names.push_back(it.first);
  }


  // std::cout << "Calling GetOrCreateExecutors()" << std::endl; // DEBUG
  // Check if we already have an executor for these arguments.
  DebugExecutorsAndKeys* executors_and_keys;
  RunStateArgs run_state_args;
  TF_RETURN_IF_ERROR(GetOrCreateExecutors(input_tensor_names, output_names,
                                          target_nodes, &executors_and_keys,
                                          &run_state_args));

  // Create a run state and start execution.
  RunState run_state(input_tensor_names, output_names);
  run_state.rendez = new IntraProcessRendezvous(device_mgr_.get());

  // Send inputs.
  // std::cout << "Calling SendInputs()" << std::endl; // DEBUG
  TF_RETURN_IF_ERROR(SendInputs(inputs, executors_and_keys, run_state.rendez));

  // Start parallel Executors.
  const int num_executors = executors_and_keys->items.size();
  // std::cout << "num_executors = " << num_executors << std::endl;  // DEBUG
  ExecutorBarrier* barrier = new ExecutorBarrier(
      num_executors, run_state.rendez, [&run_state](const Status& ret) {
        // std::cout << "In barrier StatusCallback" << std::endl;  // DEBUG
        {
          mutex_lock l(run_state.mu_);
          run_state.status.Update(ret);
        }
        run_state.executors_done.Notify();
      });

  Executor::Args args;
  args.step_id = step_id_counter_.fetch_add(1);
  args.rendezvous = run_state.rendez;
  args.cancellation_manager = cancellation_manager_;
  args.runner = [this](Executor::Args::Closure c) {
    SchedClosure(c);
  };
  if (LogMemory::IsEnabled()) {
    LogMemory::RecordStep(args.step_id, run_state_args.handle);
  }

  if (run_options.trace_level() == RunOptions::FULL_TRACE ||
      options_.config.graph_options().build_cost_model()) {
    args.stats_collector = new StepStatsCollector(
        run_metadata->mutable_step_stats(), &cost_models_);
    run_state.collector = args.stats_collector;
  }

  // int executor_idx = 0;
  for (const auto& item : executors_and_keys->items) {
    // std::cout << "Calling RunAsync() from Run(): "
    //           << executor_idx++ << " / "
    //           << num_executors << std::endl; // DEBUG

    debug_executor = item.executor;  // tfdb(cais)
    item.executor->RunAsync(args, barrier->Get());

    // std::cout << "Calling Run()" << std::endl; // DEBUG
    // item.executor->Run(args); // cais IDE
  }

  // WaitForNotification(&run_state, run_options.timeout_in_ms() > 0
  //                                     ? run_options.timeout_in_ms()
  //                                     : operation_timeout_in_ms_);

  {
    mutex_lock l(run_state.mu_);
    TF_RETURN_IF_ERROR(run_state.status);
  }

  // std::cout << "Exiting Run()" << std::endl;  // DEBUG

  // Receive outputs.
  TF_RETURN_IF_ERROR(
      RecvOutputs(output_names, executors_and_keys, &run_state, outputs));
  return Status::OK();
}

Status DebugSession::PRunSetup(const std::vector<string>& input_names,
                                const std::vector<string>& output_names,
                                const std::vector<string>& target_nodes,
                                string* handle) {
  {
    mutex_lock l(graph_def_lock_);
    if (!graph_created_) {
      return errors::InvalidArgument(
          "Session was not created with a graph before PRunSetup()!");
    }
  }

  // Check if we already have an executor for these arguments.
  DebugExecutorsAndKeys* executors_and_keys;
  RunStateArgs run_state_args;
  run_state_args.is_partial_run = true;
  Status s = GetOrCreateExecutors(input_names, output_names, target_nodes,
                                  &executors_and_keys, &run_state_args);
  TF_RETURN_IF_ERROR(s);

  // Create the run state and save it for future PRun calls.
  RunState* run_state = new RunState(input_names, output_names);
  run_state->rendez = new IntraProcessRendezvous(device_mgr_.get());
  {
    mutex_lock l(executor_lock_);
    if (!partial_runs_.insert({run_state_args.handle, run_state}).second) {
      return errors::Internal("The handle ", run_state_args.handle,
                              " created for this partial"
                              " run is not unique.");
    }
  }

  // Start parallel Executors.
  Notification& executors_done = run_state->executors_done;
  Status* run_status = &run_state->status;
  const int num_executors = executors_and_keys->items.size();
  ExecutorBarrier* barrier = new ExecutorBarrier(
      num_executors, run_state->rendez,
      [&executors_done, run_status, this](const Status& ret) {
        if (!ret.ok()) {
          mutex_lock l(executor_lock_);
          *run_status = ret;
        }
        executors_done.Notify();
      });

  Executor::Args args;
  {
    mutex_lock l(mu_);
    args.step_id = name_counter_++;
  }
  args.rendezvous = run_state->rendez;
  args.cancellation_manager = cancellation_manager_;
  args.runner = [this](Executor::Args::Closure c) { SchedClosure(c); };
  if (LogMemory::IsEnabled()) {
    LogMemory::RecordStep(args.step_id, run_state_args.handle);
  }

  if (options_.config.graph_options().build_cost_model()) {
    run_state->collector = new StepStatsCollector(nullptr, &cost_models_);
    args.stats_collector = run_state->collector;
  }

  int executor_idx = 0;
  for (auto& item : executors_and_keys->items) {
    // std::cout << "Calling RunAsync() from PRunSetup(): executor "
    //           << executor_idx++ << " / "
    //           << num_executors << std::endl;  // DEBUG

    DebugExecutorImpl* exec = item.executor;
    debug_executor = exec;

    exec->RunAsync(args, barrier->Get());
  }

  *handle = run_state_args.handle;
  return Status::OK();
}

Status DebugSession::PRun(const string& handle, const NamedTensorList& inputs,
                           const std::vector<string>& output_names,
                           std::vector<Tensor>* outputs) {
  std::vector<string> parts = str_util::Split(handle, ';');
  const string& key = parts[0];
  // Get the executors for this partial run.
  DebugExecutorsAndKeys* executors_and_keys;
  RunState* run_state;
  {
    mutex_lock l(executor_lock_);  // could use reader lock
    auto exc_it = executors_.find(key);
    if (exc_it == executors_.end()) {
      return errors::InvalidArgument(
          "Must run 'setup' before performing partial runs!");
    }
    executors_and_keys = exc_it->second;

    auto prun_it = partial_runs_.find(handle);
    if (prun_it == partial_runs_.end()) {
      return errors::InvalidArgument(
          "Must run 'setup' before performing partial runs!");
    }
    run_state = prun_it->second;

    // Make sure that this is a new set of feeds that are still pending.
    for (const auto& input : inputs) {
      auto it = run_state->pending_inputs.find(input.first);
      if (it == run_state->pending_inputs.end()) {
        return errors::InvalidArgument("The feed ", input.first,
                                       " had already been fed.");
      }
    }
    // Check that this is a new set of fetches that are still pending.
    for (const auto& output : output_names) {
      auto it = run_state->pending_outputs.find(output);
      if (it == run_state->pending_outputs.end()) {
        return errors::InvalidArgument("The fetch ", output,
                                       " had already been fetched.");
      }
    }
  }

  // Check that this new set of fetches can be computed from all the
  // feeds we have supplied.
  TF_RETURN_IF_ERROR(
      CheckFetch(inputs, output_names, executors_and_keys, run_state));

  // Send inputs.
  Status s = SendInputs(inputs, executors_and_keys, run_state->rendez);

  // Receive outputs.
  if (s.ok()) {
    s = RecvOutputs(output_names, executors_and_keys, run_state, outputs);
  }

  // Delete the run state if there is an error or all fetches are done.
  {
    mutex_lock l(executor_lock_);
    bool done = true;
    if (s.ok()) {
      {
        mutex_lock l(run_state->mu_);
        if (!run_state->status.ok()) {
          LOG(WARNING) << "An error unrelated to this prun has been detected. "
                       << run_state->status;
        }
      }
      for (const auto& it : inputs) {
        run_state->pending_inputs.erase(it.first);
      }
      for (const auto& name : output_names) {
        run_state->pending_outputs.erase(name);
      }
      done = (run_state->pending_inputs.size() == 0 &&
              run_state->pending_outputs.size() == 0);
    }
    if (done) {
      WaitForNotification(run_state, operation_timeout_in_ms_);
      partial_runs_.erase(handle);
      delete run_state;
    }
  }
  return s;
}

Status DebugSession::SendInputs(
    const NamedTensorList& inputs,
    const DebugExecutorsAndKeys* executors_and_keys,
    IntraProcessRendezvous* rendez) {
  Status s;
  // Insert the input tensors into the local rendezvous by their
  // rendezvous key.
  for (const auto& input : inputs) {
    auto it = executors_and_keys->input_keys.find(input.first);
    if (it == executors_and_keys->input_keys.end()) {
      return errors::InvalidArgument("'", input.first,
                                     "' is not a pre-defined feed!");
    }
    const string& input_key = it->second;
    s = rendez->Send(input_key, Rendezvous::Args(), input.second, false);
    if (!s.ok()) {
      rendez->StartAbort(s);
      return s;
    }
  }
  return Status::OK();
}

Status DebugSession::RecvOutputs(
    const std::vector<string>& output_names,
    const DebugExecutorsAndKeys* executors_and_keys,
    RunState* run_state,
    std::vector<Tensor>* outputs) {
  Status s;
  if (!output_names.empty()) {
    outputs->resize(output_names.size());
  }

  // Get the outputs from the rendezvous
  for (size_t output_offset = 0; output_offset < output_names.size();
       ++output_offset) {
    const string& output_name = output_names[output_offset];
    auto it = executors_and_keys->output_keys.find(output_name);
    if (it == executors_and_keys->output_keys.end()) {
      return errors::InvalidArgument("'", output_name,
                                     "' was not defined as a fetch"
                                     " target in PRunSetup.");
    }
    const string& output_key = it->second;
    Tensor output_tensor;
    bool is_dead;

    // Fetch data from the Rendezvous.
    IntraProcessRendezvous* rendez = run_state->rendez;
    s = rendez->Recv(output_key, Rendezvous::Args(), &output_tensor, &is_dead);
    if (is_dead && s.ok()) {
      s = errors::InvalidArgument("The tensor returned for ",
                                  output_names[output_offset],
                                  " was not valid.");
    }
    if (!s.ok()) {
      rendez->StartAbort(s);
      outputs->clear();
      return s;
    }

    (*outputs)[output_offset] = output_tensor;
  }
  return Status::OK();
}

Status DebugSession::CheckFetch(
    const NamedTensorList& feeds,
    const std::vector<string>& fetches,
    const DebugExecutorsAndKeys* executors_and_keys,
    const RunState* run_state) {
  const Graph* graph = executors_and_keys->graph;
  const NameNodeMap* name_to_node = executors_and_keys->name_to_node;

  // Build the set of pending feeds that we haven't seen.
  std::unordered_set<TensorId, TensorId::Hasher> pending_feeds;
  {
    mutex_lock l(executor_lock_);
    for (const string& feed : run_state->pending_inputs) {
      TensorId id(ParseTensorName(feed));
      auto it = name_to_node->find(id.first);
      if (it == name_to_node->end()) {
        return errors::NotFound("Feed ", feed, ": not found");
      }
      pending_feeds.insert(id);
    }
  }
  for (const auto& it : feeds) {
    TensorId id(ParseTensorName(it.first));
    pending_feeds.erase(id);
  }

  // Initialize the stack with the fetch nodes.
  std::vector<const Node*> stack;
  for (const string& fetch : fetches) {
    TensorId id(ParseTensorName(fetch));
    auto it = name_to_node->find(id.first);
    if (it == name_to_node->end()) {
      return errors::NotFound("Fetch ", fetch, ": not found");
    }
    stack.push_back(it->second);
  }

  // Any tensor needed for fetches can't be in pending_feeds.
  std::vector<bool> visited(graph->num_node_ids(), false);
  while (!stack.empty()) {
    const Node* n = stack.back();
    stack.pop_back();

    for (const Edge* in_edge : n->in_edges()) {
      const Node* in_node = in_edge->src();
      if (pending_feeds.count({in_node->name(), in_edge->src_output()}) > 0) {
        return errors::InvalidArgument("Fetch ", in_node->name(), ":",
                                       in_edge->src_output(),
                                       " can't be computed from the feeds"
                                       " that have been fed so far.");
      }
      if (!visited[in_node->id()]) {
        visited[in_node->id()] = true;
        stack.push_back(in_node);
      }
    }
  }
  return Status::OK();
}

Status DebugSession::GetOrCreateExecutors(
    gtl::ArraySlice<string> inputs,
    gtl::ArraySlice<string> outputs,
    gtl::ArraySlice<string> target_nodes,
    DebugExecutorsAndKeys** executors_and_keys,
    RunStateArgs* run_state_args) {
  // Sort the inputs and outputs, so we don't create separate
  // executors when a user passes in the same inputs/outputs in
  // different orders.
  //
  // We could consider some other signature instead of sorting that
  // preserves the same property to avoid the sort in the future.
  std::vector<string> inputs_sorted(inputs.begin(), inputs.end());
  std::vector<string> outputs_sorted(outputs.begin(), outputs.end());
  std::vector<string> tn_sorted(target_nodes.begin(), target_nodes.end());
  std::sort(inputs_sorted.begin(), inputs_sorted.end());
  std::sort(outputs_sorted.begin(), outputs_sorted.end());
  std::sort(tn_sorted.begin(), tn_sorted.end());

  const string key = strings::StrCat(str_util::Join(inputs_sorted, ","), "->",
                                     str_util::Join(outputs_sorted, ","), "/",
                                     str_util::Join(tn_sorted, ","));

  // Set the handle.
  {
    mutex_lock l(mu_);
    run_state_args->handle = strings::StrCat(key, ";", name_counter_++);
  }

  // See if we already have the executors for this run.
  {
    mutex_lock l(executor_lock_);  // could use reader lock
    auto it = executors_.find(key);
    if (it != executors_.end()) {
      *executors_and_keys = it->second;
      return Status::OK();
    }
  }

  // The executor_lock_ is intentionally released while executor is
  // being created.

  // std::cout << "Calling CreateGraphs()" << std::endl;
  FunctionLibraryDefinition* fdefs;
  std::unordered_map<string, Graph*> graphs;
  Status s = CreateGraphs(inputs, outputs, target_nodes, &fdefs, &graphs,
                          run_state_args);
  TF_RETURN_IF_ERROR(s);
  // std::cout << "~ Done calling CreateGraphs()" << std::endl;

  std::unique_ptr<DebugExecutorsAndKeys> ek(new DebugExecutorsAndKeys);
  ek->func_defs = fdefs;
  if (run_state_args->is_partial_run) {
    ek->graph = run_state_args->graph;
    ek->name_to_node = new NameNodeMap;
    std::unordered_set<StringPiece, StringPiece::Hasher> names;
    for (const string& input : inputs) {
      TensorId id(ParseTensorName(input));
      names.emplace(id.first);
    }
    for (const string& output : outputs) {
      TensorId id(ParseTensorName(output));
      names.emplace(id.first);
    }
    for (Node* n : run_state_args->graph->nodes()) {
      if (names.count(n->name()) > 0) {
        ek->name_to_node->insert({n->name(), n});
      }
    }
  }
  ek->items.reserve(graphs.size());

  auto runner = [this](Executor::Args::Closure c) { SchedClosure(c); };
  const auto& optimizer_opts =
      options_.config.graph_options().optimizer_options();
  GraphOptimizer optimizer(optimizer_opts);

  // int graph_counter = 0;  // TODO(cais): Remove
  for (auto iter = graphs.begin(); iter != graphs.end(); ++iter) {
    const string& partition_name = iter->first;
    Graph* partition_graph = iter->second;
    const int graph_def_version = partition_graph->versions().producer();

    Device* device;
    s = device_mgr_->LookupDevice(partition_name, &device);
    if (!s.ok()) break;

    ek->items.resize(ek->items.size() + 1);
    auto* item = &(ek->items.back());

    // item->flib = NewFunctionLibraryRuntime(device, runner, graph_def_version,
    //                                        fdefs, optimizer_opts);
    item->flib =
        NewFunctionLibraryRuntime(device_mgr_.get(), device, runner,
                                  graph_def_version, fdefs, optimizer_opts);

    LocalExecutorParams params;
    params.device = device;
    params.function_library = item->flib;
    auto lib = item->flib;
    auto opseg = device->op_segment();
    params.create_kernel = [this, lib, opseg](const NodeDef& ndef,
                                              OpKernel** kernel) {
      // Caches the kernel only if the node is stateful.
      if (!lib->IsStateful(ndef.op())) {
        return lib->CreateKernel(ndef, kernel);
      }
      auto create_fn = [lib, &ndef](OpKernel** kernel) {
        return lib->CreateKernel(ndef, kernel);
      };
      // Kernels created for subgraph nodes need to be cached.  On
      // cache miss, create_fn() is invoked to create a kernel based
      // on the function library here + global op registry.
      return opseg->FindOrCreate(session_handle_, ndef.name(), kernel,
                                 create_fn);
    };
    params.delete_kernel = [lib](OpKernel* kernel) {
      // If the node is stateful, opseg owns it. Otherwise, delete it.
      if (kernel && !lib->IsStateful(kernel->type_string())) {
        delete kernel;
      }
    };

    // tfdb(cais): Disabled optimizer, so that RunAsync needs to run only once.
    // DEBUG
    // std::cout << "Calling optimzer.Optimizer() on graph "
              // << graph_counter << " of " << graphs.size() << std::endl;
    // optimizer.Optimize(lib, device, &partition_graph);
    // DEBUG
    // std::cout << "~ DONE calling optimzer.Optimizer() on graph "
    // << graph_counter++ << " of " << graphs.size() << std::endl;

    // s = ValidateMemoryTypes(DeviceType(device->device_type()),
    //                         partition_graph);
    s = EnsureMemoryTypes(DeviceType(device->device_type()), device->name(),
                          partition_graph);

    if (!s.ok()) {
      break;
    }
    // NewLocalDebugExecutor takes ownership of *partition_graph.
    iter->second = nullptr;
    item->executor = nullptr;

    // s = NewLocalDebugExecutor(params, partition_graph, &item->executor);

    // Status NewLocalDebugExecutor(const LocalExecutorParams& params,
    //                              const Graph* graph,
    //                              Executor** executor) {
    DebugExecutorImpl* impl = new DebugExecutorImpl(params, partition_graph);
    s = impl->Initialize();
    if (s.ok()) {
      *(&item->executor) = impl;
    } else {
      delete impl;
    }

    if (!s.ok()) {
      break;
    }
  }
  if (!s.ok()) {
    gtl::STLDeleteValues(&graphs);
    return s;
  }

  // std::cout << "Computing rendezvous keys..." << std::endl;
  // Compute the rendezvous keys to avoid recomputing them every time.
  //
  // We always use the first device as the device name portion of the
  // key, even if we're feeding another graph.
  for (const string& input : inputs) {
    ek->input_keys[input] = GetRendezvousKey(
        input, device_set_.client_device()->attributes(), FrameAndIter(0, 0));
  }
  for (const string& output : outputs) {
    ek->output_keys[output] = GetRendezvousKey(
        output, device_set_.client_device()->attributes(), FrameAndIter(0, 0));
  }

  // Reacquire the lock, try to insert into the map.
  mutex_lock l(executor_lock_);
  const bool inserted = executors_.insert(std::make_pair(key, ek.get())).second;
  if (!inserted) {
    // Another thread created the entry before us, so delete the
    // one we created and return the already created one.
    auto it = executors_.find(key);
    *executors_and_keys = it->second;
  } else {
    *executors_and_keys = ek.release();
  }

  // std::cout << "~ Exiting GetOrCreateExecutors()" << std::endl;
  return Status::OK();
}

void DebugSession::SaveStatefulNodes(Graph* graph) {
  for (Node* n : graph->nodes()) {
    if (n->op_def().is_stateful()) {
      VLOG(2) << "Saving " << n->DebugString();
      stateful_placements_[n->name()] = n->assigned_device_name();
    }
  }
}

void DebugSession::RestoreStatefulNodes(Graph* graph) {
  for (Node* n : graph->nodes()) {
    if (n->op_def().is_stateful()) {
      auto iter = stateful_placements_.find(n->name());
      if (iter != stateful_placements_.end()) {
        n->set_assigned_device_name(iter->second);
        VLOG(2) << "Restored " << n->DebugString();
      }
    }
  }
}

Status DebugSession::CreateGraphs(gtl::ArraySlice<string> feeds,
                                   gtl::ArraySlice<string> fetches,
                                   gtl::ArraySlice<string> target_nodes,
                                   FunctionLibraryDefinition** func_defs,
                                   std::unordered_map<string, Graph*>* outputs,
                                   RunStateArgs* run_state_args) {
  std::unique_ptr<FunctionLibraryDefinition> fdefs;
  std::unique_ptr<Graph> graph;
  {
    mutex_lock l(graph_def_lock_);
    fdefs.reset(new FunctionLibraryDefinition(graph_def_.library()));
    graph.reset(new Graph(fdefs.get()));
    GraphConstructorOptions opts;
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, graph_def_, graph.get()));
  }

  // Remember the graph in run state if this is a partial run.
  if (run_state_args->is_partial_run) {
    run_state_args->graph = new Graph(fdefs.get());
    CopyGraph(*graph.get(), run_state_args->graph);
  }

  TF_RETURN_IF_ERROR(subgraph::RewriteGraphForExecution(
      graph.get(), feeds, fetches, target_nodes,
      device_set_.client_device()->attributes()));

  // Run the simple placer after rewriting the graph.
  std::unordered_map<string, int32> node_name_to_cost_map;
  for (Node* n : graph->nodes()) {
    node_name_to_cost_map[n->name()] = n->cost_id();
  }
  SimplePlacer placer(graph.get(), &device_set_, &node_name_to_cost_map,
                      &options_);

  {
    mutex_lock l(mu_);
    // Restore stateful nodes.
    RestoreStatefulNodes(graph.get());
    TF_RETURN_IF_ERROR(placer.Run());
    // Save stateful nodes.
    SaveStatefulNodes(graph.get());
  }

  // Partition the graph across devices.
  std::unordered_map<string, GraphDef> partitions;
  PartitionOptions popts;
  popts.node_to_loc = [](const Node* node) {
    return node->assigned_device_name();
  };
  popts.new_name = [this](const string& prefix) {
    mutex_lock l(mu_);
    return strings::StrCat(prefix, "/_", name_counter_++);
  };
  popts.get_incarnation = [](const string& name) {
    // The debug session does not have changing incarnation numbers.
    // Just return '1'.
    return 1;
  };
  popts.control_flow_added = false;
  TF_RETURN_IF_ERROR(Partition(popts, graph.get(), &partitions));

  std::vector<string> device_names;
  for (auto device : devices_) {
    // Extract the LocalName from the device.
    device_names.push_back(DeviceNameUtils::LocalName(device->name()));
  }

  // Check for valid partitions.
  for (const auto& partition : partitions) {
    const string& local_partition_name =
        DeviceNameUtils::LocalName(partition.first);
    if (std::count(device_names.begin(), device_names.end(),
                   local_partition_name) == 0) {
      return errors::InvalidArgument(
          "Creating a partition for ", local_partition_name,
          " which doesn't exist in the list of available devices. Available "
          "devices: ",
          str_util::Join(device_names, ","));
    }
  }

  Status s;
  for (auto&& partition : partitions) {
    const string& partition_name = partition.first;

    GraphDef* graph_def = &partition.second;
    VLOG(2) << "Created " << graph_def->DebugString() << " for "
            << partition_name;

    // Give the device an opportunity to rewrite its subgraph.
    Device* d;
    s = device_mgr_->LookupDevice(partition_name, &d);
    if (!s.ok()) break;
    {
      mutex_lock l(graph_def_lock_);
      // TODO(pbar) The library is currently shared and immutable. There
      // may be possible use cases where a device may want to modify
      // function definitions - in which case the library would need to be
      // replicated per device.
      s = d->MaybeRewriteGraph(graph_def_.library(), graph_def);
      if (!s.ok()) {
        break;
      }
    }
    Graph* device_graph = new Graph(fdefs.get());
    GraphConstructorOptions device_opts;
    // There are internal operations (e.g., send/recv) that we now
    // allow.
    device_opts.allow_internal_ops = true;
    device_opts.expect_device_spec = true;
    Status s = ConvertGraphDefToGraph(device_opts, *graph_def, device_graph);
    if (!s.ok()) {
      delete device_graph;
      break;
    }
    outputs->insert(std::make_pair(partition_name, device_graph));
  }
  if (!s.ok()) {
    // Also delete other graphs created during the loop.
    gtl::STLDeleteValues(outputs);
    return s;
  }
  *func_defs = fdefs.release();
  return Status::OK();
}

::tensorflow::Status DebugSession::Close() {
  cancellation_manager_->StartCancel();
  return ::tensorflow::Status::OK();
}

::tensorflow::DebuggerResponse DebugSession::SendDebugMessage(
    const DebuggerRequest& request) {
  // std::cout << "In DebugSession::SendDebugMessage(): debug_msg = \""
  //           << debug_msg << "\"" << std::endl;  // DEBUG

  // TODO(cais): mutex lock needed?
  {
    mutex_lock l(debug_lock_);

    if (debug_executor != nullptr) {
      return debug_executor->HandleDebuggerMessage(request);
    } else {
      return DebuggerResponse();    // TODO(cais): Throw proper exception.
    }
  }
}

DebugSession::RunState::~RunState() {
  if (rendez != nullptr) {
    if (!executors_done.HasBeenNotified()) {
      rendez->StartAbort(errors::Cancelled("PRun cancellation"));
      executors_done.WaitForNotification();
    }
    rendez->Unref();
  }
  if (collector != nullptr) {
    delete collector;
  }
}

void DebugSession::WaitForNotification(RunState* run_state,
                                        int64 timeout_in_ms) {
  if (timeout_in_ms > 0) {
    bool timed_out =
        run_state->executors_done.WaitForNotificationWithTimeout(
            timeout_in_ms);

    if (timed_out) {
      {
        mutex_lock l(run_state->mu_);
        run_state->status.Update(Status(error::DEADLINE_EXCEEDED,
                                        "Timed out waiting for notification"));
      }
      // TODO(sherrym): This cancels all steps in the session, even ones that
      // have not exceeded their deadline. An alternative would be to use a
      // two-level cancellation manager with a Session-global one containing
      // several step-local ones. Probably the RunState should have its own
      // CancellationManager.
      cancellation_manager_->StartCancel();
    }
  } else {
    run_state->executors_done.WaitForNotification();
  }
}

class DebugSessionFactory : public SessionFactory {
 public:
  DebugSessionFactory() {}

  bool AcceptsOptions(const SessionOptions& options) override {
    return (options.target == "debug");
  }

  Session* NewSession(const SessionOptions& options) override {
    std::vector<Device*> devices;
    DeviceFactory::AddDevices(options, "/job:localhost/replica:0/task:0",
                              &devices);
    return new DebugSession(options, new DeviceMgr(devices));
  }
};

class DebugSessionRegistrar {
 public:
  DebugSessionRegistrar() {
    SessionFactory::Register("DEBUG_SESSION", new DebugSessionFactory());
  }
};
static DebugSessionRegistrar registrar;

}  // namespace tensorflow
