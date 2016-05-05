/* Copyright 2016 Google Inc. All Rights Reserved.

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

DebugExecutorImpl::DebugExecutorImpl(const LocalExecutorParams& p,
                                     const Graph* g)
    : ExecutorImpl(p, g),
      debugger_notification(), node_value_store(), node_ref_store(),
      thread_pool_(), break_at_node(), injected_tensors() {
  debugger_notification.reset(new MultiUseNotification());
  thread_pool_.reset(new thread::ThreadPool(Env::Default(), "Debugger", 1));
}

// Helper functions
const Node* DebugExecutorImpl::NodeName2Node(const string& node_name) const {
  const Node* the_node = nullptr;
  for (const Node* n : graph_->nodes()) {
    if (n->name() == node_name) {
      the_node = n;
      break;
    }
  }

  return the_node;
}

bool DebugExecutorImpl::NodeName2NodeKernelIsExpensive(const string& node_name) const {
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
  // std::cout << "- In SimPropagateOutputs: node_name = " << node_name << std::endl;

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
      // std::cout << "out_edge: " << the_node->name() << " --> "
      //           << dst_node->name()
      //           << "; has " << dst_node->in_edges().size() << " input(s); "
      //           << "all_inputs_ready = " << all_inputs_ready
      //           << "; Pushing node " << dst_node->name()
      //           << " to ready_queue" << std::endl;  // DEBUG
      ready_queue->push_back(dst_node->name());
    } else {
      // std::cout << "out_edge: " << the_node->name() << " --> "
      //           << dst_node->name()
      //           << "; has " << dst_node->in_edges().size() << " input(s); "
      //           << "all_inputs_ready = " << all_inputs_ready 
      //           << "; Node not ready yet." << std::endl;  // DEBUG
    }

    // getchar();
  }
}

void DebugExecutorImpl::SimNodeDone(const string& node_name,
                                    const std::deque<string>& ready_queue,
                                    std::deque<string>* inline_ready_queue) {
  // std::cout << "In SimNodeDone: node_name = " << node_name << std::endl;  // DEBUG

  // DEBUG
  // DebugPrintQueue("ready_queue", ready_queue);
  // DebugPrintQueue("inline_ready_queue", *inline_ready_queue);

  // getchar();
  SimScheduleReady(ready_queue, inline_ready_queue);
}

void DebugExecutorImpl::SimScheduleReady(const std::deque<string>& ready_queue,
                                         std::deque<string>* inline_ready_queue) {
  if (ready_queue.empty()) {
    // std::cout << "return from SimScheduleReady()" << std::endl;  // DEBUG
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
    // std::cout << "DEBUG SimScheduleReady: node_name = " << node_name
    //           << "; kernel_is_expensive = " << kernel_is_expensive << std::endl;

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
      // std::cout << "%% Tail recursion optimization: Pushing expensive node "
      //           << curr_expensive_node << std::endl; // DEBUG
      inline_ready_queue->push_back(curr_expensive_node);
    } else {
      // std::cout << "%% Calling runner_ SimProcess() C, node name = "
      //           << curr_expensive_node << std::endl; // DEBUG
      SimProcess(curr_expensive_node);
    }
  }

}

void DebugExecutorImpl::CalcNodeOrder() {
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
  // std::cout << "Calling SimProcess with init_node = " << init_node << std::endl;
  // getchar();

  SimProcess(init_node);
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

DebugExecutorState::DebugExecutorState(const Executor::Args& args,
                                       DebugExecutorImpl* impl)
    : ExecutorState(args, impl) {
  debug_exec_impl_ = reinterpret_cast<DebugExecutorImpl*>(impl);
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

void DebugExecutorState::RunAsync(DebugExecutorImpl::DoneCallback done) {
  // tfdb(cais): Create new thread for debugging control: Keyboard for now
  const Graph* graph = impl_->graph_;
  // std::cout << "In RunAsync: graph->num_nodes() = "
  //           << graph->num_nodes() << std::endl;  // DEBUG

  debug_exec_impl_->node_value_store.clear();

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

        debug_exec_impl_->node_value_store.insert(
            {output_src_node_name, tensor_val_copy});
      } else if (outputs[src_slot].ref != nullptr) {
        // std::cout << "outputs[src_slot].ref = "
        //           << outputs[src_slot].ref << std::endl;  // DEBUG

        debug_exec_impl_->node_ref_store.insert(
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

bool DebugExecutorState::NodeDone(const Status& s, const Node* node,
                             const TaggedNodeSeq& ready, NodeExecStats* stats,
                             std::deque<TaggedNode>* inline_ready) {
  const string& node_name = node->name();
  // std::cout << "In NodeDone(): node = " << node_name << " (Step ";
  debug_exec_impl_->break_at_node = node_name;

  size_t step_idx = 0;
  while (step_idx < debug_exec_impl_->node_order.size()) {
    if (debug_exec_impl_->node_order[step_idx] == node_name) {
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
    debug_exec_impl_->debugger_notification->WaitForNotification();

    ScheduleReady(ready, inline_ready);
  }
  return completed;
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
  c();  // tfdb: Single-threaded execution
#endif  // __ANDROID__
}

DebugSession::DebugSession(const SessionOptions& options,
                             const DeviceMgr* device_mgr)
    : DirectSession(options, device_mgr), debug_executor(nullptr) {
  // TODO(cais): Remove inherited thread_pool_ if it will not ever be used.

  session_handle_ = "debug";
  InitializeDeviceManager();
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
  ExecutorsAndKeys* executors_and_keys;
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
  args.session_state = &session_state_;
  args.tensor_store = &run_state.tensor_store;

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

  // Save the output tensors of this run we choose to keep.
  TF_RETURN_IF_ERROR(
      run_state.tensor_store.SaveTensors(output_names, &session_state_));

  return Status::OK();
}

Status DebugSession::GetOrCreateExecutors(
    gtl::ArraySlice<string> inputs,
    gtl::ArraySlice<string> outputs,
    gtl::ArraySlice<string> target_nodes,
    ExecutorsAndKeys** executors_and_keys,
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

  std::unique_ptr<ExecutorsAndKeys> ek(new ExecutorsAndKeys);
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

    // Pre-calculate node execution order
    impl->CalcNodeOrder();

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

::tensorflow::DebuggerResponse DebugSession::SendDebugMessage(
    const DebuggerRequest& request) {
  // std::cout << "In DebugSession::SendDebugMessage(): debug_msg = \""
  //           << debug_msg << "\"" << std::endl;  // DEBUG

  // TODO(cais): mutex lock needed?
  {
    mutex_lock l(debug_lock_);

    if (debug_executor != nullptr) {
      DebugExecutorImpl* debug_exec_impl
          = reinterpret_cast<DebugExecutorImpl*>(debug_executor);
      return debug_exec_impl->HandleDebuggerMessage(request);
    } else {
      return DebuggerResponse();    // TODO(cais): Throw proper exception.
    }
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
