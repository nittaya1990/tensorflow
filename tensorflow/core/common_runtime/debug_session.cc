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

#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

DebugExecutorImpl::DebugExecutorImpl(const LocalExecutorParams& p,
                                     const Graph* g)
    : ExecutorImpl(p, g),
      debug_notification(),
      exec_notification(),
      node_value_store(),
      node_ref_store(),
      thread_pool_(),
      break_at_node(),
      injected_tensors(),
      completed_debug_requests(0) {
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

bool DebugExecutorImpl::NodeName2NodeKernelIsExpensive(
    const string& node_name) const {
  const Node* the_node = NodeName2Node(node_name);
  return nodes[the_node->id()].kernel_is_expensive;
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

    SimNodeDone(curr_node, ready_queue, &inline_ready_queue);
  }
}

void DebugExecutorImpl::SimPropagateOutputs(const string& node_name,
                                            std::deque<string>* ready_queue) {
  // Simulates both PropagateOutputs and ActivateNodes
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
      ready_queue->push_back(dst_node->name());
    }
  }
}

void DebugExecutorImpl::SimNodeDone(const string& node_name,
                                    const std::deque<string>& ready_queue,
                                    std::deque<string>* inline_ready_queue) {
  SimScheduleReady(ready_queue, inline_ready_queue);
}

void DebugExecutorImpl::SimScheduleReady(
    const std::deque<string>& ready_queue,
    std::deque<string>* inline_ready_queue) {
  if (ready_queue.empty()) {
    return;
  }

  // TODO(cais): Simulate inline_ready_queue == nullptr

  string curr_expensive_node("");

  for (const string& node_name : ready_queue) {
    bool kernel_is_expensive = NodeName2NodeKernelIsExpensive(node_name);
    if (!kernel_is_expensive) {  // Assume is_dead = false
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
      inline_ready_queue->push_back(curr_expensive_node);
    } else {
      SimProcess(curr_expensive_node);
    }
  }
}

void DebugExecutorImpl::CalcNodeOrder() {
  node_order.clear();

  // Calculate node order through simulation methods
  string init_node;
  for (const Node* n : graph_->nodes()) {
    if (n->in_edges().size() == 0) {
      init_node = n->name();
      break;
    }
  }

  SimProcess(init_node);
}

// tfdb: Handle debugger message
DebuggerResponse DebugExecutorImpl::HandleDebuggerMessage(
    const DebuggerRequest& debugger_request) {
  // TODO(cais): Replace with string constants in debugger.h
  static const string STEP("step");
  static const string PRINT_PREFIX("inspect_value ");
  static const string WHERE("where");
  static const string INJECT_VALUE_PREFIX("inject_value ");

  // If the first node ("_SOURCE") has not finished yet, wait.
  if (completed_debug_requests == 0) {
    debug_notification->WaitForNotification();
  }

  DebuggerResponse response;
  response.command = debugger_request.command;  // Record command in response

  // Determind completed nodes and remaining nodes
  std::vector<string> completed_nodes = GetCompletedNodes();
  std::vector<string> not_completed_nodes = GetNotCompletedNodes();

  response.completed_nodes = completed_nodes;
  response.remaining_nodes = not_completed_nodes;

  // In response, provide info about whether this debug round is complete
  if (not_completed_nodes.empty()) {
    response.is_completed = true;
  }

  if (debugger_request.command == STEP) {
    // Step once
    exec_notification->NotifyOnce();
    if (!response.is_completed) {
      debug_notification->WaitForNotification();
    }
  } else if (debugger_request.command.find(PRINT_PREFIX) == 0) {
    // Print the tensor value on a node

    const string& node_name =
        debugger_request.command.substr(PRINT_PREFIX.size());

    for (const Node* n : graph_->nodes()) {
      if (node_name == n->name()) {
        if (node_value_store.count(node_name) == 1) {
          const Tensor& node_val = node_value_store.at(node_name);
          response.output_tensor = node_val;
          response.has_output_tensor = true;

          break;
        } else if (node_ref_store.count(node_name) == 1) {
          const Tensor* node_ref = node_ref_store.at(node_name);
          response.output_tensor = *node_ref;
          response.has_output_tensor = true;

          break;
        }
      }
    }

  } else if (debugger_request.command == WHERE) {
    // Get current debugger location: No special action required here

  } else if (debugger_request.command.find(INJECT_VALUE_PREFIX) == 0) {
    // Inject value to the current (just completed) node

    const string& node_name =
        debugger_request.command.substr(INJECT_VALUE_PREFIX.size());

    if (!node_name.empty()) {
      executor_state->InjectNodeValue(debugger_request.input_tensor);
    } else {
      std::cerr << "Invalid node name for inject_value" << std::endl;
    }
  } else if (debugger_request.command.empty()) {
    // NOOP

  } else {
    std::cerr << "Unrecognized command: \"" << debugger_request.command << "\""
              << std::endl;
  }

  completed_debug_requests++;
  return response;
}

void DebugExecutorImpl::RunAsync(const Args& args, DoneCallback done) {
  // Create new notification objects for the execution and the debugger,
  // respectively.
  exec_notification.reset(new MultiUseNotification());
  debug_notification.reset(new MultiUseNotification());

  // Reset the counter for completed debug requests in the current round
  // (i.e., Run() call)
  completed_debug_requests = 0;

  executor_state = new DebugExecutorState(args, this);

  executor_state->RunAsync(done);

  exec_notification->MarkAsCompleted();
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

void DebugExecutorState::PreRunAsync(Executor::DoneCallback done) {
  debug_exec_impl_->node_value_store.clear();
}

// tfdb: Inject a new Tensor value into the current node.
void DebugExecutorState::InjectNodeValue(Tensor value) {
  const NodeItem* nodes = impl_->nodes;
  IterationState* output_iter_state =
      stored_output_frame->GetIteration(stored_output_iter);

  for (const Edge* e : stored_node->out_edges()) {
    const Node* dst_node = e->dst();
    const int dst_id = dst_node->id();
    const int src_slot = e->src_output();

    bool dst_need_input = !e->IsControlEdge();

    if (dst_need_input) {
      const NodeItem& dst_item = nodes[dst_id];
      const int dst_slot = e->dst_input();
      Entry* input_tensors = output_iter_state->input_tensors;
      int dst_loc = dst_item.input_start + dst_slot;

      if (stored_outputs[src_slot].ref != nullptr) {
        // Inject value through the reference
        *(stored_outputs[src_slot].ref) = value;
      } else {
        // Inject new value to the input tensor
        input_tensors[dst_loc].val = value;
      }
    }
  }
}

void DebugExecutorState::ActivateNode(const Node* node, const bool is_dead,
                                      FrameState* output_frame,
                                      int64 output_iter,
                                      const EntryVector& outputs,
                                      TaggedNodeSeq* ready) {
  // Store output_frame, output_iter and outputs
  stored_node = node;
  stored_output_frame = output_frame;
  stored_output_iter = output_iter;
  stored_outputs = outputs;

  const NodeItem* nodes = impl_->nodes;
  IterationState* output_iter_state = output_frame->GetIteration(output_iter);
  for (const Edge* e : node->out_edges()) {
    const Node* dst_node = e->dst();
    const int dst_id = dst_node->id();
    const int src_slot = e->src_output();

    // tfdb(cais): Record output
    const Node* output_src_node = e->src();
    const string& output_src_node_name = output_src_node->name();

    bool dst_dead = false;
    bool dst_ready = false;
    // True iff this input for dst is needed. We only set this input for
    // dst if this flag is true. This is needed to make the thread safety
    // analysis happy.
    bool dst_need_input = !e->IsControlEdge();
    if (IsMerge(dst_node)) {
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

      // Debugger: supply output tensor for inspect_value
      if (outputs[src_slot].val.IsInitialized()) {
        // Store a copy of the output value
        Tensor tensor_val_copy(outputs[src_slot].val);

        debug_exec_impl_->node_value_store.insert(
            {output_src_node_name, tensor_val_copy});
      } else if (outputs[src_slot].ref != nullptr) {
        // Store a copy of the ref to the Tensor value
        debug_exec_impl_->node_ref_store.insert(
            {output_src_node_name, outputs[src_slot].ref});
      }

      input_tensors[dst_loc] = outputs[src_slot];
    }

    // Add dst to the ready queue if it's ready
    if (dst_ready) {
      dst_dead = dst_dead && !IsControlTrigger(dst_node);
      ready->push_back(
          TaggedNode(dst_node, output_frame, output_iter, dst_dead));
      output_iter_state->outstanding_ops++;
    }
  }
}

void DebugExecutorState::NodeDoneEarlyHook(const Node* node) {
  // Supply information about at which node the debugger is at.
  debug_exec_impl_->break_at_node = node->name();

  // Notify the debug thread that a new node has just finished executing.
  debug_exec_impl_->debug_notification->NotifyOnce();
}

void DebugExecutorState::NodeDoneLateHook(const Node* node) {
  // This thread (execution thread) waits for notification from the debugger
  // thread (e.g., caused by a step command) before proceeding to the next
  // node.
  debug_exec_impl_->exec_notification->WaitForNotification();
}

// TODO(cais): Dedupe with direct_session.cc
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
// TODO(sanjay): Get rid of __ANDROID__ path
#ifdef __ANDROID__
  // On Android, there is no implementation of ThreadPool that takes
  // std::function, only Closure, which we cannot easily convert.
  //
  // Instead, we just run the function in-line, which is currently
  // safe given the reasoning above.
  c();
#else
  c();  // tfdb Debugger: Single-threaded execution for now
#endif  // __ANDROID__
}

DebugSession::DebugSession(const SessionOptions& options,
                           const DeviceMgr* device_mgr)
    : DirectSession(options, device_mgr),
      debug_executor(nullptr),
      debug_init_notif(nullptr) {
  // TODO(cais): Remove inherited thread_pool_ if it will never be used.

  // Debug sessions will not optimize graphs
  optimize_graphs_ = false;

  session_handle_ = "debug";
  InitializeDeviceManager();
}

void DebugSession::WaitForNotification(RunState* run_state,
                                       int64 timeout_in_ms) {
  // tfdb: Do nothing here.
}

Status DebugSession::GetOrCreateExecutors(gtl::ArraySlice<string> inputs,
                                          gtl::ArraySlice<string> outputs,
                                          gtl::ArraySlice<string> target_nodes,
                                          ExecutorsAndKeys** executors_and_keys,
                                          RunStateArgs* run_state_args) {
  // Invoke the parent version of the method
  Status s = DirectSession::GetOrCreateExecutors(
      inputs, outputs, target_nodes, executors_and_keys, run_state_args);

  if (s.ok()) {
    // tfdb: Register the DebugExecutorImpl instance
    debug_executor = (*executors_and_keys)->items[0].executor;
  }

  return s;
}

Status DebugSession::CreateLocalExecutor(const LocalExecutorParams& params,
                                         const Graph* graph,
                                         Executor** executor) {
  DebugExecutorImpl* impl = new DebugExecutorImpl(params, graph);
  Status s = impl->Initialize();

  if (s.ok()) {
    // Pre-calculate node execution order
    impl->CalcNodeOrder();

    *executor = impl;
    debug_executor = impl;
  } else {
    delete impl;
  }

  return s;
}

Status DebugSession::Run(const NamedTensorList& inputs,
                         const std::vector<string>& output_names,
                         const std::vector<string>& target_nodes,
                         std::vector<Tensor>* outputs) {
  debug_init_notif.reset(new Notification());

  // Invoke the version of the method in parent class
  Status s = DirectSession::Run(inputs, output_names, target_nodes, outputs);

  return s;
}

::tensorflow::DebuggerResponse DebugSession::SendDebugMessage(
    const DebuggerRequest& request) {
  mutex_lock l(debug_lock_);

  // Wait until debug_executor is not nullptr anymore.
  // This means that calling SendDebugMessage before calling Run() will hang
  // until Run() is finally called.
  while (debug_executor == nullptr) {
    Env::Default()->SleepForMicroseconds(1000);
  }
  Env::Default()->SleepForMicroseconds(1000);

  DebugExecutorImpl* debug_exec_impl =
      reinterpret_cast<DebugExecutorImpl*>(debug_executor);

  return debug_exec_impl->HandleDebuggerMessage(request);
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
