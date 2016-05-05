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

#ifndef TENSORFLOW_COMMON_RUNTIME_DEBUG_SESSION_H_
#define TENSORFLOW_COMMON_RUNTIME_DEBUG_SESSION_H_

#include <atomic>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/pending_counts.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/session_state.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/debugger.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

class DebugExecutorState;  // tfdb(cais)

class DebugExecutorImpl : public ExecutorImpl {
 public:
  // Constructor
  DebugExecutorImpl(const LocalExecutorParams& p, const Graph* g);

  // tfdb(cais)
  DebuggerResponse HandleDebuggerMessage(const DebuggerRequest& request);

  void RunAsync(const Args& args, DoneCallback done) override;

  // Notification object for debugger thread
  std::shared_ptr<MultiUseNotification> debug_notification;

  // Notification object for graph execution thread
  std::shared_ptr<MultiUseNotification> exec_notification;

  std::unordered_map<string, Tensor> node_value_store;
  std::unordered_map<string, Tensor*> node_ref_store;

  // Pre-calculates order of nodes during execution for debugging
  void CalcNodeOrder();

 private:
  friend class DebugExecutorState;

  // Simulation methods for calculating node order
  void SimProcess(const string& node_name);
  void SimPropagateOutputs(const string& node_name,
                           std::deque<string>* ready);
  void SimNodeDone(const string& node_name,
                   const std::deque<string>& ready_queue,
                   std::deque<string>* inline_ready_queue);
  void SimScheduleReady(const std::deque<string>& ready_queue,
                        std::deque<string>* inline_ready_queue);

  const Node* NodeName2Node(const string& node_name) const;
  bool NodeName2NodeKernelIsExpensive(const string& node_name) const;

  std::vector<string> GetCompletedNodes();
  std::vector<string> GetNotCompletedNodes();

  std::deque<string> node_order;
  std::unordered_set<string> done_nodes;

  std::shared_ptr<thread::ThreadPool> thread_pool_;

  string break_at_node;

  DebugExecutorState* executor_state;

  // tfdb(cais)
  std::unordered_map<string, Tensor> injected_tensors;

  TF_DISALLOW_COPY_AND_ASSIGN(DebugExecutorImpl);
};  // end class DebugExecutorImpl

// The state associated with one invocation of DebugExecutorImpl::Run.
// DebugExecutorState dispatches nodes when they become ready and keeps
// track of how many predecessors of a node have not done (pending_).
class DebugExecutorState : public ExecutorState {
 public:
  DebugExecutorState(const DebugExecutorImpl::Args& args, DebugExecutorImpl* impl);

  void RunAsync(DebugExecutorImpl::DoneCallback done) override;

  // tfdb(cais)
  void InjectNodeValue(Tensor value);

 private:

  // Activate the successors of a node.
  void ActivateNode(const Node* node, const bool is_dead, FrameState* frame,
                    int64 iter, const EntryVector& outputs,
                    TaggedNodeSeq* ready) EXCLUSIVE_LOCKS_REQUIRED(mu_)
      override;

  // Override the two hooks for debugging
  void NodeDoneEarlyHook(const Node* node) override;
  void NodeDoneLateHook(const Node* node) override;

  DebugExecutorImpl* debug_exec_impl_;

  const Node* stored_node;
  FrameState* stored_output_frame;
  int64 stored_output_iter;
  EntryVector stored_outputs;
};  // class DebugExecutorState

class CostModel;
class Device;
class ThreadPool;

class DebugSession : public DirectSession {
 public:
  // Takes ownership of 'device_mgr'.
  DebugSession(const SessionOptions& options, const DeviceMgr* device_mgr);

  ::tensorflow::Status Run(const NamedTensorList& inputs,
                           const std::vector<string>& output_names,
                           const std::vector<string>& target_nodes,
                           std::vector<Tensor>* outputs) override;

  ::tensorflow::DebuggerResponse
      SendDebugMessage(const DebuggerRequest& request) override;

 private:
  ::tensorflow::Status CreateLocalExecutor(
      const LocalExecutorParams& params, const Graph* graph,
      Executor** executor) override;

  void SchedClosure(std::function<void()> c) override;

  ::tensorflow::Status GetOrCreateExecutors(
      gtl::ArraySlice<string> inputs, gtl::ArraySlice<string> outputs,
      gtl::ArraySlice<string> target_nodes,
      ExecutorsAndKeys** executors_and_keys, RunStateArgs* run_state_args)
      override;

  void WaitForNotification(RunState* run_state, int64 timeout_in_ms) override;

  // tfdb: Special executor for debugging
  Executor* debug_executor;

  TF_DISALLOW_COPY_AND_ASSIGN(DebugSession);

  mutex debug_lock_;
  std::shared_ptr<Notification> debug_init_notif;
};  // end class DebugSession

}  // end namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_DEBUG_SESSION_H_
