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

// 1-D, 0 element tensor.
static const Tensor* const kEmptyTensor = new Tensor;

class DebugExecutorState;  // tfdb(cais)

class DebugExecutorImpl : public ExecutorImpl {
 public:
  // Constructor
  DebugExecutorImpl(const LocalExecutorParams& p, const Graph* g);

  // tfdb(cais)
  DebuggerResponse HandleDebuggerMessage(const DebuggerRequest& request);

  void RunAsync(const Args& args, DoneCallback done) override;

  std::shared_ptr<MultiUseNotification> debugger_notification;
  std::unordered_map<string, Tensor> node_value_store;
  std::unordered_map<string, Tensor*> node_ref_store;

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

  string break_at_node;  // TODO(cais): Remove unused

  DebugExecutorState* executor_state;

  // tfdb(cais)
  std::unordered_map<string, Tensor> injected_tensors;

  TF_DISALLOW_COPY_AND_ASSIGN(DebugExecutorImpl);
};  // end class DebugExecutorImpl

// The state associated with one invocation of DebugExecutorImpl::Run.
// DebugExecutorState dispatches nodes when they become ready and keeps
// track of how many predecessors of a node have not done (pending_).
class DebugExecutorState {
 public:
  DebugExecutorState(const DebugExecutorImpl::Args& args, DebugExecutorImpl* impl);
  ~DebugExecutorState();

  void RunAsync(DebugExecutorImpl::DoneCallback done);

  // tfdb(cais)
  void InjectNodeValue(Tensor value);

 private:
  typedef DebugExecutorState ME;

  // Either a tensor pointer (pass-by-reference) or a tensor (pass-by-value).
  // TODO(yuanbyu): A better way to do "has_value"?
  struct Entry {
    Tensor val = *kEmptyTensor;  // A tensor value.
    Tensor* ref = nullptr;       // A tensor reference.
    mutex* ref_mu = nullptr;     // mutex for *ref if ref is not nullptr.
    bool has_value = false;      // Whether the value exists
    // The attributes of the allocator that creates the tensor.
    AllocatorAttributes alloc_attr;

    // Every entry carries an optional DeviceContext containing
    // Device-specific information about how the Tensor was produced.
    DeviceContext* device_context = nullptr;
  };

  // Contains a map from node id to the DeviceContext object that was
  // assigned by the device at the beginning of a step.
  DeviceContextMap device_context_map_;

  struct IterationState {
    explicit IterationState(const DebugExecutorImpl* impl)
        : input_tensors(new Entry[impl->total_input_tensors_]),
          outstanding_ops(0),
          outstanding_frame_count(0),
          counts_(impl->graph_->num_node_ids()) {
      counts_.InitializeFrom(impl->initial_pending_counts_);
    }

    // The state of an iteration.

    // One copy per iteration. For iteration k, i-th node's j-th input is in
    // input_tensors[k][impl_->nodes[i].input_start + j]. An entry is either
    // a tensor pointer (pass-by-reference) or a tensor (pass-by-value).
    //
    // NOTE: No need to protect input_tensors[i] by any locks because it
    // is resized once. Each element of tensors_ is written once by the
    // source node of an edge and is cleared by the destination of the same
    // edge. The latter node is never run concurrently with the former node.
    Entry* input_tensors;

    // The number of outstanding ops for each iteration.
    int outstanding_ops;

    // The number of outstanding frames for each iteration.
    int outstanding_frame_count;
    int pending(int id) { return counts_.pending(id); }
    int decrement_pending(int id, int v) {
      return counts_.decrement_pending(id, v);
    }
    // Mark a merge node as live
    // REQUIRES: Node corresponding to "id" is a merge node
    void mark_live(int id) { counts_.mark_live(id); }
    // Mark a node to show that processing has started.
    void mark_started(int id) { counts_.mark_started(id); }
    // Mark a node to show that processing has completed.
    void mark_completed(int id) { counts_.mark_completed(id); }
    PendingCounts::NodeState node_state(int id) {
      return counts_.node_state(id);
    }

    int dead_count(int id) { return counts_.dead_count(id); }
    void increment_dead_count(int id) { counts_.increment_dead_count(id); }

    ~IterationState() { delete[] input_tensors; }

   private:
    PendingCounts counts_;
  };  // end struct IterationState

  struct FrameState {
    // A new frame is created for each loop. Execution starts at iteration 0.
    // When a value at iteration 0 passes through a NextIteration node,
    // iteration 1 is created and starts running. Note that iteration 0 may
    // still be running so multiple iterations may run in parallel. The
    // frame maintains the state of iterations in several data structures
    // such as pending_count and input_tensors. When iteration 0 completes,
    // we garbage collect the state of iteration 0.
    //
    // A frame instance is considered "done" and can be garbage collected
    // if all its inputs have entered and all its iterations are "done".
    //
    // A frame manages the live iterations of an iterative computation.
    // Iteration i is considered "done" when there are no outstanding ops,
    // frames at iteration i are done, all recvs for this iteration are
    // completed, and iteration i-1 is done. For iteration 0, we instead
    // wait for there to be no more pending inputs of the frame.
    //
    // Frames and iterations are garbage collected once they are done.
    // The state we need to keep around is highly dependent on the
    // parallelism enabled by the scheduler. We may want to have the
    // scheduler dynamically control the outstanding number of live
    // parallel frames and iterations. To reduce the state space, the
    // scheduler might want to schedule ops in inner frames first and
    // lower iterations first.
    //
    // This frame state is mostly initialized lazily on demand so we
    // don't introduce unnecessary overhead.

    // The name of this frame, which is the concatenation of its parent
    // frame name, the iteration of the parent frame when this frame was
    // created, and the value of the attr 'frame_name'.
    string frame_name;

    // The unique id for this frame. Generated by fingerprinting
    // frame_name.
    uint64 frame_id;

    // The iteration id of its parent frame when this frame is created.
    // -1 if there is no parent frame. The frame_name/parent_iter pair
    // uniquely identifies this FrameState.
    int64 parent_iter = -1;

    // The FrameState of its parent frame.
    FrameState* parent_frame = nullptr;

    // The highest iteration number we have reached so far in this frame.
    int64 iteration_count = 0;

    // The number of inputs this frame is still waiting.
    int num_pending_inputs = 0;

    // The number of outstanding iterations.
    int num_outstanding_iterations = 0;

    // The maximum allowed number of parallel iterations.
    int max_parallel_iterations = 1;

    // The iteration states of this frame.
    gtl::InlinedVector<IterationState*, 12> iterations;

    // The NextIteration nodes to enter a new iteration. If the number of
    // outstanding iterations reaches the limit, we will defer the start of
    // the next iteration until the number of outstanding iterations falls
    // below the limit.
    std::vector<std::pair<const Node*, Entry>> next_iter_roots;

    // The values of the loop invariants for this loop. They are added into
    // this list as they "enter" the frame. When a loop invariant enters,
    // we make it available to all active iterations. When the frame starts
    // a new iteration, we make all the current loop invariants available
    // to the new iteration.
    std::vector<std::pair<const Node*, Entry>> inv_values;

    // The list of dead exit nodes for the current highest iteration. We
    // will only "execute" the dead exits of the final iteration.
    std::vector<const Node*> dead_exits;

    IterationState* GetIteration(int64 iter) {
      int index = iter % iterations.size();
      return iterations[index];
    }

    void SetIteration(int64 iter, IterationState* state) {
      int index = iter % iterations.size();
      iterations[index] = state;
    }

    ~FrameState() {
      for (size_t i = 0; i < iterations.size(); ++i) {
        delete iterations[i];
        iterations[i] = nullptr;
      }
    }
  };  // struct FrameState

  // A tagged node: <frame*, iter, node*>.
  struct TaggedNode {
    const Node* node = nullptr;
    FrameState* input_frame = nullptr;
    int64 input_iter = -1;
    bool is_dead = false;

    TaggedNode(const Node* t_node, FrameState* in_frame, int64 in_iter,
               bool dead) {
      node = t_node;
      input_frame = in_frame;
      input_iter = in_iter;
      is_dead = dead;
    }
  };

  typedef gtl::InlinedVector<TaggedNode, 8> TaggedNodeSeq;
  typedef gtl::InlinedVector<Entry, 4> EntryVector;

  int64 step_id_;
  // Not owned.
  Rendezvous* rendezvous_;
  SessionState* session_state_;
  TensorStore* tensor_store_;
  StepStatsCollector* stats_collector_;
  // QUESTION: Make it a checkpoint::TensorSliceReaderCacheWrapper
  // instead of a pointer?  (avoids having to delete).
  checkpoint::TensorSliceReaderCacheWrapper* slice_reader_cache_;
  FunctionCallFrame* call_frame_;
  DebugExecutorImpl* impl_;
  CancellationManager* cancellation_manager_;
  DebugExecutorImpl::Args::Runner runner_;

  // Owned.

  // Step-local resource manager.
  ResourceMgr step_resource_manager_;

  // A flag that is set on error after the frame state has been
  // dumped for diagnostic purposes.
  bool dumped_on_error_ = false;

  // The root frame in which the execution of this step is started.
  FrameState* root_frame_;

  // Invoked when the execution finishes.
  DebugExecutorImpl::DoneCallback done_cb_;

  std::atomic_int_fast32_t num_outstanding_ops_;

  mutex mu_;
  Status status_ GUARDED_BY(mu_);

  // Mapping from frame name to outstanding frames. A new frame is created
  // at some iteration of an active frame. So the unique key for the new
  // child frame is composed of the name of the parent frame, the iteration
  // number at which the parent frame is creating the new frame, and the
  // name of the new frame from nodedef.
  std::unordered_map<string, FrameState*> outstanding_frames_ GUARDED_BY(mu_);

  // The unique name of a frame.
  inline string MakeFrameName(FrameState* frame, int64 iter_id, string name) {
    return strings::StrCat(frame->frame_name, ";", iter_id, ";", name);
  }

  // Find an existing or create a new child frame in the frame 'frame' at
  // iteration 'iter'.
  void FindOrCreateChildFrame(FrameState* frame, int64 iter, const Node* node,
                              FrameState** child) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Increments the iteration id. If this is a new iteration, initialize it.
  void IncrementIteration(FrameState* frame, TaggedNodeSeq* ready)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Returns true if the computation in the frame is completed.
  bool IsFrameDone(FrameState* frame) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Returns true if the iteration of the frame is completed.
  bool IsIterationDone(FrameState* frame, int64 iter)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Get the output frame/iter of a node. Create new frame/iteration if
  // needed. If there are dead roots for the new iteration, we need to
  // "execute" them so ad them to the ready queue. Returns true if
  // we need to check for the completion of output frame/iter._idx
  bool SetOutputFrameIter(const TaggedNode& tagged_node,
                          const EntryVector& outputs, FrameState** frame,
                          int64* iter, TaggedNodeSeq* ready)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Cleanup frames and iterations
  void CleanupFramesIterations(FrameState* frame, int64 iter,
                               TaggedNodeSeq* ready)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Activate all the deferred NextIteration nodes in a new iteration.
  void ActivateNexts(FrameState* frame, int64 iter, TaggedNodeSeq* ready)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Activate all the current loop invariants in a new iteration.
  void ActivateLoopInvs(FrameState* frame, int64 iter, TaggedNodeSeq* ready)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Add a new loop invariant and make it available to all active iterations.
  void AddLoopInv(FrameState* frame, const Node* node, const Entry& value,
                  TaggedNodeSeq* ready) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Activate the successors of a node.
  void ActivateNode(const Node* node, const bool is_dead, FrameState* frame,
                    int64 iter, const EntryVector& outputs,
                    TaggedNodeSeq* ready) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Process a ready node in current thread.
  void Process(TaggedNode node, int64 scheduled_usec);

  // Before invoking item->kernel, fills in its "inputs".
  Status PrepareInputs(const NodeItem& item, Entry* first_input,
                       TensorValueVec* inputs,
                       DeviceContextVec* input_device_contexts,
                       AllocatorAttributeVec* input_alloc_attrs,
                       bool* is_input_dead);

  // After item->kernel computation is done, processes its outputs.
  Status ProcessOutputs(const NodeItem& item, OpKernelContext* ctx,
                        EntryVector* outputs, NodeExecStats* stats);

  // After processing the outputs, propagates the outputs to their dsts.
  void PropagateOutputs(const TaggedNode& tagged_node,
                        const EntryVector& outputs,
                        TaggedNodeSeq* ready);

  // "node" just finishes. Takes ownership of "stats". Returns true if
  // execution has completed.
  bool NodeDone(const Status& s, const Node* node, const TaggedNodeSeq& ready,
                NodeExecStats* stats, std::deque<TaggedNode>* inline_ready);

  // Call Process() on all nodes in 'inline_ready'.
  void ProcessInline(const std::deque<TaggedNode>& inline_ready);

  // Schedule all the expensive nodes in 'ready', and put all the inexpensive
  // nodes in 'ready' into 'inline_ready'.
  void ScheduleReady(const TaggedNodeSeq& ready,
                     std::deque<TaggedNode>* inline_ready);

  // Provide debugging output about an outstanding node in the executor.
  void DumpCompletedNodeState(const int node_id, const Entry* input_vector);
  void DumpPendingNodeState(const int node_id, const Entry* input_vector,
                            bool show_nodes_with_no_ready_inputs);
  void DumpActiveNodeState(const int node_id, const Entry* input_vector);

  // Provide debugging output about an outstanding iteration in the executor.
  void DumpIterationState(IterationState* iteration);

  // Provide debugging output of the state of the executor.
  void DumpState();

  // One thread of control finishes.
  void Finish();

  // A standalone routine for this expression so that we can express
  // that we don't want thread safety analysis on this reference (it's
  // safe to do without the lock because the iterations array never
  // resizes and this particular iteration's array element will not
  // be changed out from under us because the iteration is still alive).
  Entry* GetInputTensors(FrameState* input_frame,
                         int64 input_iter) const NO_THREAD_SAFETY_ANALYSIS {
    return input_frame->GetIteration(input_iter)->input_tensors;
  }

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

  // NOTE: Experimental and subject to change.
  ::tensorflow::Status Run(const ::tensorflow::RunOptions& run_options,
                           const NamedTensorList& inputs,
                           const std::vector<string>& output_names,
                           const std::vector<string>& target_nodes,
                           std::vector<Tensor>* outputs,
                           RunMetadata* run_metadata) override;

  ::tensorflow::DebuggerResponse
      SendDebugMessage(const DebuggerRequest& request) override;

 private:
  // typedef DebugSession ME; // TODO(cais): Remove inheritied

  // Retrieves an already existing set of executors to run 'inputs' and
  // 'outputs', or creates and caches them for future use.
  ::tensorflow::Status GetOrCreateExecutors(
      gtl::ArraySlice<string> inputs, gtl::ArraySlice<string> outputs,
      gtl::ArraySlice<string> target_nodes,
      ExecutorsAndKeys** executors_and_keys, RunStateArgs* run_state_args)
      override;

  // Schedules 'c' for execution.     // Schedules 'c' for execution.
  void SchedClosure(std::function<void()> c) override;

  // tfdb(cais)
  Executor* debug_executor;

  TF_DISALLOW_COPY_AND_ASSIGN(DebugSession);

  mutex debug_lock_;
};  // end class DebugSession

}  // end namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_DEBUG_SESSION_H_
