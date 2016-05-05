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

#ifndef TENSORFLOW_COMMON_RUNTIME_EXECUTOR_H_
#define TENSORFLOW_COMMON_RUNTIME_EXECUTOR_H_

#include <deque>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/pending_counts.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/session_state.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

namespace {

bool IsInitializationOp(const Node* node) {
  return node->op_def().allows_uninitialized_input();
}

// Sets the timeline_label field of *node_stats, using data from *node.
// Returns true iff the node is a transfer node.
// TODO(tucker): merge with the DetailText function in session.cc
// in a common location.
bool SetTimelineLabel(const Node* node, NodeExecStats* node_stats) {
  bool is_transfer_node = false;
  string memory;
  for (auto& all : node_stats->memory()) {
    int64 tot = all.total_bytes();
    if (tot >= 0.1 * 1048576.0) {
      int64 peak = all.peak_bytes();
      if (peak > 0) {
        memory =
            strings::StrCat(memory, "[", all.allocator_name(),
                            strings::Printf(" %.1fMB %.1fMB] ", tot / 1048576.0,
                                            peak / 1048576.0));
      } else {
        memory = strings::StrCat(memory, "[", all.allocator_name(),
                                 strings::Printf(" %.1fMB] ", tot / 1048576.0));
      }
    }
  }
  const NodeDef& def = node->def();
  string text = "";
  if (IsSend(node)) {
    string tensor_name;
    TF_CHECK_OK(GetNodeAttr(def, "tensor_name", &tensor_name));
    string recv_device;
    TF_CHECK_OK(GetNodeAttr(def, "recv_device", &recv_device));
    text = strings::StrCat(memory, def.name(), " = ", def.op(), "(",
                           tensor_name, " @", recv_device);
    is_transfer_node = true;
  } else if (IsRecv(node)) {
    string tensor_name;
    TF_CHECK_OK(GetNodeAttr(def, "tensor_name", &tensor_name));
    string send_device;
    TF_CHECK_OK(GetNodeAttr(def, "send_device", &send_device));
    text = strings::StrCat(memory, def.name(), " = ", def.op(), "(",
                           tensor_name, " @", send_device);
    is_transfer_node = true;
  } else {
    text = strings::StrCat(
        memory, def.name(), " = ", def.op(), "(",
        str_util::Join(
            std::vector<StringPiece>(def.input().begin(), def.input().end()),
            ", "),
        ")");
  }
  node_stats->set_timeline_label(text);
  return is_transfer_node;
}

// Helper routines for collecting step stats.
namespace nodestats {
inline int64 NowInUsec() { return Env::Default()->NowMicros(); }

void SetScheduled(NodeExecStats* nt, int64 t) { nt->set_scheduled_micros(t); }

void SetAllStart(NodeExecStats* nt) { nt->set_all_start_micros(NowInUsec()); }

void SetOpStart(NodeExecStats* nt) {
  DCHECK_NE(nt->all_start_micros(), 0);
  nt->set_op_start_rel_micros(NowInUsec() - nt->all_start_micros());
}

void SetOpEnd(NodeExecStats* nt) {
  DCHECK_NE(nt->all_start_micros(), 0);
  nt->set_op_end_rel_micros(NowInUsec() - nt->all_start_micros());
}

void SetAllEnd(NodeExecStats* nt) {
  DCHECK_NE(nt->all_start_micros(), 0);
  nt->set_all_end_rel_micros(NowInUsec() - nt->all_start_micros());
}

void SetOutput(NodeExecStats* nt, int slot, const Tensor* v) {
  DCHECK(v);
  NodeOutput* no = nt->add_output();
  no->set_slot(slot);
  v->FillDescription(no->mutable_tensor_description());
}

void SetMemory(NodeExecStats* nt, OpKernelContext* ctx) {
  for (const auto& allocator_pair : ctx->wrapped_allocators()) {
    AllocatorMemoryUsed* memory = nt->add_memory();
    // retrieving the sizes from the wrapped allocator removes the
    // executor's reference to it, so allocator_pair.second must not
    // be dereferenced again after this statement
    auto sizes = allocator_pair.second->GetSizesAndUnRef();
    memory->set_allocator_name(allocator_pair.first->Name());
    int tb = sizes.first;
    memory->set_total_bytes(tb);
    if (allocator_pair.first->TracksAllocationSizes()) {
      memory->set_peak_bytes(sizes.second);
    }
  }
}

void SetReferencedTensors(NodeExecStats* nt,
                          const TensorReferenceVector& tensors) {
  // be careful not to increment the reference count on any tensor
  // while recording the information
  for (size_t i = 0; i < tensors.size(); ++i) {
    AllocationDescription* description = nt->add_referenced_tensor();
    tensors.at(i).FillDescription(description);
  }
}

}  // end namespace nodestats

}  // end namespace

class StepStatsCollector;

// Creates an Executor that computes the given "graph".
//
// If successful, returns the constructed executor in "*executor". The
// caller keeps the ownership of "device". The returned executor takes
// the ownership of "graph". Otherwise, returns an error status.
//
// "params" provides a set of context for the executor. We expect that
// different context would provide different implementations.
struct LocalExecutorParams {
  Device* device;

  // The library runtime support.
  FunctionLibraryRuntime* function_library;

  // create_kernel returns an instance of op kernel based on NodeDef.
  // delete_kernel is called for every kernel used by the executor
  // when the executor is deleted.
  std::function<Status(const NodeDef&, OpKernel**)> create_kernel;
  std::function<void(OpKernel*)> delete_kernel;
};

struct NodeItem {
  // A graph node.
  const Node* node = nullptr;

  // The kernel for this node.
  OpKernel* kernel = nullptr;

  bool kernel_is_expensive = false;  // True iff kernel->IsExpensive()
  bool kernel_is_async = false;      // True iff kernel->AsAsync() != nullptr
  bool is_merge = false;             // True iff IsMerge(node)

  // Cached values of node->num_inputs() and node->num_outputs(), to
  // avoid levels of indirection.
  int num_inputs;
  int num_outputs;

  // ExecutorImpl::tensors_[input_start] is the 1st positional input
  // for this node.
  int input_start = 0;

  // ExecutorImpl::output_attrs_[output_attr_start] is the 1st
  // positional attribute for the 0th output of this node.
  int output_attr_start = 0;

  DataType input_type(int i) const {
    DCHECK_LT(i, num_inputs);
    return (i < 4) ? inlined_input_type[i] : node->input_type(i);
  }
  DataType output_type(int i) const {
    DCHECK_LT(i, num_outputs);
    return (i < 4) ? inlined_output_type[i] : node->output_type(i);
  }
  // Cache first 4 input and output types to reduce levels of indirection
  DataType inlined_input_type[4];
  DataType inlined_output_type[4];
};

// Executor runs a graph computation.
// Example:
//   Graph* graph = ...;
//      ... construct graph ...
//   Executor* executor;
//   TF_CHECK_OK(NewSimpleExecutor(my_device, graph, &executor));
//   Rendezvous* rendezvous = NewNaiveRendezvous();
//   TF_CHECK_OK(rendezvous->Send("input", some_input_tensor));
//   TF_CHECK_OK(executor->Run({ExecutorOpts, rendezvous, nullptr}));
//   TF_CHECK_OK(rendezvous->Recv("input", &output_tensor));
//   ... ...
//
// Multiple threads can call Executor::Run concurrently.
class Executor {
 public:
  virtual ~Executor() {}

  // RunAsync() executes the graph computation. "done" is run when the
  // graph computation completes. If any error happens during the
  // computation, "done" is run and the error is passed to "done".
  //
  // RunAsync() is given a few arguments in Args. The caller must
  // ensure objects passed in Args (rendezvous, stats_collector, etc.)
  // are alive at least until done is invoked. All pointers to the
  // argument objects can be nullptr.
  //
  // "step_id" is a process-wide unique identifier for the step being
  // run. Executors on different devices may receive the same step_id
  // in the case that a step runs Ops on more than one device. The
  // step_id is used for tracking resource usage of a given step.
  //
  // RunAsync() uses the given "rendezvous", if not null, as the
  // mechanism to communicate inputs and outputs of the underlying
  // graph computation.
  //
  // RunAsync() calls "stats_collector", if not null, to keep track of
  // stats. This allows us to collect statistics and traces on demand.
  //
  // RunAsync() is provided a "call_frame", if the executor is used
  // for executing a function, is used to pass arguments and return
  // values between the caller and the callee.
  //
  // RunAsync() uses "cancellation_manager", if not nullptr, to
  // register callbacks that should be called if the graph computation
  // is cancelled. Note that the callbacks merely unblock any
  // long-running computation, and a cancelled step will terminate by
  // returning/calling the DoneCallback as usual.
  //
  // RunAsync() dispatches closures to "runner". Typically, "runner"
  // is backed up by a bounded threadpool.
  struct Args {
    int64 step_id = 0;
    Rendezvous* rendezvous = nullptr;
    StepStatsCollector* stats_collector = nullptr;
    FunctionCallFrame* call_frame = nullptr;
    CancellationManager* cancellation_manager = nullptr;
    SessionState* session_state = nullptr;
    TensorStore* tensor_store = nullptr;

    typedef std::function<void()> Closure;
    typedef std::function<void(Closure)> Runner;
    Runner runner = nullptr;
  };
  typedef std::function<void(const Status&)> DoneCallback;
  virtual void RunAsync(const Args& args, DoneCallback done) = 0;

  // Synchronous wrapper for RunAsync().
  Status Run(const Args& args) {
    Status ret;
    Notification n;
    RunAsync(args, [&ret, &n](const Status& s) {
      ret = s;
      n.Notify();
    });
    n.WaitForNotification();
    return ret;
  }

  // TODO(cais): Make the following protected or private
  // Owned
  // A cached value of params_
  LocalExecutorParams params_;
  const Graph* graph_;
  NodeItem* nodes_ = nullptr;     // array of size "graph_.num_node_ids()"
  int total_input_tensors_ = 0;   // == sum(nodes_[*].num_inputs())
  int total_output_tensors_ = 0;  // == sum(nodes_[*].num_outputs())

  bool device_record_tensor_accesses_ = false;

  // Root nodes (with no in edges) that should form the initial ready queue
  std::vector<const Node*> root_nodes_;

  PendingCounts initial_pending_counts_;

  // The number of inputs for each frame in this graph. This is static
  // information of the graph.
  std::unordered_map<string, int> frame_input_count_;

  std::vector<AllocatorAttributes> output_attrs_;

 protected:
  Executor(const LocalExecutorParams& p, const Graph* g)
      : graph_(g), params_(p), initial_pending_counts_(graph_->num_node_ids())  {
    CHECK(p.create_kernel != nullptr);
    CHECK(p.delete_kernel != nullptr);
  }
};

::tensorflow::Status NewLocalExecutor(const LocalExecutorParams& params,
                                      const Graph* graph, Executor** executor);

// A class to help run multiple executors in parallel and wait until
// all of them are complete.
//
// ExecutorBarrier deletes itself after the function returned by Get()
// is called.
class ExecutorBarrier {
 public:
  typedef std::function<void(const Status&)> StatusCallback;

  // Create an ExecutorBarrier for 'num' different executors.
  //
  // 'r' is the shared Rendezvous object that is used to communicate
  // state.  If any of the executors experiences an error, the
  // rendezvous object will be aborted exactly once.
  //
  // 'done' is called after the last executor completes, and
  // ExecutorBarrier is deleted.
  ExecutorBarrier(int num, Rendezvous* r, StatusCallback done)
      : rendez_(r), done_cb_(done), pending_(num) {}

  ~ExecutorBarrier() {}

  // Returns a closure that Executors must call when they are done
  // computing, passing the status of their execution as an argument.
  StatusCallback Get() {
    return std::bind(&ExecutorBarrier::WhenDone, this, std::placeholders::_1);
  }

 private:
  Rendezvous* rendez_ = nullptr;
  StatusCallback done_cb_ = nullptr;

  mutable mutex mu_;
  int pending_ GUARDED_BY(mu_) = 0;
  Status status_ GUARDED_BY(mu_);

  void WhenDone(const Status& s) {
    bool error = false;
    Rendezvous* error_rendez = nullptr;
    StatusCallback done = nullptr;
    Status status;
    {
      mutex_lock l(mu_);
      // If we are the first error encountered, mark the status
      // appropriately and later trigger an abort of the Rendezvous
      // object by this thread only.
      if (status_.ok() && !s.ok()) {
        error = true;
        error_rendez = rendez_;
        error_rendez->Ref();
        status_ = s;
      }

      // If this is the last call to WhenDone, call the final callback
      // below.
      if (--pending_ == 0) {
        CHECK(done_cb_ != nullptr);
        done = done_cb_;
        done_cb_ = nullptr;
      }

      status = status_;
    }

    if (error) {
      error_rendez->StartAbort(status);
      error_rendez->Unref();
    }
    if (done != nullptr) {
      delete this;
      done(status);
    }
  }

  TF_DISALLOW_COPY_AND_ASSIGN(ExecutorBarrier);
};

// An implementation of Executor
class ExecutorImpl : public Executor {
 public:
  ExecutorImpl(const LocalExecutorParams& p, const Graph* g)
      : Executor(p, g) {}

  virtual ~ExecutorImpl() override {
    for (int i = 0; i < graph_->num_node_ids(); i++) {
      params_.delete_kernel(nodes_[i].kernel);
    }
    delete[] nodes_;
    delete graph_;
  }

  Status Initialize();

  // Infer memory allocation attributes of a node n's output,
  // based on its use node dst.  Note that dst might not be directly
  // connected to n by a single edge, but might be a downstream
  // consumer of n's output by reference.  *attr is updated with any
  // necessary attributes.
  Status InferAllocAttr(const Node* n, const Node* dst,
                        const DeviceNameUtils::ParsedName& local_dev_name,
                        AllocatorAttributes* attr);

  // Process all Nodes in the current graph, attempting to infer the
  // memory allocation attributes to be used wherever they may allocate
  // a tensor buffer.
  Status SetAllocAttrs();

  void RunAsync(const Args& args, DoneCallback done) override;

 protected:
  // friend class ExecutorState; // TODO(cais): Remove this

  static void InitializePending(const Graph* graph, PendingCounts* counts);

  TF_DISALLOW_COPY_AND_ASSIGN(ExecutorImpl);
};

// A few helpers to facilitate create/delete kernels.

// Creates a kernel based on "ndef" on device "device". The kernel can
// access the functions in the "flib". The caller takes ownership of
// returned "*kernel".
Status CreateNonCachedKernel(Device* device, FunctionLibraryRuntime* flib,
                             const NodeDef& ndef, int graph_def_version,
                             OpKernel** kernel);

// Deletes "kernel" returned by CreateKernel.
void DeleteNonCachedKernel(OpKernel* kernel);

typedef gtl::InlinedVector<TensorValue, 4> TensorValueVec;
typedef gtl::InlinedVector<DeviceContext*, 4> DeviceContextVec;
typedef gtl::InlinedVector<AllocatorAttributes, 4> AllocatorAttributeVec;

// The state associated with one invocation of ExecutorImpl::Run.
// ExecutorState dispatches nodes when they become ready and keeps
// track of how many predecessors of a node have not done (pending_).
class ExecutorState {
 public:
  ExecutorState(const Executor::Args& args, ExecutorImpl* impl);
  ~ExecutorState();

  void RunAsync(Executor::DoneCallback done);

 private:
  typedef ExecutorState ME;

  // 1-D, 0 element tensor.
  // static const Tensor* const kEmptyTensor = new Tensor;  // TODO(cais): Remove
  static const Tensor* const kEmptyTensor;

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
    explicit IterationState(const ExecutorImpl* impl)
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
  };

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
  };

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

  const bool vlog_;  // true if VLOG_IS_ON(1). Used to check vlog cheaply.

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
  const ExecutorImpl* impl_;
  CancellationManager* cancellation_manager_;
  Executor::Args::Runner runner_;

  // Owned.

  // Step-local resource manager.
  ResourceMgr step_resource_manager_;

  // A flag that is set on error after the frame state has been
  // dumped for diagnostic purposes.
  bool dumped_on_error_ = false;

  // The root frame in which the execution of this step is started.
  FrameState* root_frame_;

  // Invoked when the execution finishes.
  Executor::DoneCallback done_cb_;

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
  // we need to check for the completion of output frame/iter.
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
                        const EntryVector& outputs, TaggedNodeSeq* ready);

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
};  // end class ExecutorState

}  // end namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_EXECUTOR_H_
