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

#include "tensorflow/core/common_runtime/executor.h"

#include <atomic>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/pending_counts.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_segment.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/edgeset.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

namespace tensorflow {

Status ExecutorImpl::Initialize() {
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

  // Preprocess every node in the graph to create an instance of op
  // kernel for each node;
  for (const Node* n : graph_->nodes()) {
    const int id = n->id();

    // See if this node is a root node, and if so, add to root_nodes_
    const int num_in_edges = n->in_edges().size();
    if (num_in_edges == 0) {
      root_nodes_.push_back(n);
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
  if (!s.ok()) return s;
  return SetAllocAttrs();
}

Status ExecutorImpl::SetAllocAttrs() {
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

Status ExecutorImpl::InferAllocAttr(
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

ExecutorState::ExecutorState(const Executor::Args& args, ExecutorImpl* impl)
    : vlog_(VLOG_IS_ON(1)),
      step_id_(args.step_id),
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

  if (vlog_) VLOG(2) << "Create frame: " << root_frame_->frame_name;

  // Initialize the iteration.
  IterationState* iter_state = new IterationState(impl);
  root_frame_->iterations[0] = iter_state;

  // Initialize the executor state.
  outstanding_frames_.insert({root_frame_->frame_name, root_frame_});
}

// }  // end namespace

ExecutorState::~ExecutorState() {
  for (auto name_frame : outstanding_frames_) {
    delete name_frame.second;
  }

  for (auto it : device_context_map_) {
    it.second->Unref();
  }

  delete slice_reader_cache_;
}

void ExecutorImpl::InitializePending(const Graph* graph,
                                     PendingCounts* counts) {
  for (int id = 0; id < graph->num_node_ids(); id++) {
    counts->set_initial_count(id, 0, 0);  // Make sure everything is initialized
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

void ExecutorState::RunAsync(Executor::DoneCallback done) {
  const Graph* graph = impl_->graph_;
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
    DCHECK_EQ(n->in_edges().size(), 0);
    ready.push_back(TaggedNode{n, root_frame_, 0, false});
  }
  if (ready.empty()) {
    done(Status::OK());
  } else {
    num_outstanding_ops_ = ready.size();
    root_frame_->iterations[0]->outstanding_ops = ready.size();
    done_cb_ = done;
    // Schedule to run all the ready ops in thread pool.
    ScheduleReady(ready, nullptr);
  }
}

namespace {

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

}  // end namespace

const Tensor* const ExecutorState::kEmptyTensor = new Tensor;

void ExecutorState::Process(TaggedNode tagged_node, int64 scheduled_usec) {
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
    if (vlog_ && VLOG_IS_ON(1)) {
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
      SetScheduled(stats, scheduled_usec);
      SetAllStart(stats);
    }

    if (vlog_) {
      VLOG(1) << "Process node: " << id << " step " << params.step_id << " "
              << SummarizeNodeDef(node->def());
    }

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
        if (vlog_ && VLOG_IS_ON(1)) {
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
        // Asynchronous computes.
        AsyncOpKernel* async = item.kernel->AsAsync();
        DCHECK(async != nullptr);
        launched_asynchronously = true;
        auto pcopy = CopyParams(params);
        auto ctx = new OpKernelContext(pcopy, item.num_outputs);
        auto done = [this, tagged_node, item, first_input, ctx, stats, pcopy,
                     device]() {
          if (vlog_) {
            VLOG(2) << this << " Async kernel done: "
                    << SummarizeNodeDef(item.node->def());
          }
          if (stats_collector_) SetOpEnd(stats);
          EntryVector outputs;
          Status s = ProcessOutputs(item, ctx, &outputs, stats);
          if (stats_collector_) SetMemory(stats, ctx);
          // Clears inputs.
          int num_inputs = item.num_inputs;
          for (int i = 0; i < num_inputs; ++i) {
            (first_input + i)->val = *kEmptyTensor;
          }
          // TODO(misard) Replace with a finer-grain enabling flag once we
          // add better optional debugging support.
          if (vlog_ && VLOG_IS_ON(1)) {
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
              SetReferencedTensors(stats, accessed);
            // callee takes ownership of the vector
            device->ConsumeListOfAccessedTensors(ctx->op_device_context(),
                                                 accessed);
          }
          bool completed = NodeDone(s, item.node, ready, stats, nullptr);
          delete ctx;
          DeleteParams(pcopy);
          if (completed) Finish();
        };
        if (stats_collector_) SetOpStart(stats);
        device->ComputeAsync(async, ctx, done);
      } else {
        // Synchronous computes.
        OpKernelContext ctx(&params, item.num_outputs);
        if (stats_collector_) SetOpStart(stats);
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
        if (stats_collector_) SetOpEnd(stats);

        s = ProcessOutputs(item, &ctx, &outputs, stats);
        if (s.ok() && impl_->device_record_tensor_accesses_) {
          // Get the list of all tensors accessed during the execution
          ctx.retrieve_accessed_tensors(&accessed_tensors);
          device_context = ctx.op_device_context();
        }
        if (stats_collector_) SetMemory(stats, &ctx);
      }
    }

    if (!launched_asynchronously) {
      // Clears inputs.
      const int num_inputs = item.num_inputs;
      for (int i = 0; i < num_inputs; ++i) {
        (first_input + i)->val = *kEmptyTensor;
      }
      // TODO(misard) Replace with a finer-grain enabling flag once we
      // add better optional debugging support.
      if (vlog_ && VLOG_IS_ON(1)) {
        mutex_lock l(mu_);
        IterationState* iter_state = input_frame->GetIteration(input_iter);
        iter_state->mark_completed(id);
      }
      // Propagates outputs.
      if (s.ok()) {
        PropagateOutputs(tagged_node, outputs, &ready);
      }
      outputs.clear();
      if (!accessed_tensors.empty()) {
        if (stats_collector_)
          SetReferencedTensors(stats, accessed_tensors);
        // device_context is set above in synchronous computes
        device->ConsumeListOfAccessedTensors(device_context, accessed_tensors);
      }
      if (stats_collector_) {
        scheduled_usec = NowInUsec();
      }
      // Postprocess.
      completed = NodeDone(s, item.node, ready, stats, &inline_ready);
    }
  }  // while !inline_ready.empty()

  // This thread of computation is done if completed = true.
  if (completed) Finish();
}

Status ExecutorState::PrepareInputs(const NodeItem& item, Entry* first_input,
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

Status ExecutorState::ProcessOutputs(const NodeItem& item, OpKernelContext* ctx,
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
    if (vlog_ && VLOG_IS_ON(1)) {
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
          SetOutput(stats, i, val.tensor);
        }
      } else {
        s.Update(errors::Internal("Output ", i, " of type ",
                                  DataTypeString(dtype),
                                  " does not match declared output type ",
                                  DataTypeString(item.output_type(i)),
                                  " for node ", SummarizeNodeDef(node->def())));
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

void ExecutorState::PropagateOutputs(const TaggedNode& tagged_node,
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

void ExecutorState::ActivateNode(const Node* node, const bool is_dead,
                                 FrameState* output_frame, int64 output_iter,
                                 const EntryVector& outputs,
                                 TaggedNodeSeq* ready) {
  const NodeItem* nodes = impl_->nodes_;
  IterationState* output_iter_state = output_frame->GetIteration(output_iter);
  for (const Edge* e : node->out_edges()) {
    const Node* dst_node = e->dst();
    const int dst_id = dst_node->id();
    const int src_slot = e->src_output();

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

void ExecutorState::ActivateNexts(FrameState* frame, int64 iter,
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

void ExecutorState::ActivateLoopInvs(FrameState* frame, int64 iter,
                                     TaggedNodeSeq* ready) {
  // Propagate loop invariants to the new iteration.
  for (auto& node_entry : frame->inv_values) {
    const Node* node = node_entry.first;
    const Entry& entry = node_entry.second;
    const bool is_dead = !entry.has_value;
    ActivateNode(node, is_dead, frame, iter, {entry}, ready);
  }
}

void ExecutorState::AddLoopInv(FrameState* frame, const Node* node,
                               const Entry& entry, TaggedNodeSeq* ready) {
  // Store this value.
  frame->inv_values.push_back({node, entry});

  // Make this value available to all iterations.
  bool is_dead = !entry.has_value;
  for (int i = 1; i <= frame->iteration_count; ++i) {
    ActivateNode(node, is_dead, frame, i, {entry}, ready);
  }
}

bool ExecutorState::NodeDone(const Status& s, const Node* node,
                             const TaggedNodeSeq& ready, NodeExecStats* stats,
                             std::deque<TaggedNode>* inline_ready) {
  if (stats_collector_) {
    SetAllEnd(stats);
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
    ScheduleReady(ready, inline_ready);
  }
  return completed;
}

void ExecutorState::ProcessInline(const std::deque<TaggedNode>& inline_ready) {
  if (inline_ready.empty()) return;
  int64 scheduled_usec = 0;
  if (stats_collector_) {
    scheduled_usec = NowInUsec();
  }
  for (auto& tagged_node : inline_ready) {
    Process(tagged_node, scheduled_usec);
  }
}

void ExecutorState::ScheduleReady(const TaggedNodeSeq& ready,
                                  std::deque<TaggedNode>* inline_ready) {
  if (ready.empty()) return;

  int64 scheduled_usec = 0;
  if (stats_collector_) {
    scheduled_usec = NowInUsec();
  }
  if (inline_ready == nullptr) {
    // Schedule to run all the ready ops in thread pool.
    for (auto& tagged_node : ready) {
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
      inline_ready->push_back(tagged_node);
    } else {
      if (curr_expensive_node) {
        // Dispatch to another thread since there is plenty of work to
        // do for this thread.
        runner_(std::bind(&ME::Process, this, *curr_expensive_node,
                          scheduled_usec));
      }
      curr_expensive_node = &tagged_node;
    }
  }
  if (curr_expensive_node) {
    if (inline_ready->empty()) {
      // Tail recursion optimization
      inline_ready->push_back(*curr_expensive_node);
    } else {
      // There are inline nodes to run already. We dispatch this expensive
      // node to other thread.
      runner_(
          std::bind(&ME::Process, this, *curr_expensive_node, scheduled_usec));
    }
  }
}

void ExecutorState::DumpCompletedNodeState(const int node_id,
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

void ExecutorState::DumpPendingNodeState(
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

void ExecutorState::DumpActiveNodeState(const int node_id,
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

void ExecutorState::DumpIterationState(IterationState* iteration) {
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

void ExecutorState::DumpState() {
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

void ExecutorState::Finish() {
  mu_.lock();
  auto status = status_;
  auto done_cb = done_cb_;
  auto runner = runner_;
  mu_.unlock();
  delete this;
  CHECK(done_cb != nullptr);
  runner([done_cb, status]() { done_cb(status); });
}

bool ExecutorState::IsFrameDone(FrameState* frame) {
  return (frame->num_pending_inputs == 0 &&
          frame->num_outstanding_iterations == 0);
}

bool ExecutorState::IsIterationDone(FrameState* frame, int64 iter) {
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

void ExecutorState::FindOrCreateChildFrame(FrameState* frame, int64 iter,
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
    if (vlog_) {
      VLOG(2) << "Create frame: " << child_name;
    }

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

void ExecutorState::IncrementIteration(FrameState* frame,
                                       TaggedNodeSeq* ready) {
  frame->iteration_count++;
  int64 next_iter = frame->iteration_count;

  if (vlog_) {
    VLOG(2) << "Create iteration: [" << frame->frame_name << ", " << next_iter
            << "]";
  }

  IterationState* iter_state = new IterationState(impl_);
  frame->SetIteration(next_iter, iter_state);
  frame->num_outstanding_iterations++;
  frame->dead_exits.clear();

  // Activate the successors of the deferred roots in the new iteration.
  ActivateNexts(frame, next_iter, ready);

  // Activate the loop invariants in the new iteration.
  ActivateLoopInvs(frame, next_iter, ready);
}

bool ExecutorState::SetOutputFrameIter(const TaggedNode& tagged_node,
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

void ExecutorState::CleanupFramesIterations(FrameState* frame, int64 iter,
                                            TaggedNodeSeq* ready) {
  int64 curr_iter = iter;
  while (curr_iter <= frame->iteration_count &&
         IsIterationDone(frame, curr_iter)) {
    // Delete the iteration curr_iter
    if (vlog_) {
      VLOG(2) << "Delete iteration [" << frame->frame_name << ", " << curr_iter
              << "].";
    }

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
    if (vlog_) {
      VLOG(2) << "Delete frame " << frame_name;
    }
    outstanding_frames_.erase(frame_name);
    delete frame;

    // Cleanup recursively
    if (parent_frame != nullptr) {
      parent_frame->GetIteration(parent_iter)->outstanding_frame_count--;
      CleanupFramesIterations(parent_frame, parent_iter, ready);
    }
  }
}

bool ExecutorState::SetTimelineLabel(const Node* node, NodeExecStats* node_stats) {
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

void ExecutorImpl::RunAsync(const Args& args, DoneCallback done) {
  (new ExecutorState(args, this))->RunAsync(done);
}

Status NewLocalExecutor(const LocalExecutorParams& params, const Graph* graph,
                        Executor** executor) {
  ExecutorImpl* impl = new ExecutorImpl(params, graph);
  Status s = impl->Initialize();
  if (s.ok()) {
    *executor = impl;
  } else {
    delete impl;
  }
  return s;
}

Status CreateNonCachedKernel(Device* device, FunctionLibraryRuntime* flib,
                             const NodeDef& ndef, int graph_def_version,
                             OpKernel** kernel) {
  auto device_type = DeviceType(device->attributes().device_type());
  auto allocator = device->GetAllocator(AllocatorAttributes());
  return CreateOpKernel(device_type, device, allocator, flib, ndef,
                        graph_def_version, kernel);
}

void DeleteNonCachedKernel(OpKernel* kernel) { delete kernel; }

Status CreateCachedKernel(Device* device, const string& session,
                          FunctionLibraryRuntime* flib, const NodeDef& ndef,
                          int graph_def_version, OpKernel** kernel) {
  auto op_seg = device->op_segment();
  auto create_fn = [device, flib, &ndef, graph_def_version](OpKernel** kernel) {
    return CreateNonCachedKernel(device, flib, ndef, graph_def_version, kernel);
  };
  return op_seg->FindOrCreate(session, ndef.name(), kernel, create_fn);
}

// Deletes "kernel".
void DeleteCachedKernel(Device* device, const string& session,
                        OpKernel* kernel) {
  // Do nothing.
}

}  // end namespace tensorflow
