/* Copyright 2016 TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/debug/debug_session.h"

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

DebugSession::DebugSession(const SessionOptions& options,
                           const DeviceMgr* device_mgr)
  : DirectSession(options, device_mgr), host_tensors_() {
  // Disable the graph optimization by default
  SetOptimizeGraph(false);

  // Supply node output callback
  SetNodeOutputsCallback([this](const string& node_name,
                                const int output_slot,
                                const Tensor* tensor,
                                const bool is_ref,
                                OpKernelContext* ctx) {
    // std::ostringstream stream;

    if (comp_cb_ != nullptr) {
      comp_cb_(node_name, output_slot, is_ref);
    }

    // DEBUG
    // stream << "node_outputs_cb from debug session: node_name = "
    //        << node_name << std::endl;
    // stream << "  is_ref: " << is_ref << std::endl;
    // stream << "  shape: " << tensor->shape().DebugString() << std::endl;
    // stream << "  dtype: " << tensor->dtype() << std::endl;
    // stream << "  val of " << node_name << " (on cpu): "
    //        << tensor->DebugString() << std::endl;

    // state_dumper_.dump(stream.str(), node_name, tensor);
    // std::cout << stream.str() << std::endl << std::flush;

    // Copy tensor values (e.g., from GPU to host) only if the
    // value callback is not nullptr.
    // if (val_cb_ != nullptr) {
      CopyTensor(
          node_name, output_slot, tensor, ctx,
          [this, node_name, output_slot, is_ref](
              const Tensor* copied_tensor) {
            std::cout << "||| copied_tensor of node " << node_name
                      << "; value = " << copied_tensor->DebugString()
                      << std::endl << std::flush;  // DEBUG
            // val_cb_(node_name, output_slot, *copied_tensor, is_ref);
      });
    // }

    return Status::OK();
  });

}

void DebugSession::SetNodeCompletionCallback(
    NodeCompletionCallback callback) {
  comp_cb_ = callback;
}

void DebugSession::SetNodeValueCallback(NodeValueCallback callback) {
  val_cb_ = callback;
}

Status DebugSession::Run(const std::vector<std::pair<string, Tensor> >& inputs,
                         const std::vector<string>& output_tensor_names,
                         const std::vector<string>& target_node_names,
                         std::vector<Tensor>* outputs) {
  {
    mutex_lock l(mu_);
    host_tensors_.clear();
  }

  return DirectSession::Run(inputs, output_tensor_names, target_node_names,
                            outputs);
}

void DebugSession::CopyTensor(const string& node_name,
                              const int output_slot,
                              const Tensor* src_tensor,
                              OpKernelContext* ctx,
                              CopyDoneCallback copy_done_cb) {
  Device* device = static_cast<Device*>(ctx->device());
  AllocatorAttributes alloc_attrs = ctx->output_alloc_attr(output_slot);

  const bool is_tensor_initialized = src_tensor->IsInitialized();

  // Omit uniniatizlied Tensors.
  if (device->name().find("gpu:") != string::npos &&
      !alloc_attrs.on_host() && is_tensor_initialized) {
      // For GPU tensors, copy them to CPU.
      Allocator* cpu_allocator = tensorflow::cpu_allocator();
      Tensor* cpu_tensor = new Tensor(cpu_allocator,
                                      src_tensor->dtype(),
                                      src_tensor->shape());

      // Keep track of the tensors created for copying and free them later.
      {
        mutex_lock l(mu_);
        host_tensors_.insert(std::make_pair(node_name, cpu_tensor));
      }

      DeviceContext* device_ctxt = ctx->op_device_context();

      // std::cout << "||| Copying Tensor from GPU to CPU: " << node_name
      //           << ", src = " << src_tensor << "; dst = " << cpu_tensor
      //           << std::endl << std::flush;
      device_ctxt->CopyDeviceTensorToCPU(
          src_tensor, "TensorCopy", device, cpu_tensor,
          [node_name, cpu_tensor, copy_done_cb](const Status& s) {
            std::cout << "||| In CopyDeviceTensorToCPU callback: s.ok() = "
                      << s.ok() << std::endl << std::flush;

            if (s.ok()) {
              // std::cout << "||| After copying, cpu_tensor \"" << node_name
	      // 		<< "\": pointer = " << cpu_tensor
	      // 	        << "; value = " << cpu_tensor->DebugString()
	      // 		<< std::endl << std::flush;
	      copy_done_cb(cpu_tensor);
            } else {
	      // TODO: Log copy failure
	    }
      });

  } else {
    // For CPU tensors, simply copy the reference
    const Tensor* dst_tensor = src_tensor;

    {
      mutex_lock l(mu_);
      host_tensors_.insert(std::make_pair(node_name, dst_tensor));
    }

    copy_done_cb(dst_tensor);
  }
}

class DebugSessionFactory : public SessionFactory {
 public:
  DebugSessionFactory() {}

  bool AcceptsOptions(const SessionOptions& options) override {
    return options.target == "debug";
  }

  Session* NewSession(const SessionOptions& options) override {
    // Must do this before the CPU allocator is created.
    if (options.config.graph_options().build_cost_model() > 0) {
      EnableCPUAllocatorFullStats(true);
    }
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
