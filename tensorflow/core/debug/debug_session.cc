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

#include <chrono>  // TODO(cais): Replace chrono
#include <sstream>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

DebugSession::DebugSession(const SessionOptions& options,
                           const DeviceMgr* device_mgr)
    : DirectSession(options, device_mgr) {
  // Disable the graph optimization by default
  SetOptimizeGraph(false);

  // Supply node output callback
  SetNodeOutputsCallback([this](const string& node_name,
                                const int output_slot,
                                const Tensor* tensor,
                                const bool is_ref,
                                OpKernelContext* ctx) {
    std::ostringstream stream;

    std::chrono::milliseconds ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch());
    int64 epoch_timestamp = ms.count();

    if (comp_cb_ != nullptr) {
      comp_cb_(node_name, output_slot, epoch_timestamp, is_ref);
    }

    // DEBUG
    stream << "(" << epoch_timestamp
           << ") node_outputs_cb from debug session: "
           << node_name << std::endl;
    stream << "  is_ref: " << is_ref << std::endl;
    stream << "  shape: " << tensor->shape().DebugString() << std::endl;
    stream << "  dtype: " << tensor->dtype() << std::endl;

    Device* device = static_cast<Device*>(ctx->device());
    AllocatorAttributes alloc_attrs = ctx->output_alloc_attr(output_slot);

    const bool is_tensor_initialized = tensor->IsInitialized();

    // Omit uniniatizlied Tensors.
    // Copying non-ref Tensors from GPU often fails. Limiting the copying to
    // ref Tensors for now.
    if (device->name().find("gpu:") != string::npos &&
        !alloc_attrs.on_host() && is_tensor_initialized && is_ref) {
      stream << "  device: " << device->name() << std::endl;

      Allocator* cpu_allocator = tensorflow::cpu_allocator();
      Tensor* cpu_tensor = new Tensor(cpu_allocator,
                                      tensor->dtype(),
                                      tensor->shape());
      // TODO(cais): Delete to prevent memory leak.

      DeviceContext* device_ctxt = ctx->op_device_context();

      bool copy_done = false;

      std::cout << "||| Copying Tensor from GPU to CPU: " << node_name
                << ", src = " << tensor << "; dst = " << cpu_tensor
                << std::endl << std::flush;
      device_ctxt->CopyDeviceTensorToCPU(
          tensor, "TensorCopy", device, cpu_tensor,
          [&copy_done, &node_name, &cpu_tensor](const Status& s) {
            copy_done = true;
            // std::cout << "CopyDeviceTensorToCPU: s.ok() = " << s.ok()
            //           << std::endl << std::flush;

          if (s.ok()) {
            std::cout << "||| After copying, cpu_tensor \"" << node_name
                      << "\" = " << cpu_tensor
                      << "; data type: " << DataTypeString(cpu_tensor->dtype())
                      << "; shape: " << cpu_tensor->shape().DebugString()
                      << "; value: " << cpu_tensor->DebugString()
                      << std::endl << std::flush;
          }
      });

      while (!copy_done) {
        Env::Default()->SleepForMicroseconds(1 * 1000);
      }


    } else {
        stream << "  val of " << node_name << " (on cpu): "
               << tensor->DebugString() << std::endl;
        // state_dumper_.dump(stream.str(), node_name, tensor);
        std::cout << stream.str() << std::endl << std::flush;

        if (val_cb_ != nullptr) {
          val_cb_(node_name, output_slot, *tensor, is_ref);
        }
    }

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
