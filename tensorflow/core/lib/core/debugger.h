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

#ifndef TENSORFLOW_CORE_LIB_CORE_DEBUGGER_H_
#define TENSORFLOW_CORE_LIB_CORE_DEBUGGER_H_

#include <string>

namespace tensorflow {

// TensorFlow Debugger request class.
class DebuggerRequest {
 public:
  DebuggerRequest() : command() {}
  DebuggerRequest(const string& command_) : command(command_) {}
  ~DebuggerRequest() {}

  // Command for the debugger
  // The core debugger API supports the following commands:
  //    step: For stepping to the next node in the graph execution rder
  //    where: For querying the current status of the debugging round, i.e.,
  //      what nodes have finished execution and what nodes remain to be
  //      executed
  //    inspect_value: Get the Tensor value on a completed node
  //    inject_value: Modify the Tensor value on a completed node
  //      (see input_tensor below)
  string command;

  // For inject_value
  Tensor input_tensor;
};

// TensorFlow Debugger response class.
class DebuggerResponse {
 public:
  DebuggerResponse()
      : command(),
        is_completed(false),
        all_nodes(),
        completed_nodes(),
        remaining_nodes(),
        has_output_tensor(false) {}
  ~DebuggerResponse() {}

  // Echo of the command in request (e.g., "inspect_value node_a")
  string command;

  // Is this debugger round complete
  bool is_completed;

  // All nodes in the graph
  std::vector<string> all_nodes;

  // List of nodes have finished executing, in the right order
  std::vector<string> completed_nodes;

  // List of nodes that remaining to be executed, in the right order
  std::vector<string> remaining_nodes;

  // Output from inspect_value
  bool has_output_tensor;
  Tensor output_tensor;
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_CORE_STATUS_H_
