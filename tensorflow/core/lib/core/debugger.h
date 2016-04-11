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

#ifndef TENSORFLOW_CORE_LIB_CORE_DEBUGGER_H_
#define TENSORFLOW_CORE_LIB_CORE_DEBUGGER_H_

#include <string>

namespace tensorflow {

class DebuggerRequest {
 public:
  DebuggerRequest()
    : command() {}
  DebuggerRequest(const string& command_)
    : command(command_) {}
  ~DebuggerRequest() {}

  string command;
  Tensor input_tensor;  //TODO(cais): Avoid unnecessary construction
};

class DebuggerResponse {
 public:
  /// Create a success status.
  DebuggerResponse() 
    : command(),
      is_completed(false),
      all_nodes(),
      completed_nodes(),
      remaining_nodes(),
      has_output_tensor(false) {}
  ~DebuggerResponse() {}

  string command;
  bool is_completed;
  std::vector<string> all_nodes;
  std::vector<string> completed_nodes;
  std::vector<string> remaining_nodes;

  bool has_output_tensor;
  Tensor output_tensor;
  

};

} // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_CORE_STATUS_H_
