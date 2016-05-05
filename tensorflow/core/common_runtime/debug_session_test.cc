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

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/debugger.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace {

Session* CreateDebugSession() {
  SessionOptions options;
  // (*options.config.mutable_device_count())["CPU"] = 1;
  options.target = "debug";
  return NewSession(options);
}

class DebugSessionMinusAXTest : public ::testing::Test {
 public:
  void Initialize(std::initializer_list<float> a_values) {
    Graph graph(OpRegistry::Global());

    Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
    test::FillValues<float>(&a_tensor, a_values);
    Node* a = test::graph::Constant(&graph, a_tensor);
    a->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");

    Tensor x_tensor(DT_FLOAT, TensorShape({2, 1}));
    test::FillValues<float>(&x_tensor, {1, 1});
    Node* x = test::graph::Constant(&graph, x_tensor);
    x->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");
    x_ = x->name();

    // y = A * x
    Node* y = test::graph::Matmul(&graph, a, x, false, false);
    y->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");
    y_ = y->name();

    Node* y_neg = test::graph::Unary(&graph, "Neg", y);
    y_neg_ = y_neg->name();
    y_neg->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");

    test::graph::ToGraphDef(&graph, &def_);
  }

  string x_;
  string y_;
  string y_neg_;
  GraphDef def_;
};

TEST_F(DebugSessionMinusAXTest, RunSimpleNetwork) {
  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test", 1);

  Initialize({3, 2, -1, 0});
  std::unique_ptr<Session> session(CreateDebugSession());
  ASSERT_TRUE(session != nullptr);

  TF_ASSERT_OK(session->Create(def_));
  std::vector<std::pair<string, Tensor>> inputs;

  std::vector<string> output_names = {y_ + ":0"};
  std::vector<string> target_nodes = {y_neg_};
  std::vector<Tensor> outputs;

  std::cout << "*** Calling Schedule()" << std::endl;  // DEBUG
  thread_pool->Schedule([&session, &inputs, 
  	                     &output_names, &target_nodes, &outputs] {
  	std::cout << "*** Calling session->Run()" << std::endl;  // DEBUG
  	Status s = session->Run(inputs, output_names, target_nodes, &outputs);
	std::cout << "*** Returned from session->Run()" << std::endl;  // DEBUG
  	// TF_ASSERT_OK(s);
  });
  
  const int microsec_to_sleep = 10 * 1000;
  // const int microsec_to_sleep = 0;

  // TODO(cais): Remove sleep
  // Env::Default()->SleepForMicroseconds(microsec_to_sleep);

  DebuggerRequest request("where");
  std::cout << "*** Calling session->SendDebugMessage()" << std::endl;  // DEBUG
  DebuggerResponse response = session->SendDebugMessage(request);

  std::cout << "responses.is_completed = " << response.is_completed << std::endl;
  std::cout << "responses.completed_nodes.size() = " << response.completed_nodes.size() << std::endl;
  std::cout << "responses.remaining_nodes.size() = " << response.remaining_nodes.size() << std::endl;
  ASSERT_TRUE(response.completed_nodes.size() > 0);
  ASSERT_TRUE(response.remaining_nodes.size() > 0);
  ASSERT_FALSE(response.is_completed);

  // ASSERT_EQ(1, outputs.size());
  // // The first output should be initialized and have the correct
  // // output.
  // auto mat = outputs[0].matrix<float>();
  // ASSERT_TRUE(outputs[0].IsInitialized());
  // EXPECT_FLOAT_EQ(5.0, mat(0, 0));
}

}  // namespace
}  // namespace tensorflow