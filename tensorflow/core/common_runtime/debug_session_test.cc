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
  void Initialize(std::initializer_list<float> a_values,
                  std::initializer_list<float> a_prime_values) {
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

    // Injected alternative value for a
    Tensor a_prime(DT_FLOAT, TensorShape({2, 2}));
    a_prime_tensor = a_prime;
    test::FillValues<float>(&a_prime_tensor, a_prime_values);
  }

  string x_;
  string y_;
  string y_neg_;
  GraphDef def_;
  Tensor a_prime_tensor;
};

TEST_F(DebugSessionMinusAXTest, RunSimpleNetworkStepRightAfterRun) {
  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test", 1);

  Initialize({3, 2, -1, 0}, {-3, -2, 1, 0});
  std::unique_ptr<Session> session(CreateDebugSession());
  ASSERT_TRUE(session != nullptr);

  TF_ASSERT_OK(session->Create(def_));
  std::vector<std::pair<string, Tensor>> inputs;

  std::vector<string> output_names = {y_ + ":0"};
  std::vector<string> target_nodes = {y_neg_};
  std::vector<Tensor> outputs;

  thread_pool->Schedule([&session, &inputs, 
                         &output_names, &target_nodes, &outputs] {
    Status s = session->Run(inputs, output_names, target_nodes, &outputs);
  });

  DebuggerRequest step_request("step");
  DebuggerResponse step_response = session->SendDebugMessage(step_request);

  DebuggerRequest where_request("where");
  DebuggerResponse where_response = session->SendDebugMessage(where_request);

  // Stepping right after the Run() call should lead to two completed nodes
  ASSERT_EQ(2, where_response.completed_nodes.size());

  const int steps_remaining = where_response.remaining_nodes.size();
  for (int i = 0; i < steps_remaining; ++i) {
    session->SendDebugMessage(step_request);
  }

  // Step one last time to finish the debug round
  session->SendDebugMessage(step_request);
  delete thread_pool;  // Wait for all closures to finish.

  // Verify the output the debug Session's Run()
  auto mat = outputs[0].matrix<float>();
  ASSERT_TRUE(outputs[0].IsInitialized());
  EXPECT_FLOAT_EQ(5.0, mat(0, 0));
}

TEST_F(DebugSessionMinusAXTest, RunSimpleNetworkWithInspectionAndInjection) {
  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test", 1);

  Initialize({3, 2, -1, 0}, {-3, -2, 1, 0});
  std::unique_ptr<Session> session(CreateDebugSession());
  ASSERT_TRUE(session != nullptr);

  TF_ASSERT_OK(session->Create(def_));
  std::vector<std::pair<string, Tensor>> inputs;

  std::vector<string> output_names = {y_ + ":0"};
  std::vector<string> target_nodes = {y_neg_};
  std::vector<Tensor> outputs;

  thread_pool->Schedule([&session, &inputs, 
  	                     &output_names, &target_nodes, &outputs] {
  	Status s = session->Run(inputs, output_names, target_nodes, &outputs);
  });

  DebuggerRequest where_request("where");
  DebuggerResponse where_response = session->SendDebugMessage(where_request);

  const int total_num_nodes = where_response.completed_nodes.size() +
  							  where_response.remaining_nodes.size();

  // Record all nodes. Expected node order:
  //   _SOURCE
  //   n/_0 (a)
  //   n/_1 (x)
  //   n/_2 (y: MatMul)
  //   _send_n/_2_0
  //   n/_3
  //   _SINK
  std::vector<string> all_nodes;
  for (const string& node_name : where_response.completed_nodes) {
    all_nodes.push_back(node_name);
  }
  for (const string& node_name : where_response.remaining_nodes) {
    all_nodes.push_back(node_name);
  }

  ASSERT_EQ("_SOURCE", all_nodes[0]);
  ASSERT_EQ("n/_0", all_nodes[1]);
  ASSERT_EQ("n/_1", all_nodes[2]);
  ASSERT_EQ("n/_2", all_nodes[3]);
  ASSERT_EQ("_send_n/_2_0", all_nodes[4]);
  ASSERT_EQ("n/_3", all_nodes[5]);
  ASSERT_EQ("_SINK", all_nodes[6]);

  ASSERT_TRUE(total_num_nodes > 0);
  ASSERT_EQ(1, where_response.completed_nodes.size());
  ASSERT_TRUE(where_response.remaining_nodes.size() > 0);
  ASSERT_FALSE(where_response.is_completed);

  DebuggerRequest step_request("step");
  for (int k = 1; k < total_num_nodes; ++k) {
    DebuggerResponse step_response = session->SendDebugMessage(step_request);

    // Send another "where" request
    where_response = session->SendDebugMessage(where_request);
    ASSERT_EQ(total_num_nodes,
              where_response.completed_nodes.size() +
                  where_response.remaining_nodes.size());

    const int num_completed = where_response.completed_nodes.size();
    const string& curr_node =
        where_response.completed_nodes[num_completed - 1];

    // Verify the progression along the nodes
    ASSERT_EQ(k + 1, where_response.completed_nodes.size());
    for (size_t i = 0; i < where_response.completed_nodes.size(); ++i) {
      ASSERT_EQ(all_nodes[i], where_response.completed_nodes[i]);
    }
    for (size_t i = 0; i < where_response.remaining_nodes.size(); ++i) {
      ASSERT_EQ(all_nodes[i + where_response.completed_nodes.size()], 
                where_response.remaining_nodes[i]);
    }

    // Verify is_completed state
    if (k < total_num_nodes - 1) {
      ASSERT_FALSE(where_response.is_completed);
    } else {
      ASSERT_TRUE(where_response.is_completed);
    }

    // Inspect value request
    if (curr_node == "n/_0") {  // a
      DebuggerRequest inspect_request("print n/_0");
      DebuggerResponse inspect_response =
          session->SendDebugMessage(inspect_request);
      const Tensor& val = inspect_response.output_tensor;
      ASSERT_TRUE(val.shape() == TensorShape({2, 2}));

      auto mat = val.matrix<float>();
      EXPECT_FLOAT_EQ(3.0, mat(0, 0));
      EXPECT_FLOAT_EQ(2.0, mat(0, 1));
      EXPECT_FLOAT_EQ(-1.0, mat(1, 0));
      EXPECT_FLOAT_EQ(0.0, mat(1, 1));

      // Inject new value to a
      DebuggerRequest inject_request("inject_value n/_0");
      inject_request.input_tensor = a_prime_tensor;

      DebuggerResponse inject_response =
          session->SendDebugMessage(inject_request);
    } else if (curr_node == "n/_1") {  // x
      DebuggerRequest inspect_request("print n/_1");
      DebuggerResponse inspect_response =
          session->SendDebugMessage(inspect_request);
      const Tensor& val = inspect_response.output_tensor;
      ASSERT_TRUE(val.shape() == TensorShape({2, 1}));

      auto mat = val.matrix<float>();
      EXPECT_FLOAT_EQ(1.0, mat(0, 0));
      EXPECT_FLOAT_EQ(1.0, mat(1, 0));
    } else if (curr_node == "n/_2") {  // MatMul: a * x
      DebuggerRequest inspect_request("print n/_2");
      DebuggerResponse inspect_response =
          session->SendDebugMessage(inspect_request);
      const Tensor& val = inspect_response.output_tensor;
      ASSERT_TRUE(val.shape() == TensorShape({2, 1}));

      auto mat = val.matrix<float>();
      EXPECT_FLOAT_EQ(-5.0, mat(0, 0));  // Without injection, this would be 5.0
      EXPECT_FLOAT_EQ(1.0, mat(1, 0));  // Without injection, this would be -1.0
    }
  }

  // Step one last time to finish the debug round
  session->SendDebugMessage(step_request);
  delete thread_pool;  // Wait for all closures to finish.

  // Verify the output the debug Session's Run()
  auto mat = outputs[0].matrix<float>();
  ASSERT_TRUE(outputs[0].IsInitialized());
  EXPECT_FLOAT_EQ(-5.0, mat(0, 0));  // Without injection, this would be 5.0
}

}  // namespace
}  // namespace tensorflow