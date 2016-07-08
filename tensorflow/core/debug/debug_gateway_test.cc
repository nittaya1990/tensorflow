/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/debug/debug_gateway.h"

#include <algorithm>
#include <unordered_map>

#include "tensorflow/core/debug/debug_graph_utils.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace {

DirectSession* CreateSession() {
  SessionOptions options;
  // Turn off graph optimizer so we can observe intermediate node states.
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(OptimizerOptions_Level_L0);

  (*options.config.mutable_device_count())["CPU"] = 2;
  return dynamic_cast<DirectSession*>(NewSession(options));
}

class SessionDebugMinusAXTest : public ::testing::Test {
 public:
  void Initialize(std::initializer_list<float> a_values) {
    Graph graph(OpRegistry::Global());

    Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
    test::FillValues<float>(&a_tensor, a_values);
    Node* a = test::graph::Constant(&graph, a_tensor);
    a->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");
    a_ = a->name();

    Tensor x_tensor(DT_FLOAT, TensorShape({2, 1}));
    test::FillValues<float>(&x_tensor, {1, 1});
    Node* x = test::graph::Constant(&graph, x_tensor);
    x->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:1");
    x_ = x->name();

    // y = A * x
    Node* y = test::graph::Matmul(&graph, a, x, false, false);
    y->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");
    y_ = y->name();

    Node* y_neg = test::graph::Unary(&graph, "Neg", y);
    y_neg_ = y_neg->name();
    y_neg->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:1");

    test::graph::ToGraphDef(&graph, &def_);
  }

  string a_;
  string x_;
  string y_;
  string y_neg_;
  GraphDef def_;
};

class SessionDebugVariableTest : public ::testing::Test {
 public:
  void Initialize() {
    Graph graph(OpRegistry::Global());

    const string kDeviceName = "/job:localhost/replica:0/task:0/cpu:0";

    // Define variable node.
    Node* var = test::graph::Var(&graph, DT_FLOAT, TensorShape({3}));
    var->set_assigned_device_name(kDeviceName);
    var_node_name_ = var->name();

    // Define the initial value and the initial-value node.
    Tensor nan_nan_seven(DT_FLOAT, TensorShape({3}));
    nan_nan_seven.flat<float>()(0) = std::numeric_limits<float>::quiet_NaN();
    nan_nan_seven.flat<float>()(1) = std::numeric_limits<float>::quiet_NaN();
    nan_nan_seven.flat<float>()(2) = 7.0;

    Node* init_val = test::graph::Constant(&graph, nan_nan_seven);
    init_val->set_assigned_device_name(kDeviceName);
    init_val_node_name_ = init_val->name();

    // Define node for variable value initialization
    Node* init = test::graph::Assign(&graph, var, init_val);
    init->set_assigned_device_name(kDeviceName);
    init_node_name_ = init->name();

    // Define new value node
    Tensor nan_eight_eight(DT_FLOAT, TensorShape({3}));
    nan_eight_eight.flat<float>()(0) = std::numeric_limits<float>::quiet_NaN();
    nan_eight_eight.flat<float>()(1) = 8.0;
    nan_eight_eight.flat<float>()(2) = 8.0;

    Node* new_val = test::graph::Constant(&graph, nan_eight_eight);
    new_val->set_assigned_device_name(kDeviceName);
    new_val_node_name_ = new_val->name();

    // Define node for assigning new value
    Node* assign = test::graph::Assign(&graph, var, new_val);
    assign->set_assigned_device_name(kDeviceName);
    assign_node_name_ = assign->name();

    test::graph::ToGraphDef(&graph, &def_);
  }

  string var_node_name_;
  string init_val_node_name_;
  string init_node_name_;
  string new_val_node_name_;
  string assign_node_name_;
  GraphDef def_;
};

TEST_F(SessionDebugMinusAXTest, RunSimpleNetwork) {
  Initialize({3, 2, -1, 0});
  std::unique_ptr<DirectSession> session(CreateSession());
  ASSERT_TRUE(session != nullptr);

  DebugGateway debug_gateway(session.get());

  // Supply completion and value callbacks
  mutex mu;
  // Completed nodes with and without outputs
  std::vector<string> completed_nodes_w_outputs;
  std::vector<string> completed_nodes_wo_outputs;

  Notification callbacks_done;
  debug_gateway.SetNodeCompletionCallback(
      [&mu, &completed_nodes_w_outputs, &completed_nodes_wo_outputs](
          const string& node_name, const bool any_output) {
        mutex_lock l(mu);
        if (any_output) {
          completed_nodes_w_outputs.push_back(node_name);
        } else {
          completed_nodes_wo_outputs.push_back(node_name);
        }
      });

  std::vector<bool> tensors_initialized;
  std::unordered_map<string, Tensor> tensor_vals;
  // output_slot values recorded in value callbacks
  std::vector<int> output_slots_val;
  // is_ref values recorded in value callbacks
  std::vector<bool> is_refs_val;

  debug_gateway.SetNodeValueCallback(
      [this, &mu, &tensors_initialized, &tensor_vals, &output_slots_val,
       &is_refs_val,
       &callbacks_done](const string& node_name, const int output_slot,
                        const Tensor& tensor_value, const bool is_ref) {
        mutex_lock l(mu);
        tensors_initialized.push_back(tensor_value.IsInitialized());
        tensor_vals.insert(std::make_pair(node_name, tensor_value));
        output_slots_val.push_back(output_slot);
        is_refs_val.push_back(is_ref);

        // Set the notification once we have the value from the target node.
        if (node_name == y_neg_ && !callbacks_done.HasBeenNotified()) {
          callbacks_done.Notify();
        }
      });

  TF_ASSERT_OK(session->Create(def_));

  std::vector<std::pair<string, Tensor>> inputs;

  // Request two targets: one fetch output and one non-fetched output.
  std::vector<string> output_names = {y_ + ":0"};
  std::vector<string> target_nodes = {y_neg_};
  std::vector<Tensor> outputs;
  Status s = session->Run(inputs, output_names, target_nodes, &outputs);
  TF_ASSERT_OK(s);

  // Wait for callbacks to complete.
  callbacks_done.WaitForNotification();

  ASSERT_EQ(1, outputs.size());
  // The first output should be initialized and have the correct
  // output.
  auto mat = outputs[0].matrix<float>();
  ASSERT_TRUE(outputs[0].IsInitialized());
  EXPECT_FLOAT_EQ(5.0, mat(0, 0));

  // Verify the calling history of the completion callback
  // The following verifies each node with output(s) invoked the callback
  // exactly once.
  ASSERT_GE(completed_nodes_w_outputs.size(), 4);  // There may be added nodes.

  ASSERT_EQ(1, std::count(completed_nodes_w_outputs.begin(),
                          completed_nodes_w_outputs.end(), a_));
  ASSERT_EQ(1, std::count(completed_nodes_w_outputs.begin(),
                          completed_nodes_w_outputs.end(), x_));
  ASSERT_EQ(1, std::count(completed_nodes_w_outputs.begin(),
                          completed_nodes_w_outputs.end(), y_));
  ASSERT_EQ(1, std::count(completed_nodes_w_outputs.begin(),
                          completed_nodes_w_outputs.end(), y_neg_));

  // Apart from nodes with outputs, there are also no-output (control) nodes.
  // They ought to be captured by the DebugGateway through
  // NodeOutputCallback as well.
  ASSERT_GT(completed_nodes_wo_outputs.size(), 0);

  // The DebugGateway should have captured the _SOURCE and _SINK nodes.
  ASSERT_LE(1, std::count(completed_nodes_wo_outputs.begin(),
                          completed_nodes_wo_outputs.end(), "_SOURCE"));
  ASSERT_LE(1, std::count(completed_nodes_wo_outputs.begin(),
                          completed_nodes_wo_outputs.end(), "_SINK"));

  // Verify the calling history of the value callabck
  ASSERT_EQ(completed_nodes_w_outputs.size(), tensors_initialized.size());

  // In this graph, there is no uninitialized node value.
  ASSERT_EQ(
      tensors_initialized.end(),
      std::find(tensors_initialized.begin(), tensors_initialized.end(), false));

  ASSERT_EQ(completed_nodes_w_outputs.size(), tensor_vals.size());
  ASSERT_EQ(completed_nodes_w_outputs.size(), output_slots_val.size());
  ASSERT_EQ(completed_nodes_w_outputs.size(), is_refs_val.size());

  // Verify the intermediate tensor values captured through the value callback
  auto mat_a = tensor_vals[a_].matrix<float>();
  ASSERT_EQ(3.0, mat_a(0, 0));
  ASSERT_EQ(2.0, mat_a(0, 1));
  ASSERT_EQ(-1.0, mat_a(1, 0));
  ASSERT_EQ(0.0, mat_a(1, 1));

  auto mat_x = tensor_vals[x_].matrix<float>();
  ASSERT_EQ(1.0, mat_x(0, 0));
  ASSERT_EQ(1.0, mat_x(1, 0));

  auto mat_y = tensor_vals[y_].matrix<float>();
  ASSERT_EQ(5.0, mat_y(0, 0));
  ASSERT_EQ(-1.0, mat_y(1, 0));

  auto mat_y_neg = tensor_vals[y_neg_].matrix<float>();
  ASSERT_EQ(-5.0, mat_y_neg(0, 0));
  ASSERT_EQ(1.0, mat_y_neg(1, 0));

  // In this graph, all outputs are on the first slot
  ASSERT_EQ(output_slots_val.size(),
            std::count_if(output_slots_val.begin(), output_slots_val.end(),
                          [](int slot) { return slot == 0; }));

  // In this graph, there is no ref-type tensor.
  ASSERT_EQ(is_refs_val.end(),
            std::find(is_refs_val.begin(), is_refs_val.end(), true));
}

TEST_F(SessionDebugMinusAXTest, RunSimpleNetworkWithTwoDebugNodesInserted) {
  // Tensor contains one count of NaN
  Initialize({3, std::numeric_limits<float>::quiet_NaN(), -1, 0});
  std::unique_ptr<DirectSession> session(CreateSession());
  ASSERT_TRUE(session != nullptr);

  DebugGateway debug_gateway(session.get());

  // Create debug tensor watch options with two debug ops:
  // DebugIdentity and DebugNanCount
  RunOptions run_opts;
  const string debug_identity = "DebugIdentity";
  const string debug_nan_count = "DebugNanCount";
  DebugTensorWatch* tensor_watch_opts = run_opts.add_debug_tensor_watch_opts();
  tensor_watch_opts->set_node_name(a_);
  tensor_watch_opts->set_output_slot(0);
  // "No deep copy" should be implicit
  tensor_watch_opts->add_debug_ops(debug_identity);
  tensor_watch_opts->add_debug_ops(debug_nan_count);

  // Expected name of the inserted debug node
  string debug_identity_node_name =
      DebugNodeInserter::GetDebugNodeName(a_, 0, 0, debug_identity);
  string debug_nan_count_node_name =
      DebugNodeInserter::GetDebugNodeName(a_, 0, 1, debug_nan_count);

  // Supply completion and value callbacks
  mutex mu;
  // Completed nodes with and without outputs
  std::vector<string> completed_debug_nodes;

  Notification callbacks_done;
  debug_gateway.SetNodeCompletionCallback(
      [&mu, &debug_identity_node_name, &debug_nan_count_node_name,
       &completed_debug_nodes](const string& node_name, const bool any_output) {
        mutex_lock l(mu);
        if (any_output && (node_name == debug_identity_node_name ||
                           node_name == debug_nan_count_node_name)) {
          completed_debug_nodes.push_back(node_name);
        }
      });

  std::vector<Tensor> watched_tensor_vals;
  std::vector<Tensor> debug_identity_tensor_vals;
  std::vector<Tensor> debug_nan_count_tensor_vals;

  debug_gateway.SetNodeValueCallback(
      [this, &mu, &debug_identity_node_name, &debug_nan_count_node_name,
       &watched_tensor_vals, &debug_identity_tensor_vals,
       &debug_nan_count_tensor_vals,
       &callbacks_done](const string& node_name, const int output_slot,
                        const Tensor& tensor_value, const bool is_ref) {
        mutex_lock l(mu);
        if (node_name == a_) {
          watched_tensor_vals.push_back(tensor_value);
        } else if (node_name == debug_identity_node_name && output_slot == 1) {
          // output_slot==1 carries the debug signal. Same below.
          debug_identity_tensor_vals.push_back(tensor_value);
        } else if (node_name == debug_nan_count_node_name && output_slot == 1) {
          debug_nan_count_tensor_vals.push_back(tensor_value);
        }

        // Set the notification once we have the value from the target node.
        if (node_name == y_neg_ && !callbacks_done.HasBeenNotified()) {
          callbacks_done.Notify();
        }
      });

  TF_ASSERT_OK(session->Create(def_));

  std::vector<std::pair<string, Tensor>> inputs;

  // Request two targets: one fetch output and one non-fetched output.
  std::vector<string> output_names = {y_ + ":0"};
  std::vector<string> target_nodes = {y_neg_};
  std::vector<Tensor> outputs;

  RunMetadata run_metadata;
  Status s = session->Run(run_opts, inputs, output_names, target_nodes,
                          &outputs, &run_metadata);
  TF_ASSERT_OK(s);

  // Wait for callbacks to complete.
  callbacks_done.WaitForNotification();

  // Verify that each of the two debug nodes has completed exactly once.
  ASSERT_EQ(2, completed_debug_nodes.size());
  ASSERT_EQ(
      1, std::count(completed_debug_nodes.begin(), completed_debug_nodes.end(),
                    debug_identity_node_name));
  ASSERT_EQ(
      1, std::count(completed_debug_nodes.begin(), completed_debug_nodes.end(),
                    debug_nan_count_node_name));

  // Verify that the tensor values from the watched node and the identity
  // debug node are received and they are equal (owing to the debug op being
  // "DebugIdentity")
  ASSERT_EQ(1, watched_tensor_vals.size());
  ASSERT_EQ(1, debug_identity_tensor_vals.size());

  auto mat_a = watched_tensor_vals[0].matrix<float>();
  auto mat_identity = debug_identity_tensor_vals[0].matrix<float>();
  ASSERT_EQ(mat_a(0, 0), mat_identity(0, 0));
  // ASSERT_EQ doesn't work for nan == nan
  ASSERT_TRUE(std::isnan(mat_a(0, 1)));
  ASSERT_TRUE(std::isnan(mat_identity(0, 1)));
  ASSERT_EQ(mat_a(1, 0), mat_identity(1, 0));
  ASSERT_EQ(mat_a(1, 1), mat_identity(1, 1));

  // Verify that the output from the NaN-count debug node indicates exactly
  // one NaN.
  ASSERT_EQ(1, debug_nan_count_tensor_vals.size());
  ASSERT_EQ(1, debug_nan_count_tensor_vals[0].scalar<int64>()());
}

TEST_F(SessionDebugVariableTest, InitializeVariableWithDeepCopyDebugOps) {
  // Tensor contains one count of NaN
  Initialize();
  std::unique_ptr<DirectSession> session(CreateSession());
  ASSERT_TRUE(session != nullptr);

  DebugGateway debug_gateway(session.get());

  TF_ASSERT_OK(session->Create(def_));

  // First run the initialization op
  std::vector<std::pair<string, Tensor>> inputs_init;
  std::vector<Tensor> outputs_init;
  Status s = session->Run(inputs_init, {init_node_name_}, {}, &outputs_init);
  TF_ASSERT_OK(s);

  // Create debug tensor watch options with two ref-type debug ops:
  // DebugRefIdentity and DebugRefNanCount
  RunOptions run_opts;
  const string debug_identity = "DebugRefIdentity";
  const string debug_nan_count = "DebugRefNanCount";
  DebugTensorWatch* tensor_watch_opts = run_opts.add_debug_tensor_watch_opts();
  tensor_watch_opts->set_node_name(var_node_name_);
  tensor_watch_opts->set_output_slot(0);
  // "No deep copy" should be implicit.
  tensor_watch_opts->add_debug_ops(debug_identity);
  tensor_watch_opts->add_debug_ops(debug_nan_count);
  tensor_watch_opts->set_deep_copy(true);

  // Expected name of the inserted debug node
  string debug_identity_node_name =
      DebugNodeInserter::GetDebugNodeName(var_node_name_, 0, 0, debug_identity);
  string debug_nan_count_node_name = DebugNodeInserter::GetDebugNodeName(
      var_node_name_, 0, 1, debug_nan_count);

  // Supply completion and value callbacks
  mutex mu;
  // Completed nodes with and without outputs
  std::vector<string> completed_debug_nodes;

  Notification callbacks_done;
  debug_gateway.SetNodeCompletionCallback(
      [this, &mu, &debug_identity_node_name, &debug_nan_count_node_name,
       &completed_debug_nodes,
       &callbacks_done](const string& node_name, const bool any_output) {
        mutex_lock l(mu);
        if (any_output && (node_name == debug_identity_node_name ||
                           node_name == debug_nan_count_node_name)) {
          completed_debug_nodes.push_back(node_name);
        }
      });

  std::vector<Tensor> debug_identity_tensor_vals;
  std::vector<Tensor> debug_nan_count_tensor_vals;

  debug_gateway.SetNodeValueCallback(
      [this, &mu, &debug_identity_node_name, &debug_nan_count_node_name,
       &debug_identity_tensor_vals, &debug_nan_count_tensor_vals,
       &callbacks_done](const string& node_name, const int output_slot,
                        const Tensor& tensor_value, const bool is_ref) {
        mutex_lock l(mu);
        std::cout << "In node value callback: " << node_name << ":"
                  << output_slot << std::endl;
        if (node_name == debug_identity_node_name && output_slot == 1) {
          // output_slot==1 carries the debug signal. Same below.
          debug_identity_tensor_vals.push_back(tensor_value);
        } else if (node_name == debug_nan_count_node_name && output_slot == 1) {
          debug_nan_count_tensor_vals.push_back(tensor_value);
        }

        // Set the notification once we have the value from the target node.
        if (node_name == assign_node_name_ &&
            !callbacks_done.HasBeenNotified()) {
          callbacks_done.Notify();
        }
      });

  // // Request two targets: one fetch output and one non-fetched output.
  std::vector<std::pair<string, Tensor>> inputs;
  std::vector<string> output_names = {assign_node_name_ + ":0"};
  std::vector<string> target_nodes = {assign_node_name_};
  std::vector<Tensor> outputs;

  // Run with RunOptions that has tensor watches
  RunMetadata run_metadata;
  s = session->Run(run_opts, inputs, output_names, target_nodes, &outputs,
                   &run_metadata);
  TF_ASSERT_OK(s);

  // Wait for callbacks to complete.
  callbacks_done.WaitForNotification();

  // Verify that the update has happened properly.
  ASSERT_EQ(1, outputs.size());
  ASSERT_TRUE(std::isnan(outputs[0].vec<float>()(0)));
  ASSERT_EQ(8.0, outputs[0].vec<float>()(1));  // Expect new value
  ASSERT_EQ(8.0, outputs[0].vec<float>()(2));  // Expect new value

  // Verify that each of the two debug nodes has completed exactly once.
  ASSERT_EQ(2, completed_debug_nodes.size());
  ASSERT_EQ(
      1, std::count(completed_debug_nodes.begin(), completed_debug_nodes.end(),
                    debug_identity_node_name));
  ASSERT_EQ(
      1, std::count(completed_debug_nodes.begin(), completed_debug_nodes.end(),
                    debug_nan_count_node_name));

  // Verify that the values from the ref identity node reflects the value
  // before the new assign.
  ASSERT_EQ(1, debug_identity_tensor_vals.size());

  auto vec_identity = debug_identity_tensor_vals[0].vec<float>();
  ASSERT_TRUE(std::isnan(vec_identity(0)));
  ASSERT_TRUE(std::isnan(vec_identity(1)));
  ASSERT_EQ(7.0, vec_identity(2));

  // Verify that the output from the NaN-count debug node indicates exactly
  // two NaNs, i.e., reflecting the value before the new assign.
  ASSERT_EQ(1, debug_nan_count_tensor_vals.size());
  ASSERT_EQ(2, debug_nan_count_tensor_vals[0].scalar<int64>()());
}

}  // end namespace
}  // end namespace tensorflow
