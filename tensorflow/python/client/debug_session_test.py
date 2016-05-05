# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for Debugger Session."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.client import debugger
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


# NOTE(mrry): Dummy shape registration for op used in the tests.
ops.RegisterShape("ConstructionFails")(None)


class DebugSessionTest(test_util.TensorFlowTestCase):

  def _auto_step(self, debug_round, do_inspect=True, val_replace=None):
    """Automatically step through a debug session, with options.

    Because _auto_step uses step(), it is not affected by breakpoints in the
    debug_round object.

    Args:
      debug_round: a DebugRound object.
      do_inspect: Inspect the values during stepping, this will lead to a return
        value that equals the result of the execution.
      val_replace: a dictionary for node value injection. The keys are the node
        names. The values are callables that take one input argument (old node
        value) and returns a new node value that is injected to the node
        specified by the corresponding dict key once the node has just finished
        executing.

    Returns:
      If do_inspect == True, the result of the graph execution.

    Raises:
      ValueError: If val_replace is specified by do_inspect is False.
    """

    if not do_inspect and val_replace is not None:
      raise ValueError("val_replace cannot be performed if do_inspect is set "
                       "to False")

    result = None
    while True:
      debug_round.step()

      node_order = debug_round.query_node_order()
      node_idx = debug_round.where()
      is_complete = debug_round.is_complete()

      node_just_completed = node_order[node_idx]

      if do_inspect:
        node_val = debug_round.inspect_value(node_just_completed)
        if node_val is not None:
          result = node_val

        if val_replace is not None and node_just_completed in val_replace:
          replace_func = val_replace[node_just_completed]
          new_val = replace_func(node_val)

          debug_round.inject_value(new_val)

      if is_complete:
        debug_round.step()
        break

    return result

  def testConstantAddingSingleSteps(self):
    with session.Session("debug") as debug_sess:
      a = constant_op.constant(6.0, shape=[1, 1], name="tphass_a")
      b = constant_op.constant(7.0, shape=[1, 1], name="tphass_b")
      s = math_ops.add(a, b, name="tphass_s")

      # Create a DebugRound object
      debug_round = debugger.DebugRound(debug_sess, s)

      node_order = debug_round.query_node_order()
      self.assertTrue(isinstance(node_order, list))
      num_nodes = len(node_order)

      curr_pos = debug_round.where()
      self.assertEquals(0, curr_pos)

      while True:
        debug_round.step()

        # Verify that stepping causes the "where index" to increment properly
        node_idx = debug_round.where()
        self.assertEquals(curr_pos + 1, node_idx)
        curr_pos = node_idx

        # Verify inspect_value returns correct values
        if node_order[curr_pos] == "tphass_a":
          node_value = debug_round.inspect_value("tphass_a")
          self.assertAllClose(np.array([[6.0]]), node_value)
        elif node_order[curr_pos] == "tphass_b":
          node_value = debug_round.inspect_value("tphass_b")
          self.assertAllClose(np.array([[7.0]]), node_value)
        elif node_order[curr_pos] == "tphass_s":
          node_value = debug_round.inspect_value("tphass_s")
          self.assertAllClose(np.array([[13.0]]), node_value)

        # Verify is_complete
        is_complete = debug_round.is_complete()
        self.assertEquals(curr_pos == num_nodes - 1, is_complete)

        node_just_completed = node_order[node_idx]

        if is_complete:
          debug_round.step()
          break

  def testConstantAddingMultiSteps(self):
    with session.Session("debug") as debug_sess:
      a = constant_op.constant(6.0, shape=[1, 1], name="a")
      b = constant_op.constant(7.0, shape=[1, 1], name="b")
      s = math_ops.add(a, b, name="s")

      # Create a DebugRound object
      debug_round = debugger.DebugRound(debug_sess, s)

      node_order = debug_round.query_node_order()
      self.assertTrue(isinstance(node_order, list))
      num_nodes = len(node_order)

      curr_pos = debug_round.where()
      self.assertEquals(0, curr_pos)

      while True:
        debug_round.step(2)

        # Verify that stepping causes the "where index" to increment properly
        node_idx = debug_round.where()
        if curr_pos + 2 >= num_nodes:
          self.assertEquals(num_nodes - 1, node_idx)
        else:
          self.assertEquals(curr_pos + 2, node_idx)
        curr_pos = node_idx

        # Verify is_complete
        is_complete = debug_round.is_complete()
        self.assertEquals(curr_pos == num_nodes - 1, is_complete)

        if is_complete:
          debug_round.step()
          break

  def testConstantAddingContinue(self):
    with session.Session("debug") as debug_sess:
      a = constant_op.constant(6.0, shape=[1, 1], name="a")
      b = constant_op.constant(7.0, shape=[1, 1], name="b")
      s = math_ops.add(a, b, name="s")

      # Create a DebugRound object
      debug_round = debugger.DebugRound(debug_sess, s)

      node_order = debug_round.query_node_order()
      self.assertTrue(node_order.count("s") == 1)

      # Continue until node "s" has just finished executing
      debug_round.cont("s")

      # Verify that the debug breaks on "s"
      self.assertEquals(node_order.index("s"), debug_round.where())

      self._auto_step(debug_round)

  def testConstantAddingContinueToEnd(self):
    with session.Session("debug") as debug_sess:
      a = constant_op.constant(6.0, shape=[1, 1], name="a")
      b = constant_op.constant(7.0, shape=[1, 1], name="b")
      s = math_ops.add(a, b, name="s")

      # Create a DebugRound object
      debug_round = debugger.DebugRound(debug_sess, s)

      # Calling cont() without node_name specified should let the debug round
      # continue to the end
      debug_round.cont()

      # Verify that the debug breaks on the last node
      self.assertEquals(len(debug_round.query_node_order()) - 1,
                        debug_round.where())

      self._auto_step(debug_round)

  def testConstantAddingWithInjection(self):
    with session.Session("debug") as debug_sess:
      a = constant_op.constant(np.array([[6.0]]).astype(np.float32),
                               name="phawi_a")
      b = constant_op.constant(np.array([[7.0]]).astype(np.float32),
                               name="phawi_b")
      s = math_ops.add(a, b, name="phawi_s")

      # Create a DebugRound object
      debug_round = debugger.DebugRound(debug_sess, s)
      node_order = debug_round.query_node_order()
      num_nodes = len(node_order)
      self.assertEquals(0, debug_round.where())
      curr_pos = 0

      while True:
        debug_round.step()

        # Verify that stepping causes the "where index" to increment properly
        node_idx = debug_round.where()
        self.assertEquals(curr_pos + 1, node_idx)
        curr_pos = node_idx

        # Verify inspect_value returns correct values
        if node_order[curr_pos] == "phawi_a":
          node_value = debug_round.inspect_value("phawi_a")
          self.assertAllClose(np.array([[6.0]]), node_value)

          debug_round.inject_value(np.array([[60.0]]).astype(np.float32))
        elif node_order[curr_pos] == "phawi_b":
          node_value = debug_round.inspect_value("phawi_b")
          self.assertAllClose(np.array([[7.0]]), node_value)

          debug_round.inject_value(np.array([[70.0]]).astype(np.float32))
        elif node_order[curr_pos] == "phawi_s":
          node_value = debug_round.inspect_value("phawi_s")

          # The sum should reflect the two newly injected values
          self.assertAllClose(np.array([[130.0]]).astype(np.float32),
                              node_value)

        # Verify is_complete
        is_complete = debug_round.is_complete()
        self.assertEquals(curr_pos == num_nodes - 1, is_complete)

        if is_complete:
          debug_round.step()
          break

  def testPlaceHolderFeed(self):
    with session.Session("debug") as debug_sess:
      a = array_ops.placeholder(shape=[2], dtype=np.float32, name="ph_A")
      b = array_ops.placeholder(shape=[2], dtype=np.float32, name="ph_B")
      s = math_ops.add(a, b, name="ph_s")

      a_feed = np.array([10.0, 20.0]).astype(np.float32)
      b_feed = np.array([30.0, 40.0]).astype(np.float32)
      feed = {
          a: a_feed,
          b: b_feed
      }

      debug_round = debugger.DebugRound(debug_sess, s, feed=feed)

      result = self._auto_step(debug_round)
      self.assertAllClose(a_feed + b_feed, result)

  def testVariablesWithInjection(self):
    with session.Session("debug") as debug_sess:
      a0 = np.array([[10.0]]).astype(np.float32)
      b0 = np.array([[20.0]]).astype(np.float32)

      a = variables.Variable(a0, name="vwi_a")
      b = variables.Variable(b0, name="vwi_b")

      aa = a.assign_add(b)

      # Initialize variables
      init_a = a.initializer
      debug_round = debugger.DebugRound(debug_sess, init_a)
      self._auto_step(debug_round, do_inspect=False)

      init_b = b.initializer
      debug_round = debugger.DebugRound(debug_sess, init_b)
      self._auto_step(debug_round, do_inspect=False)

      # Perform calculation
      debug_round = debugger.DebugRound(debug_sess, aa)
      self._auto_step(debug_round)

      # Get the updated value of a
      debug_round = debugger.DebugRound(debug_sess, a)
      result = self._auto_step(debug_round)

      # The new value of a should now be a0 + b0, due to the assign_add op
      self.assertAllClose(a0 + b0, result)

      # Do it twice to test repeated value injection to the same node
      for _ in xrange(2):
        # Now, run the assign_add op again, but replace a with the old (initial)
        # value.
        def inject_a(_):
          return a0
        injection = {"vwi_a": inject_a}

        debug_round = debugger.DebugRound(debug_sess, aa)
        result = self._auto_step(debug_round, val_replace=injection)

        # Get the updated value of a again
        debug_round = debugger.DebugRound(debug_sess, a)
        result = self._auto_step(debug_round)

        # Note: If it were not for the value injection, this would be equal to
        # a0 + 2 * b0 or a0 + 3 * b0 by now.
        self.assertAllClose(a0 + b0, result)

  def testNodeBreakpoint(self):
    with session.Session("debug") as debug_sess:
      m = constant_op.constant(
          np.array([[1.0, 2.0], [3.0, 4.0]]).astype(np.float32),
          name="nbp_M")
      mt = array_ops.transpose(m, name="nbp_Mt")

      debug_round = debugger.DebugRound(debug_sess, mt)

      node_order = debug_round.query_node_order()
      self.assertTrue(1, node_order.count("nbp_M"))

      # Insert a breakpoint after nbp_M
      bp_handle = debug_round.break_after("nbp_M")

      # Verify breakpoint getter
      node_bps, pred_bps = debug_round.get_breakpoints()
      self.assertTrue("nbp_M" in node_bps)
      self.assertEquals(["nbp_M"], node_bps["nbp_M"])
      self.assertEquals({}, pred_bps)

      # cont() without arg (toward the end) should break at nbp_M
      debug_round.cont()
      self.assertEquals("nbp_M", node_order[debug_round.where()])

      # Finish the rest of the execution (if any)
      result = self._auto_step(debug_round)

      self.assertAllClose(np.array([[1.0, 3.0], [2.0, 4.0]]).astype(np.float32),
                          result)

  def testBeforeNodeBreakpoint(self):
    with session.Session("debug") as debug_sess:
      m = constant_op.constant(
          np.array([[1.0, 2.0], [3.0, 4.0]]).astype(np.float32),
          name="bfnbp_M")
      mt = array_ops.transpose(m, name="bfnbp_Mt")

      debug_round = debugger.DebugRound(debug_sess, mt)

      node_order = debug_round.query_node_order()
      self.assertEquals(1, node_order.count("bfnbp_Mt"))

      # Insert a breakpoint before bfnbp_Mt
      bp_handle = debug_round.break_before("bfnbp_Mt")

      # Verify breakpoint getter
      node_bps, pred_bps = debug_round.get_breakpoints()
      self.assertTrue(bp_handle in node_bps)
      self.assertEquals([bp_handle], node_bps[bp_handle])
      self.assertEquals({}, pred_bps)

      debug_round.cont()

      # Verify that the debug round has broken at the node before bfnbp_Mt
      self.assertEquals(node_order.index("bfnbp_Mt") - 1, debug_round.where())

      # Finish the rest of the execution (if any)
      self._auto_step(debug_round)

  def testInvalidNodeBreakpoint(self):
    with session.Session("debug") as debug_sess:
      m = constant_op.constant(
          np.array([[1.0, 2.0], [3.0, 4.0]]).astype(np.float32),
          name="inbp_M")
      mt = array_ops.transpose(m, name="inbp_Mt")

      debug_round = debugger.DebugRound(debug_sess, mt)
      node_order = debug_round.query_node_order()

      with self.assertRaisesRegexp(ValueError, "does not exist"):
        debug_round.break_after("foo_bar_qux_baz")

      # Verify breakpoint getter
      node_bps, pred_bps = debug_round.get_breakpoints()
      self.assertEquals({}, node_bps)
      self.assertEquals({}, pred_bps)

      # There is no valid breakpoint, so cont() should go till the end
      debug_round.cont()
      self.assertEquals(len(node_order) - 1, debug_round.where())

      # Finish the rest of the execution (if any)
      self._auto_step(debug_round)

  def testPredBreakpoint(self):
    with session.Session("debug") as debug_sess:
      a = constant_op.constant(np.array(11.0).astype(np.float32),
                               name="pbp_a")
      b = constant_op.constant(np.array(22.0).astype(np.float32),
                               name="pbp_b")
      s = math_ops.add(a, b, name="pbp_s")

      # This predicate is not expected to be met
      def pred1(_, node_val):
        return node_val > 5.0 and node_val < 6.0

      # This predicate is expected to be met after b and s
      def pred2(_, node_val):
        return node_val > 20.0

      debug_round = debugger.DebugRound(debug_sess, s)
      node_order = debug_round.query_node_order()

      bp_handle_1 = debug_round.break_if(pred1)
      bp_handle_2 = debug_round.break_if(pred2)

      # Verify breakpoint getter
      node_bps, pred_bps = debug_round.get_breakpoints()
      self.assertEquals({}, node_bps)
      self.assertEquals(2, len(pred_bps))
      self.assertTrue(bp_handle_1 in pred_bps)
      self.assertTrue(bp_handle_2 in pred_bps)

      # First, the debug round should break at pbp_b
      debug_round.cont()
      self.assertEquals("pbp_b", node_order[debug_round.where()])

      # Second, the debug round should break at pbp_s
      debug_round.cont()
      self.assertEquals("pbp_s", node_order[debug_round.where()])
      s_val = debug_round.inspect_value("pbp_s")

      # Finish the rest of the execution (if any)
      self._auto_step(debug_round)
      self.assertAllClose(np.array(33.0).astype(np.float32), s_val)

  def testPredBreakpointRemoval(self):
    with session.Session("debug") as debug_sess:
      a = constant_op.constant(np.array(11.0).astype(np.float32),
                               name="pbpr_a")
      b = constant_op.constant(np.array(22.0).astype(np.float32),
                               name="pbpr_b")
      s = math_ops.add(a, b, name="pbpr_s")

      # This predicate is not expected to be met
      def pred1(_, node_val):
        return node_val > 5.0 and node_val < 6.0

      # This predicate is expected to be met after b and s
      def pred2(_, node_val):
        return node_val > 20.0

      debug_round = debugger.DebugRound(debug_sess, s)
      node_order = debug_round.query_node_order()

      bp_handle_1 = debug_round.break_if(pred1)
      bp_handle_2 = debug_round.break_if(pred2)

      # Remove pred2. This should lead to no breaking in debug round's cont().
      debug_round.remove_breakpoint(bp_handle_2)

      # Verify breakpoint getter
      node_bps, pred_bps = debug_round.get_breakpoints()
      self.assertEquals({}, node_bps)
      self.assertEquals(1, len(pred_bps))
      self.assertTrue(bp_handle_1 in pred_bps)
      self.assertFalse(bp_handle_2 in pred_bps)

      debug_round.cont()
      self.assertEquals(len(node_order) - 1, debug_round.where())

      # Finish the rest of the execution (if any)
      self._auto_step(debug_round)

  def testMultipleNumTimes(self):
    with session.Session("debug") as debug_sess:
      a0 = np.array([[10.0]]).astype(np.float32)
      b0 = np.array([[20.0]]).astype(np.float32)

      a = variables.Variable(a0, name="mnt_a")
      b = variables.Variable(b0, name="mnt_b")

      aa = a.assign_add(b)

      # Initialize variables
      init_a = a.initializer
      debug_round = debugger.DebugRound(debug_sess, init_a)
      self._auto_step(debug_round, do_inspect=False)

      init_b = b.initializer
      debug_round = debugger.DebugRound(debug_sess, init_b)
      self._auto_step(debug_round, do_inspect=False)

      # Perform calculation
      num_times = 2
      debug_round = debugger.DebugRound(debug_sess, aa, num_times=num_times)
      self.assertEquals(num_times, debug_round.get_num_times())

      # Verify that the node_order consists of repetition prefixes
      node_order = debug_round.query_node_order()
      curr_rep_idx = 0
      for node_name in node_order:
        rep_idx = int(node_name.split("_")[0])
        self.assertTrue(rep_idx >= 0 and rep_idx < num_times)

        if rep_idx > curr_rep_idx:
          self.assertEquals(1 + curr_rep_idx, rep_idx)
          curr_rep_idx += 1
        else:
          self.assertEquals(curr_rep_idx, rep_idx)

      self.assertEquals("0__SOURCE", node_order[0])
      self.assertEquals("%d__SINK" % (num_times - 1), node_order[-1])

      # Continuing to an indexed node (e.g., 0_mnt_b) should
      # First make sure that the ordering is what we think it is
      self.assertTrue(node_order.index("0_mnt_a") <
                      node_order.index("0_mnt_b"))

      debug_round.cont("0_mnt_a")
      self.assertEquals("0_mnt_a", node_order[debug_round.where()])
      self.assertEquals(0, debug_round.get_repetition_index())

      # Continuing to a non-indexed node should work
      debug_round.cont("mnt_b")
      self.assertEquals("0_mnt_b", node_order[debug_round.where()])
      self.assertEquals(0, debug_round.get_repetition_index())

      # Attempt to continue to a node without rep prefix should lead to an
      # exception
      with self.assertRaisesRegexp(ValueError,
                                   "has already finished executing"):
        debug_round.cont("mnt_a")

      # Continuing to prefixed node with a rep index different from the
      # current one should work
      debug_round.cont("1_mnt_b")
      self.assertEquals("1_mnt_b", node_order[debug_round.where()])
      self.assertEquals(1, debug_round.get_repetition_index())

      # Continuing to a nonexistent repetition number should lead to an
      # exception
      with self.assertRaisesRegexp(ValueError,
                                   "does not exist in the node order"):
        debug_round.cont("123_mnt_a")
      self.assertEquals(1, debug_round.get_repetition_index())

      self._auto_step(debug_round)

      # Examine the value after two runs
      debug_round = debugger.DebugRound(debug_sess, a)
      a_result = self._auto_step(debug_round)
      self.assertAllClose(a0 + num_times * b0, a_result)

  def testMultipleNumTimesBreakpoints(self):
    with session.Session("debug") as debug_sess:
      a0 = np.array([[10.0]]).astype(np.float32)
      b0 = np.array([[20.0]]).astype(np.float32)

      a = variables.Variable(a0, name="mntbp_a")
      b = variables.Variable(b0, name="mntbp_b")

      aa = a.assign_add(b)

      # Initialize variables
      init_a = a.initializer
      debug_round = debugger.DebugRound(debug_sess, init_a)
      self._auto_step(debug_round, do_inspect=False)

      init_b = b.initializer
      debug_round = debugger.DebugRound(debug_sess, init_b)
      self._auto_step(debug_round, do_inspect=False)

      # Perform calculation
      num_times = 2
      debug_round = debugger.DebugRound(debug_sess, aa, num_times=num_times)
      self.assertEquals(num_times, debug_round.get_num_times())

      node_order = debug_round.query_node_order()

      # Invalid non-prefixed node name for breakpoint
      with self.assertRaisesRegexp(ValueError, "does not exist in the subgraph"):
        debug_round.break_after("mnt_c")

      # Invalid prefixed node name for breakpoint
      with self.assertRaisesRegexp(ValueError, "does not exist in the subgraph"):
        debug_round.break_after("%d_mnt_a" % num_times)

      node_bps, pred_bps = debug_round.get_breakpoints()
      self.assertEquals({}, node_bps)
      self.assertEquals({}, pred_bps)

      # Valid non-prefixed node name
      debug_round.break_after("mntbp_a")

      # Valid prefixed node name
      debug_round.break_after("0_mntbp_b")

      node_bps, pred_bps = debug_round.get_breakpoints()
      self.assertEquals(2, len(node_bps))
      self.assertTrue("mntbp_a" in node_bps)
      self.assertEquals(["0_mntbp_a", "1_mntbp_a"], node_bps["mntbp_a"])
      self.assertTrue("0_mntbp_b" in node_bps)
      self.assertEquals(["0_mntbp_b"], node_bps["0_mntbp_b"])

      # First continue should hit mntbp_a in the 1st repetition
      debug_round.cont()
      self.assertEquals("0_mntbp_a", node_order[debug_round.where()])

      # Then we should hit mntbp_b in the 1st repeition
      debug_round.cont()
      self.assertEquals("0_mntbp_b", node_order[debug_round.where()])

      # Remove the non-prefixed breakpoint
      debug_round.remove_breakpoint("mntbp_a")

      node_bps, pred_bps = debug_round.get_breakpoints()
      self.assertEquals(1, len(node_bps))
      self.assertTrue("0_mntbp_b" in node_bps)
      self.assertEquals(["0_mntbp_b"], node_bps["0_mntbp_b"])

      # Since the breakpoint is now removed, we should be able to continue to
      # the end without breaking
      debug_round.cont()
      self.assertEquals(len(node_order) - 1, debug_round.where())

      self._auto_step(debug_round)

      # Examine the value after two runs
      debug_round = debugger.DebugRound(debug_sess, a)
      a_result = self._auto_step(debug_round)
      self.assertAllClose(a0 + num_times * b0, a_result)


if __name__ == "__main__":
  googletest.main()
