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

import threading
import time

import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.core.lib.core import error_codes_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import debugger
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.util import compat


# NOTE(mrry): Dummy shape registration for op used in the tests.
ops.RegisterShape('ConstructionFails')(None)


class DebugSessionTest(test_util.TensorFlowTestCase):

  def setUp(self):
    # TODO(cais): Proper mutex locking to push down to
    self._init_delay_sec = 0.1
    self._step_delay_sec = 0.02

  def _auto_step(self, debug_round, do_inspect=True, val_replace=None):
    """Automatically step through a debug session, with options.

    Args:
      debug_round: A DebugRound object.
      do_inspect: Inspect the values during stepping, this will lead to a return
        value that equals the result of the execution.
      val_replace: A dictionary for node value injection. The keys are the node
        names. The values are callables that take one input argument (old node
        value) and returns a new node value that is injected to the node
        specified by the corresponding dict key once the node has just finished
        executing.

    Returns:
      If do_inspect == True, the result of the graph execution.
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

          print("Calling inject_value with %s" % repr(new_val))
          debug_round.inject_value(new_val)

      if is_complete:
        debug_round.step()
        break

    return result

  def testPlaceHolderAddingSingleSteps(self):
    with session.Session("debug") as debug_sess:
      a = constant_op.constant(6.0, shape=[1, 1], name="tphass_a")
      b = constant_op.constant(7.0, shape=[1, 1], name="tphass_b")
      s = math_ops.add(a, b, name="tphass_s")

      # Create a DebugRound object
      debug_round = debugger.DebugRound(debug_sess, s)

      node_order = debug_round.query_node_order()
      self.assertTrue(isinstance(node_order, list))
      num_nodes = len(node_order)
      print(node_order)  # DEBUG

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
        print("Node just completed: %s" % node_just_completed)

        if is_complete:
          debug_round.step()
          break

      debug_round.join()

  def testPlaceHolderAddingMultiSteps(self):
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

        node_just_completed = node_order[node_idx]
        print("Node just completed: %s" % node_just_completed)

        if is_complete:
          debug_round.step()
          break

      debug_round.join()

  def testPlaceHolderAddingContinue(self):
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
      debug_round.join()

  def testPlaceHolderAddingContinueToEnd(self):
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
      debug_round.join()

  def testPlaceHolderAddingWithInjection(self):
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

        node_just_completed = node_order[node_idx]
        print("Node just completed: %s" % node_just_completed)

        if is_complete:
          debug_round.step()
          break

      debug_round.join()

  def testVariablesWithInjection(self):
    with session.Session("debug") as debug_sess:
      A0 = np.array([[10.0]]).astype(np.float32)
      B0 = np.array([[20.0]]).astype(np.float32)

      A = variables.Variable(A0, name="vwi_A")
      B = variables.Variable(B0, name="vwi_B")

      aa = A.assign_add(B0)

      # Initialize variables
      init_A = A.initializer
      debug_round = debugger.DebugRound(debug_sess, init_A)
      self._auto_step(debug_round, do_inspect=False)
      debug_round.join()

      init_B = B.initializer
      debug_round = debugger.DebugRound(debug_sess, init_B)
      self._auto_step(debug_round, do_inspect=False)
      debug_round.join()

      # Perform calculation
      debug_round = debugger.DebugRound(debug_sess, aa)
      self._auto_step(debug_round)
      debug_round.join()

      # Get the updated value of A
      debug_round = debugger.DebugRound(debug_sess, A)
      result = self._auto_step(debug_round)

      # The new value of A should now be A0 + B0, due to the assign_add op
      self.assertAllClose(A0 + B0, result)
      debug_round.join()

      # Now, run the assign_add op again, but replace A with the old (initial)
      # value.
      def inject_A(old_val):
        return A0
      injection = {"vwi_A": inject_A}

      debug_round = debugger.DebugRound(debug_sess, aa)
      result = self._auto_step(debug_round, val_replace=injection)

      # Get the updated value of A again
      debug_round = debugger.DebugRound(debug_sess, A)
      result = self._auto_step(debug_round)

      # Note: If it were not for the value injection, this would be equal to
      # A0 + 2 * B0 now.
      self.assertAllClose(A0 + B0, result)

if __name__ == '__main__':
  googletest.main()
