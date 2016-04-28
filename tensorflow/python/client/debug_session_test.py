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

from tensorflow.core.lib.core import error_codes_pb2
from tensorflow.core.protobuf import config_pb2
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


class DebugRound(object):
  """Debug round class.

  A DebugRound object is a handle to a round of debugging. The round of
  debgging is created by specifying a graph node to execute and the number of
  times it should be executed for. The DebugRound object is stateful and keeps
  track of node order, the list of completed nodes, breakpoint, etc.
  """

  def __init__(self, debug_sess, node, num_times=1, feed=None):
    """DebugRound constructor.

    Args:
      debug_sess: A tf.Session object of the "debug" type.
      node: Name of the graph node to execute (evaluate).
      num_times: Number of times to execute the node for (default: 1).

    Raises:
      ValueError: If the supplied debug_sess does not have sess_str == "debug"

    """
    # TODO(cais): Proper mutex locking in C++, so that this can be made 0
    self._init_delay_sec = 0.1
    self._step_delay_sec = 0.02

    if debug_sess.sess_str != "debug":
      raise ValueError(
          "Failed attempt to create a DebugRound from a non-debug session "
          "object")

    if num_times != 1:
      raise ValueError("num_times > 1 has not been implemented yet")

    self._sess = debug_sess

    # Start the debugger main thread
    self._main_thr = self._startDebugMainThread(node, feed=feed)

    where_output = self._sess.debug("where")
    self._node_order = where_output["completed_nodes"] + \
                       where_output["remaining_nodes"]

  def _startDebugMainThread(self, node, feed=None):
    def target_func():
      self._sess.run(node, feed_dict=feed)

    # Start the thread for the debug run. This is the thread that executes the
    # evaluated subgraph. It communicates with the control thread to step /
    # break / continue, etc.
    thr = threading.Thread(target=target_func)
    thr.start()

    # TODO(cais): Proper mutex in C++
    time.sleep(self._init_delay_sec)
    return thr

  def query_node_order(self):
    """Queries node order.

    Returns:
      A list containing the names of all nodes in the executed subgraph.
    """
    return self._node_order

  def step(self, num_steps=1):
    """Step in the debugger.

    Args:
      num_steps: Number of steps to step (default: 1)

    Returns:
      debugger output after the stepping
    """
    if num_steps == 1:
      output = self._sess.debug("step")
    elif num_steps > 1:
      output = self._sess.debug("step %d" % num_steps)
    else:
      raise ValueError("Invalid number of steps for stepping: %d" % num_steps)

    time.sleep(self._step_delay_sec)
    return output

  def cont(self, node_name=None):
    """Continue execution till the named node or first breakpoint.

    Args:
      node_name: Name of the node to try to reach (in the absence of
        breakpoints). If no node name is provided, will try to continue to the
        end.

    Returns:
      debugger output after the stepping

    Raises:
      ValueError: If the node doesn't exist in the executed subgraph or if
        the node has already finished executing.
    """

    if node_name is None:
      # If no node name is specified, try to continue to the end.
      node_name = self._node_order[-1]

    # Verify that node_name is in the node order list
    if self._node_order.count(node_name) == 0:
      raise ValueError("Node named '%s' does not exist in the node order list "
                       "of this debug round" % node_name)

    # Verify that the node has not completd yet
    node_idx = self._node_order.index(node_name)
    if node_idx <= self.where():
      raise ValueError("Cannot continue to node named '%s' because that node "
                       "has already finished executing" % node_name)

    while True:
      pos = self.where()
      print("cont(): pos = %d" % pos)
      if pos == node_idx:
        break
      output = self.step()

    return output


  def where(self):
    """Queries the debugger's current position in the node execution order.

    Returns:
      A 0-based integer indicating the current node, i.e., the node that has
        just finished executing.
    """
    curr_node = self._sess.debug("where")["completed_nodes"][-1]
    return self._node_order.index(curr_node)

  def is_complete(self):
    """Queries whether the debug round is completed.
    That is, whether all nodes in the executed subgraph have finished executing.

    Returns:
      A boolean indicating whether the debug round is complete.
    """
    return self.where() == len(self._node_order) - 1

  def join(self):
    """Join the main debug thread."""
    if self._main_thr:
      self._main_thr.join()


class DebugSessionTest(test_util.TensorFlowTestCase):

  def setUp(self):
    # TODO(cais): Proper mutex locking to push down to
    self._init_delay_sec = 0.1
    self._step_delay_sec = 0.02

  def _auto_step(self, debug_round):
    while True:
      debug_round.step()

      node_order = debug_round.query_node_order()
      node_idx = debug_round.where()
      is_complete = debug_round.is_complete()

      node_just_completed = node_order[node_idx]
      print("Node just completed: %s" % node_just_completed)

      if is_complete:
        debug_round.step()
        break

  def testPlaceHolderAddingSingleSteps(self):
    with session.Session("debug") as debug_sess:
      a = constant_op.constant(6.0, shape=[1, 1], name="a")
      b = constant_op.constant(7.0, shape=[1, 1], name="b")
      s = math_ops.add(a, b, name="s")

      # Create a DebugRound object
      debug_round = DebugRound(debug_sess, s)

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
      debug_round = DebugRound(debug_sess, s)

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
      debug_round = DebugRound(debug_sess, s)

      node_order = debug_round.query_node_order()
      self.assertTrue(node_order.count("s") == 1)

      # Continue until node "s" has just finished executing
      debug_round.cont("s")

      # Verify that the debug breaks on "s"
      self.assertEquals(node_order.index("s"), debug_round.where())

      self._auto_step(debug_round)

  def testPlaceHolderAddingContinueToEnd(self):
    with session.Session("debug") as debug_sess:
      a = constant_op.constant(6.0, shape=[1, 1], name="a")
      b = constant_op.constant(7.0, shape=[1, 1], name="b")
      s = math_ops.add(a, b, name="s")

      # Create a DebugRound object
      debug_round = DebugRound(debug_sess, s)

      # Calling cont() without node_name specified should let the debug round
      # continue to the end
      debug_round.cont()

      # Verify that the debug breaks on the last node
      self.assertEquals(len(debug_round.query_node_order()) - 1,
                        debug_round.where())

      self._auto_step(debug_round)


if __name__ == '__main__':
  googletest.main()
