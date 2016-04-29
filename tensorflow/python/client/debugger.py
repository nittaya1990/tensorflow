# Copyright 2016 Google Inc. All Rights Reserved.
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

"""A debugger interface for TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import time
import uuid


class DebugRound(object):
  """Debug round class.

  A DebugRound object is a handle to a round of debugging. The round of
  debgging is created by specifying a graph node to execute and the number of
  times it should be executed for. The DebugRound object is stateful and keeps
  track of node order, the list of completed nodes, breakpoint, etc.
  """

  def __init__(self, debug_sess, node, num_times=1, feed=None):
    """DebugRound constructor.

    This sets up the feeds but does not actually run through the evaluated
    subgraph. Instead, it breaks at the first node (_SOURCE).

    Args:
      debug_sess: A tf.Session object of the "debug" type.
      node: Name of the graph node to execute (evaluate).
      num_times: Number of times to execute the node for (default: 1).
      feed: feed_dict for the run. Same as the run feed_dict of normal
        sessions.

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
    self._main_thr = self._start_debug_main_thread(node, feed=feed)

    where_output = self._sess.debug("where")
    self._node_order = (where_output["completed_nodes"] +
                        where_output["remaining_nodes"])

    self._curr_node = self._node_order[0]

    # Breakpoint states
    # Node name breakpoints. Elements are strings.
    self._node_breakpoints = []

    # Conditional (perdicate) breakpoints. Elements are callables of the form
    # break_or_not = should_i_break(node_name, node_value)
    self._pred_breakpoints = {}

  def _start_debug_main_thread(self, node, feed=None):
    """Start the main thread of the debug Session.

    Args:
      node: The evaluated node.
      feed: feed_dict for the node evaluation.

    Returns:
      Thread object for the debug Session's run.
    """
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

    Raises:
      ValueError: If the number of steps is invalid.
    """
    if num_steps == 1:
      output = self._sess.debug("step")
    elif num_steps > 1:
      output = self._sess.debug("step %d" % num_steps)
    else:
      raise ValueError("Invalid number of steps for stepping: %d" % num_steps)

    time.sleep(self._step_delay_sec)

    # Determine just completed node (i.e., current node)
    self._curr_node = self._node_order[self.where()]

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

    starting_node = self._node_order[self.where()]

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

    output = None
    while True:
      pos = self.where()
      node_just_completed = self._node_order[pos]

      # Break if this is a node breakpoint
      if (node_just_completed in self._node_breakpoints and
          node_just_completed != starting_node):
        # If this cont() call starts from this breakpoint node, we will not
        # break here again.
        break

      # Break if a predicate breakpoint is met
      if self._pred_breakpoints:
        should_break = False

        node_val = self.inspect_value(node_just_completed)
        if node_val is not None:
          for bp_key in self._pred_breakpoints:
            pred = self._pred_breakpoints[bp_key]
            if pred(node_just_completed, node_val):
              should_break = True
              break

        if should_break and node_just_completed != starting_node:
          # If this cont() call starts from this node, we will not break here
          # again.
          break

      # Break if the specified target node is reached
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

    That is, whether all nodes in the executed subgraph have finished
      executing.

    Returns:
      A boolean indicating whether the debug round is complete.
    """
    return self.where() == len(self._node_order) - 1

  def inspect_value(self, node_name):
    """Inspect the value of a node.

    Args:
      node_name: name of the node to inspect the value of.

    Returns:
      A tensor value (numpy array) if the node exists and has finished
        executing.
    None if the node doesn't exist or has not finished executing.
    """
    output = self._sess.debug("print %s" % node_name)
    node_value = output["node_value"]

    if node_value is not None:
      self._res = node_value

    return node_value

  def inject_value(self, new_value):
    """Inject a new Tensor value (numpy array) to the current node.

    Args:
      new_value: new Tensor value (numpy array)
    """

    self._sess.debug("inject_value %s" % self._curr_node,
                     feed={self._curr_node: new_value})

  def break_after(self, node_name):
    """Insert a break-right-after-node breakpoint to the debug round.

    Args:
      node_name: Name of the node. This has to point to a node that exists in
        the executed subgraph, or an exception will be raised.

    Returns:
      A handle for the breakpoint. The handle can later be used with methods
        of this class such as remove_breakpoint.

    Raises:
      ValueError: If the input node_name is not present in the executed
        subgraph.
    """
    # Verify that node_name is in the node order
    if node_name not in self._node_order:
      raise ValueError("Node named '%s' does not exist in the subgraph being "
                       "executed" % node_name)

    # If the node breakpoint already exists, return right away
    if node_name in self._node_breakpoints:
      return node_name

    self._node_breakpoints.append(node_name)

    bp_handle = node_name  # A handle for the breakpoint
    return bp_handle

  def break_before(self, node_name):
    """Insert a break-right-before-node breakpoint to the debug round.

    Args:
      node_name: Name of the node. This node has to exist in the executed
        subgraph, and the node must not be the first node, or an exception
        will be raised.

    Returns:
      A handle for the breakpoint. The handle can later be used with methods
        of this class such as remove_breakpoint

    Raises:
      ValueError: If the input node_name is not present in the executed
        subgraph, or if the node is the first node in the executed subgraph.
    """

    # Verify that node_name is in the node order
    if node_name not in self._node_order:
      raise ValueError("Node named '%s' does not exist in the executed "
                       "subgraph" % node_name)

    node_idx = self._node_order.index(node_name)
    if node_idx == 0:
      raise ValueError("Node named '%s' is the first node in the executed "
                       "subgraph, hence the debug round cannot break before "
                       "it" % node_name)

    return self.break_after(self._node_order[node_idx - 1])

  def break_if(self, predicate):
    """Break if a predicate regarding node name and/or value is met.

    Args:
      predicate: A callable that takes two input arguments and returns a boolean
        indicating whether the debug round should break.
        The first input argument is the node name, while the second one is the
        node value. This callable will be evaluated after the completion of each
        node and if it returns true, the debug round will break there.

    Returns:
      A handle for the breakpoint.

    Raises:
      ValueError: If predicate is not callable
    """
    # TODO(cais): Check for duplicate predicates? Is that possible?
    # TODO(cais): Verify that the predicate has valid input and return
    #             signature.

    # Verify that predicate is callable
    if not callable(predicate):
      raise ValueError("Input predicate is not callable")

    # The handle for the predicate breakpoint is non-clashing a random hex
    # string
    while True:
      handle = uuid.uuid4().hex
      if (handle not in self._node_breakpoints and
          handle not in self._pred_breakpoints):
        break

    self._pred_breakpoints[handle] = predicate

    return handle

  def remove_breakpoint(self, handle):
    """Remove a breakpoint references by the handle.

    Args:
      handle: Handle to the breakpoint, either node breakpoint or predicate
        breakpoint

    Raises:
      ValueError: If the breakpoint with the specified handle does not exist
        in this debug round.
    """

    if handle in self._node_breakpoints:
      self._node_breakpoints.remove(handle)
    elif handle in self._pred_breakpoints:
      self._pred_breakpoints.pop(handle)
    else:
      raise ValueError("Breakpoint with the specified handle does not exist "
                       "in this debug round.")

  def get_breakpoints(self):
    """Get all breakpoints, node and predicate.

    Returns:
      node_breakpoints: A list of all node breakpoints.
      pred_breakpoints: A dict of all predicate breakpoints.
    """
    return self._node_breakpoints, self._pred_breakpoints

  def join(self):
    """Join the main debug thread."""
    if self._main_thr:
      self._main_thr.join()
