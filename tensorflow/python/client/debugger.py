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

from six.moves import xrange  # pylint: disable=redefined-builtin


class DebugRound(object):
  """Debug round class.

  A DebugRound object is a handle to a round of debugging. The round of
  debugging is created by specifying a graph node to execute and the number of
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
    self._init_delay_sec = 0.05

    if debug_sess.sess_str != "debug":
      raise ValueError(
          "Failed attempt to create a DebugRound from a non-debug session "
          "object")

    if not num_times >= 1:
      raise ValueError("Invalid value of num_times: %s" % repr(num_times))
    self._num_times = int(num_times)

    self._sess = debug_sess
    self._executed_node = node
    self._feed = feed

    # Start the debugger main thread
    self._main_thr = self._start_debug_main_thread(self._executed_node,
                                                   feed=self._feed)

    # Populate _node_order
    where_output = self._sess.debug("where")
    self._unprefixed_nodes = (where_output["completed_nodes"] +
                              where_output["remaining_nodes"])
    if self._num_times == 1:
      self._node_order = self._unprefixed_nodes
    else:
      # Note that this works for deterministic orderigng only
      self._node_order = []
      for i in xrange(self._num_times):
        self._node_order.extend(
            ["%d_%s" % (i, node) for node in self._unprefixed_nodes])

    self._rep_idx = 0  # Repetition index

    self._curr_node = self._node_order[0]
    # TODO(cais): Remove this
    if not isinstance(self._curr_node, bytes):
      self._curr_node = self._curr_node.encode("utf-8")

    # Breakpoint states
    # Node name breakpoints. Elements are strings.
    # This is a dict. The keys are the user-specified breakpoints, which can
    # be non-prefixed in a multi-repetition round. The values are lists of
    # actual node names, prefixed if multi-repetition.
    self._node_breakpoints = {}

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

    # Wait for the Run() call to settle
    time.sleep(self._init_delay_sec)

    print("Returning...")  # DEBUG
    return thr

  def query_node_order(self):
    """Queries node order.

    If num_times is 1, the node names are the same as in the graph. However,
    if num_times > 1, the node names will be prefixed with a 0-based index,
    such as "0__SOURCE", "1_transpose" and "3__SINK".

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
    if num_steps >= 1:
      for _ in xrange(num_steps):
        output = self._sess.debug("step")
    else:
      raise ValueError("Invalid number of steps for stepping: %d" % num_steps)

    # Determine just completed node (i.e., current node)
    self._curr_node = self._node_order[self.where()]

    # In case there are >1 repetitions, check if we have reached the
    # end and if so, kick of a new debug Session run

    if self._num_times > 1 and output["is_completed"]:
      if self._rep_idx + 1 < self._num_times:
        self._main_thr = self._start_debug_main_thread(self._executed_node,
                                                       feed=self._feed)

        print("Repetition %d/%d has completed! Starting repetition %d/%d" %
              (self._rep_idx, self._num_times,
               self._rep_idx + 1, self._num_times))
        self._rep_idx += 1

    # TODO(cais): Remove this?
    if not isinstance(self._curr_node, bytes):
      self._curr_node = self._curr_node.encode("utf-8")

    return output

  def cont(self, node_name=None):
    """Continue execution till the named node or first breakpoint.

    Args:
      node_name: Name of the node to try to reach (in the absence of
        breakpoints). If no node name is provided, will try to continue to the
        end.
        In the case of multiple-repetition (num_times > 1) executions, the
        node_name can start with a 0-based repetition prefix, e.g., "0_".
        The repetition prefix may be omitted, in which case the debugger will
        automatically append the prefix for the current repetition to the node
        name. This can cause an exception if the node for the current
        repetition has already finished executing.

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

    # If this is a multi-repetition run, check the repetition prefix in the
    # node name.
    if self._num_times > 1:
      valid_prefix = True
      try:
        node_rep_idx = int(node_name.split("_")[0])
        if node_rep_idx < 0 or node_rep_idx >= self._num_times:
          valid_prefix = False
      except ValueError:
        valid_prefix = False

      if not valid_prefix:
        rep_prefix = "%d_" % self._rep_idx
        node_name = rep_prefix + node_name

    # Verify that node_name is in the node order list
    if self._node_order.count(node_name) == 0:
      raise ValueError("Node named '%s' does not exist in the node order "
                       "list of this debug round" % node_name)

    # Verify that the node has not completd yet
    node_idx = self._node_order.index(node_name)
    if node_idx <= self.where():
      raise ValueError("Cannot continue to node named '%s' because that node "
                       "has already finished executing" % node_name)

    output = None
    while True:
      pos = self.where()
      node_just_completed = self._node_order[pos]

      # Determine if we have hit a breakpoint
      breakpoint_hit = False
      for bp_name in self._node_breakpoints:
        node_list = self._node_breakpoints[bp_name]
        if node_just_completed in node_list:
          breakpoint_hit = True
          break

      # Break if this is a node breakpoint
      if (breakpoint_hit and node_just_completed != starting_node):
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
    curr_node = str(self._sess.debug("where")["completed_nodes"][-1])
    if self._num_times > 1:
      curr_node = "%d_%s" % (self._rep_idx, curr_node)

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

    if self._num_times > 1:
      prefix = "%d_" % self._rep_idx
      if node_name.startswith(prefix):
        node_name = node_name[len(prefix):]

    output = self._sess.debug("inspect_value %s" % node_name)
    node_value = output["node_value"]

    if node_value is not None:
      self._res = node_value

    return node_value

  def inject_value(self, new_value):
    """Inject a new Tensor value (numpy array) to the current node.

    Args:
      new_value: new Tensor value (numpy array)
    """

    inject_cmd = "inject_value %s" % self._curr_node
    self._sess.debug(inject_cmd,
                     feed={self._curr_node: new_value})

  def break_after(self, node_name):
    """Insert a break-right-after-node breakpoint to the debug round.

    Args:
      node_name: Name of the node. This has to point to a node that exists in
        the executed subgraph, or an exception will be raised.
        In the case of a multi-repetition run (num_times > 1), the node name
        can be prefixed with the repetition index (e.g., 2_node_A), in which
        case the debug round will break after the node in the specific run.
        Or it can be non-prefixed, in which case the debug round will break
        after every time the node is just completed in each repetition.

    Returns:
      A handle for the breakpoint. The handle can later be used with methods
        of this class such as remove_breakpoint.

    Raises:
      ValueError: If the input node_name is not present in the executed
        subgraph.
    """
    # If the node breakpoint already exists, return right away
    if node_name in self._node_breakpoints:
      return node_name

    # Verify that node_name is in the node order
    if self._num_times == 1:
      if node_name in self._node_order:
        self._node_breakpoints[node_name] = [node_name]
      else:
        raise ValueError("Node named '%s' does not exist in the subgraph being "
                         "executed" % node_name)
    else:
      # Multi-repetition run. Check if the node name is non-prefixed but valid
      if node_name in self._node_order:
        self._node_breakpoints[node_name] = [node_name]
      elif node_name in self._unprefixed_nodes:
        self._node_breakpoints[node_name] = \
            ["%d_%s" % (i, node_name) for i in xrange(self._num_times)]
      else:
        raise ValueError("Node named '%s' does not exist in the subgraph being "
                         "executed" % node_name)

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
      self._node_breakpoints.pop(handle)
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

  def get_num_times(self):
    """Get number of times this debug round is to evaluate the node.

    Returns:
      Number of times (num_times) the node is to be evaluated for.
    """
    return self._num_times

  def get_repetition_index(self):
    """Get the current repetition index (0-based).

    Returns:
      The 0-based repetition index.
    """
    return self._rep_idx

  def join(self):
    """Join the main debug thread."""
    if self._main_thr:
      self._main_thr.join()

