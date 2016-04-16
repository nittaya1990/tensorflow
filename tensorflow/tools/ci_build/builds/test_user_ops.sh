#!/usr/bin/env bash
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
#
# Test user-defined ops against installation of TensorFlow.
#
# Usage: test_user_ops.sh [--virtualenv]
#
# If the flag --virtualenv is set, the script will use "python" as the Python
# binary path. Otherwise, it will use tools/python_bin_path.sh to determine
# the Python binary path.
#

# Helper functions
# Get the absolute path from a path
abs_path() {
  [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Process input arguments
IS_VIRTUALENV=0
while true; do
  if [[ "$1" == "--virtualenv" ]]; then
    IS_VIRTUALENV=1
  fi
  shift

  if [[ -z "$1" ]]; then
    break
  fi
done

# Obtain the path to Python binary
if [[ ${IS_VIRTUALENV} == "1" ]]; then
  PYTHON_BIN_PATH="$(which python)"
else
  source tools/python_bin_path.sh
  # Assume: PYTHON_BIN_PATH is exported by the script above
fi
echo "PYTHON_BIN_PATH: ${PYTHON_BIN_PATH}"

# Locate the op kernel C++ file
OP_KERNEL_CC="${SCRIPT_DIR}/../../../g3doc/how_tos/adding_an_op/zero_out_op_kernel_1.cc"
OP_KERNEL_CC=$(abs_path "${OP_KERNEL_CC}")

if [[ ! -f "${OP_KERNEL_CC}" ]]; then
  die "ERROR: Unable to find user-op kernel C++ file at: ${OP_KERNEL_CC}"
fi

# Copy the file to a non-TensorFlow source directory
TMP_DIR=$(mktemp -d)
mkdir -p "${TMP_DIR}"

cleanup() {
  rm -rf "${TMP_DIR}"
}

die() {
  echo $@
  cleanup
  exit 1
}

pushd "${TMP_DIR}"
cp "${OP_KERNEL_CC}" ./

# Obtain paths include and lib paths to the TensorFlow installation
TF_INC=$("${PYTHON_BIN_PATH}" \
             -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
echo "TensorFlow include path: ${TF_INC}"

# Compile the op kernel into an .so file
SRC_FILE=$(basename "${OP_KERNEL_CC}")

echo "Compiling user op C++ source file ${SRC_FILE}"

USER_OP_SO="zero_out.so"
g++ -std=c++11 -shared "${SRC_FILE}" -o "${USER_OP_SO}" \
    -fPIC -I "${TF_INC}" || \
    die "FAILED to compile ${SRC_FILE}"

# Try running the op
echo "Invoking user op file in ${USER_OP_SO} via pip installation"
ORIG_OUTPUT=$("${PYTHON_BIN_PATH}" -c "import tensorflow as tf; print(tf.Session('').run(tf.load_op_library('./${USER_OP_SO}').zero_out([42, 43, 44])))")

# Format OUTPUT for analysis
OUTPUT=$(echo "${ORIG_OUTPUT}" | sed -e 's/\[//g' | sed -e 's/\]//g')
if [[ $(echo "${OUTPUT}" | wc -w) != "3" ]]; then
  die "ERROR: Unexpected number of elements in user op output"
fi

N0=$(echo "${OUTPUT}" | awk '{print $1}')
N1=$(echo "${OUTPUT}" | awk '{print $2}')
N2=$(echo "${OUTPUT}" | awk '{print $3}')

if [[ $(echo "${N0}==42" | bc -l) != "1" ]] || \
   [[ $(echo "${N1}==0" | bc -l) != "1" ]] || \
   [[ $(echo "${N2}==0" | bc -l) != "1" ]]; then
  die "FAILED: Incorrect output from user op: ${OUTPUT} "
fi

popd

cleanup

echo "SUCCESS: Testing of user ops PASSED"
