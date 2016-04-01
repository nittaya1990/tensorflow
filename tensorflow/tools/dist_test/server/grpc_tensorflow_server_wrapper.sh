#!/bin/bash

LOG_FILE="/tmp/grpc_tensorflow_server.log"

SCRIPT_DIR=$( cd ${0%/*} && pwd -P )

touch "${LOG_FILE}"

python ${SCRIPT_DIR}/grpc_tensorflow_server.py $@ 2>&1 | tee "${LOG_FILE}"
