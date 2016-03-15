#!/bin/bash

export GCLOUD_BIN=/usr/local/bin/gcloud
export TF_DIST_LOCAL_CLUSTER=1

# Helper functions
die() {
  echo $@
  exit 1
}

NUM_WORKERS=2
NUM_PARAMETER_SERVERS=2

# Get current script directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

${DIR}/../scripts/create_tf_cluster.sh \
    ${NUM_WORKERS} ${NUM_PARAMETER_SERVERS} | \
    tee /tmp/tf_cluster.log || \
    die "FAILED to create local tf cluster"

DOCKER_CONTAINER_ID=$(cat /tmp/tf_cluster.log | \
			     grep "Docker container ID" |
			     awk '{print $NF}')
if [[ -z "${DOCKER_CONTAINER_ID}" ]]; then
  die "FAILED to determine worker0 Docker container ID"
fi

export TF_DIST_GRPC_SERVER_URL="grpc://tf-worker0:2222"
GRPC_ENV="TF_DIST_GRPC_SERVER_URL=${TF_DIST_GRPC_SERVER_URL}"

docker exec \
       ${DOCKER_CONTAINER_ID} \
       /bin/bash -c \
       "${GRPC_ENV} /var/tf-k8s/scripts/dist_test.sh"
# TODO(cais): Finish
