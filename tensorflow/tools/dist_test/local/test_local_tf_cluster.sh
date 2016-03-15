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

# Wait for the kube-system pods to be running
KUBECTL_BIN=$(which kubectl)
if [[ -z ${KUBECTL_BIN} ]]; then
  die "FAILED to find path to kubectl"
fi

are_all_pods_running() {
  # Usage: area_all_pods_running ${namespace}
  if [[ -z "$1" ]]; then
    NS_FLAG=""
  else
    NS_FLAG="--namespace=$1"
  fi

  NPODS=$("${KUBECTL_BIN}" "${NS_FLAG}" get pods | tail -n +2 | wc -l)
  NRUNNING=$("${KUBECTL_BIN}" "${NS_FLAG}" get pods | tail -n +2 | \
		    grep "Running" | wc -l)

  if [[ ${NPODS} == ${NRUNNING} ]]; then
    echo "1"
  else
    echo "0"
  fi
}

echo "Waiting for kube-system pods to be all running..."

MAX_ATTEMPTS=240
COUNTER=0
while true; do
  sleep 1
  ((COUNTER++))
  if [[ $(echo "${COUNTER}>${MAX_ATTEMPTS}" | bc -l) == "1" ]]; then
    die "Reached maximum polling attempts while waiting for all pods in "\
"kube-system to be running in local k8s TensorFlow cluster"
  fi

  if [[ $(are_all_pods_running "kube-system") == "1" ]]; then
    break
  fi
done

# Create the local k8s tf cluster
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

if [[ $? != "0" ]]; then
    die "Test of local k8s TensorFlow cluster FAILED"
else
    echo "Test of local k8s TensorFlow cluster PASSED"
fi
