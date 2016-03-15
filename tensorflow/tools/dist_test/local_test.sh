#!/bin/bash

# Configurations
DOCKER_IMG_NAME="tensorflow/tf-dist-test-local-cluster"
LOCAL_K8S_CACHE=${HOME}/kubernetes

# Arguments
NUM_WORKERS=2
NUM_PARAMETER_SERVERS=2

# Helper functions
die() {
    echo $@
    exit 1
}

# Current working directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker build -t ${DOCKER_IMG_NAME} \
   -f ${DIR}/Dockerfile.local ${DIR}
# Attempt to start docker service in docker container.
# Try multiple times if necessary.
COUNTER=1
while true; do
    ((COUNTER++))
    docker run --net=host --privileged \
	   -v ~/${LOCAL_K8S_CACHE}:/local/kubernetes \
	   ${DOCKER_IMG_NAME} \
	   /var/tf-k8s/local/start_local_k8s_service.sh
    
    if [[ $? == "23" ]]; then
	echo "Docker service failed to start... "\
	     "Will make attempt (#${COUNTER}) to start it..."
	sleep 1
    else
	break
    fi
done
