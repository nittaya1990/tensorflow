#!/bin/bash

# Usage: start_tf_cluster_container.sh <local_k8s_dir> <docker_img_name>
#
# local_k8s_dir:   Kubernetes (k8s) source directory on the host
# docker_img_name: Name of the docker image to start

if [[ $# != "2" ]]; then
    echo "Usage: $0 <host_k8s_dir> <docker_img_name>"
    exit 1
fi

HOST_K8S_DIR=$1
DOCKER_IMG_NAME=$2

# Maximum number of tries to start the docker container with docker running
# inside
MAX_ATTEMPTS=100

# Attempt to start docker service in docker container.
# Try multiple times if necessary.
COUNTER=1
while true; do
    ((COUNTER++))
    docker run --net=host --privileged \
	   -v ${HOME}/${HOST_K8S_DIR}:/local/kubernetes \
	   ${DOCKER_IMG_NAME} \
	   /var/tf-k8s/local/start_local_k8s_service.sh

    if [[ $? == "23" ]]; then
	if [[ $(echo "${COUNTER}>=${MAX_ATTEMPTS} | bc -l") == "1" ]]; then
	    echo "Reached maximum number of attempts (${MAX_ATTEMPTS}) "\
"while attempting to start docker-in-docker for local k8s TensorFlow cluster"
	    exit 1
	fi

	echo "Docker service failed to start... "\
	     "Will make another attempt (#${COUNTER}) to start it..."
	sleep 1
    else
	break
    fi
done
