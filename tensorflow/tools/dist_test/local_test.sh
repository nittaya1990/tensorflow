#!/bin/bash

# Configurations
DOCKER_IMG_NAME="tensorflow/tf-dist-test-local-cluster"
LOCAL_K8S_CACHE=${HOME}/kubernetes

# Arguments
NUM_WORKERS=2
NUM_PARAMETER_SERVERS=2

# Helper function
die() {
    echo $@
    exit 1
}

get_container_id_by_image_name() {
    # Get the id of a container by image name
    # Usage: get_docker_container_id_by_image_name <img_name>

    echo $(docker ps | grep $1 | awk '{print $1}')
}

# Current working directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker build -t ${DOCKER_IMG_NAME} \
   -f ${DIR}/Dockerfile.local ${DIR}


# Attempt to start the docker container with docker,
# which will run the k8s cluster inside.

# First, make sure that no docker-in-docker container of the same image
# is already running
if [[ ! -z $(get_container_id_by_image_name ${DOCKER_IMG_NAME}) ]]; then
    die "It appears that there is already at least one Docker container "\
"of image name ${DOCKER_IMG_NAME} running. Please stop it before trying again"
fi


# Get current script directory
CONTAINER_START_LOG=$(mktemp --suffix=.log)
echo "Log file for starting cluster container: ${CONTAINER_START_LOG}"
echo ""

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
${DIR}/local/start_tf_cluster_container.sh \
      ${LOCAL_K8S_CACHE} \
      ${DOCKER_IMG_NAME} | \
    tee ${CONTAINER_START_LOG} &

# Poll start log until the k8s service is started properly or when maximum
# attempt count is reached.
MAX_ATTEMPTS=1200

echo "Waiting for docker-in-docker container for local k8s TensorFlow "\
"cluster to start and launch Kubernetes..."

COUNTER=0
while true; do
    sleep 1

    ((COUNTER++))
    if [[ $(echo "${COUNTER}>=${MAX_ATTEMPTS} | bc -l") == "1" ]]; then
	die "Reached maximum number of attempts (${MAX_ATTEMPTS}) "\
"while waiting for docker-in-docker for local k8s TensorFlow cluster to start"
    fi

    # Check for hitting max attempt while trying to start docker-in-docker
    if [[ $(grep -i "Reached maximum number of attempts" \
	         "${CONTAINER_START_LOG}" | wc -l) == "1" ]]; then
	die "Docker-in-docker container for local k8s TensorFlow cluster "\
"FAILED to start"
    fi

    if [[ $(grep -i "Local Kubernetes cluster is running" \
		 "${CONTAINER_START_LOG}" | wc -l) == "1" ]]; then
	break
    fi
done

# Determine the id of the docker-in-docker container
DIND_ID=$(get_container_id_by_image_name ${DOCKER_IMG_NAME})

echo "Docker-in-docker container for local k8s TensorFlow cluster has been "\
"started successfully."
echo "Docker-in-docker container ID: ${DIND_ID}"
echo "Launching k8s tf cluster and tests in container ${DIND_ID} ..."
echo ""

# Launch k8s tf cluster in the docker-in-docker container and perform tests
docker exec ${DIND_ID} \
       /var/tf-k8s/local/test_local_tf_cluster.sh
TEST_RES=$?

# Tear down: stop docker-in-docker container
echo ""
echo "Stopping docker-in-docker container ${DIND_ID}"

docker stop --time=1 ${DIND_ID} || \
    echo "WARNING: Failed to stop container ${DIND_ID} !!"

echo ""

if [[ TEST_RES != "0" ]]; then
    die "Test of distributed TensorFlow runtime on docker-in-docker local "\
"k8s cluster FAILED"
else
    echo "Test of distributed TensorFlow runtime on docker-in-docker local "
"k8s cluster PASSED"
fi
