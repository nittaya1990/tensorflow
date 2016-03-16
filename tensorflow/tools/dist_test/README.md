# Testing Distributed Runtime in TensorFlow
This folder containers tools and test suites for the GRPC-based distributed runtime in TensorFlow.

There are three general modes of testing:

**1) Launch a local Kubernetes (k8s) cluster and run the test suites on it**

For example:

    export TF_DIST_SERVER_DOCKER_IMAGE="cais/tf_grpc_test_server"
    ./local_test.sh

Here the environment variable TF_DIST_SERVER_DOCKER_IMAGE overrides the default Docker image used to generate the TensorFlow GRPC server pods ("tensorflow/tf_grpc_test_server").
This option makes use of the docker-in-docker (dind) containers. It requires that the docker0 network interface is set to promiscuous mode on the host:

    sudo ip link set docker0 promisc on

**2) Launch a remote k8s cluster on Google Container Engine (GKE) and run the test suite on it**

For example:

    export TF_DIST_GCLOUD_PROJECT="tensorflow-testing"
    export TF_DIST_GCLOUD_COMPUTE_ZONE="us-central1-f"
    export CONTAINER_CLUSTER="test-cluster-1"
    export TF_DIST_GCLOUD_KEY_FILE_DIR="/tmp/gcloud-secrets"
    ./remote_test.sh

Here you specify the Google Compute Engine (GCE) project, compute zone and container cluster with the first three environment variables, in that order. The environment variable "TF_DIST_GCLOUD_KEY_FILE_DIR" is a directory in which the JSON service account key file named "tensorflow-testing.json" is located. You can use the flag "--setup_cluster_only" to perform only the cluster set up step and skip the test step:

    ./remote_test.sh --setup_cluster_only

**3) Run the test suite on an existing k8s TensorFlow cluster**

For example:

    export TF_DIST_GRPC_SERVER_URL="grpc://11.22.33.44:2222"
    ./remote_test.sh

Such a cluster may have been set up using the command describe in the previous section.


**Building the test server Docker image**

To build the Docker image for the test GRPC TensorFlow distributed server, run:

    ./build_server.sh <docker_image_name>


**Generating configuration file for TensorFlow k8s clusters**

The script at "scripts/k8s_tensorflow.py" can be used to generate yaml configuration files for a TensorFlow k8s cluster consisting of a number of workers and prameter servers (ps). For example:

    scripts/k8s_tensorflow.py \
        --num_workers 2 \
        --num_parameter_servers 2 \
        --grpc_port 2222 \
        --request_load_balancer \
        --docker_image cais/tf_grpc_test_server > tf-k8s.yaml
