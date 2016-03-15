#!/bin/bash

K8S_SRC_REPO=https://github.com/kubernetes/kubernetes.git
K8S_SRC_DIR=${TF_DIST_K8S_SRC_DIR:-/local/kubernetes}
K8S_SRC_BRANCH=${TF_DIST_K8S_SRC_BRANCH:-release-1.2}

# Helper functions
die() {
    echo $@
    exit 1
}

# Start docker service. Try multiple times if necessary.
COUNTER=0
while true; do
    ((COUNTER++))
    service docker start
    sleep 1

    service docker status
    if [[ $? == "0" ]]; then
	echo "Docker service started successfully."
	break;
    else
	echo "Docker service failed to start"
	exit 23

    fi
done

# Wait for docker0 net interface to appear
echo "Waiting for docker0 network interface to appear..."
while true; do
    if [[ -z $(netstat -i | grep "^docker0") ]]; then
	sleep 1
    else
	break
    fi
done
echo "docker0 interface has appeared."

# Set docker0 to promiscuous mode
ip link set docker0 promisc on || \
    die "FAILED to set docker0 to promiscuous"
echo "Turned promisc on for docker0"

# Check promiscuous mode of docker0
netstat -i

if [[ ! -d "${K8S_SRC_DIR}/.git" ]]; then
  mkdir -p ${K8S_SRC_DIR}
  git clone ${K8S_SRC_REPO} ${K8S_SRC_DIR} || \
      die "FAILED to clone k8s source from GitHub from: ${K8S_SRC_REPO}"
fi

pushd ${K8S_SRC_DIR}
git checkout ${K8S_SRC_BRANCH} || \
    die "FAILED to checkout k8s source branch: ${K8S_SRC_BRANCH}"
git pull origin ${K8S_SRC_BRANCH} || \
    die "FAILED to pull from k8s source branch: ${K8S_SRC_BRANCH}"

# Create kubectl binary

# Install etcd
hack/install-etcd.sh

export PATH=$(pwd)/third_party/etcd:${PATH}

# Setup golang
export PATH=/usr/local/go/bin:${PATH}

echo "etcd path: $(which etcd)"
echo "go path: $(which go)"

# Create shortcut to kubectl
echo '#!/bin/bash' > /usr/local/bin/kubectl
echo "$(pwd)/cluster/kubectl.sh \\" >> /usr/local/bin/kubectl
echo '    $@' >> /usr/local/bin/kubectl
chmod +x /usr/local/bin/kubectl

# Bring up local cluster
export KUBE_ENABLE_CLUSTER_DNS=true
hack/local-up-cluster.sh

popd
