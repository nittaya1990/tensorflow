FROM ubuntu:14.04

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y \
        bc \
        curl \
        dnsutils \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-numpy \
        python-pip \
        python-scipy \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install ipython
Run pip install ipython

# Install TensorFlow CPU version.
RUN pip --no-cache-dir install \
    http://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_CONTAINER_TYPE=CPU,TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-0.7.1-cp27-none-linux_x86_64.whl

# Copy the GRPC server binary
# TODO(cais): Use grpc python library
ADD grpc_tensorflow_server /tensorflow/grpc_tensorflow_server

ADD . /var/tf-k8s

# Copy MNIST data for tests
ADD server/mnist-data /tmp/mnist-data

ENTRYPOINT ["/tensorflow/grpc_tensorflow_server"]