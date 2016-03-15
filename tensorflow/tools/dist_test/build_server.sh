#!/bin/bash

if [[ $# != 1 ]]; then
  echo "Usage: $0 <docker_image_name>"
  exit 1
fi

DOCKER_IMG_NAME=$1

# Get current script directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker build -t "${DOCKER_IMG_NAME}" \
   -f "${DIR}/server/Dockerfile" \
   "${DIR}"
