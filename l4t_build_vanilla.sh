#!/usr/bin/env bash
set -eo pipefail

export SHORT_SHA=${SHORT_SHA:-"local"}
export BRANCH_NAME=${BRANCH_NAME:-"head"}
export PROJECT_ID=${PROJECT_ID:-"teknoir"}

# Get vanilla model and COCOA names file
download_cache(){
  FILE=$1
  URL=$2
  echo "Download ${FILE}"

  if [ -f "${FILE}" ]; then
    echo "${FILE} exists."
  else
    echo "Download ${FILE} from ${URL}"
    curl -fsSL --progress-bar -o ${FILE} ${URL} || {
      error "curl -fsSL --progress-bar -o ${FILE} ${URL}" "${FUNCNAME}" "${LINENO}"
      exit 1
    }
  fi
}
download_cache coco.names https://raw.githubusercontent.com/onnx/models/main/vision/object_detection_segmentation/yolov4/dependencies/coco.names
download_cache yolov7.pt https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt

# Make sure the latest version of base image is local
docker pull gcr.io/teknoir/yolov7:l4tr32.7.1

# Build and set values specific to this model
docker buildx build \
  --build-arg=BASE_IMAGE=gcr.io/teknoir/yolov7:l4tr32.7.1 \
  --build-arg=MODEL_NAME=yolov7-vanilla \
  --build-arg=TRAINING_DATASET=cocoa \
  --build-arg=IMG_SIZE=640 \
  --build-arg=WEIGHTS_FILE=yolov7.pt \
  --build-arg=CLASS_NAMES_FILE=coco.names \
  --platform=linux/arm64 \
  -t gcr.io/${PROJECT_ID}/yolov7-vanilla:l4tr32.7.1-${BRANCH_NAME}-${SHORT_SHA} \
  -f ./arm64v8.l4t.yolov7.Dockerfile .

docker push gcr.io/${PROJECT_ID}/yolov7-vanilla:l4tr32.7.1-${BRANCH_NAME}-${SHORT_SHA}
