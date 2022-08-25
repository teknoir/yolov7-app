# As L4T and cross compilation is a tough nut, only the base image have to be built on the device
# If we only add files in the last layers of the image, it can be done without cross compilation or directly on the device
# I.e. models, weights, settings and python files can be added here, as long as there is nothing executed (no adding of deps etc.)

ARG BASE_IMAGE=gcr.io/teknoir/yolov7:l4tr32.7.1
FROM ${BASE_IMAGE} as base

ARG MODEL_NAME=yolov7
ENV MODEL_NAME=$MODEL_NAME

ARG TRAINING_DATASET=cocoa
ENV TRAINING_DATASET=$TRAINING_DATASET

ARG IMG_SIZE=640
ENV IMG_SIZE=$IMG_SIZE

ARG WEIGHTS_FILE
ENV WEIGHTS_FILE=$WEIGHTS_FILE
ENV WEIGHTS=/usr/src/app/model.pt
ADD $WEIGHTS_FILE $WEIGHTS


ARG CLASS_NAMES_FILE
ENV CLASS_NAMES_FILE=$CLASS_NAMES_FILE
ENV CLASS_NAMES=/usr/src/app/obj.names
ADD $CLASS_NAMES_FILE $CLASS_NAMES

#RUN wget https://raw.githubusercontent.com/onnx/models/main/vision/object_detection_segmentation/yolov4/dependencies/coco.names
#ENV CLASS_NAMES=/usr/src/app/coco.names
#RUN wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
#ENV WEIGHTS=/usr/src/app/yolov7.pt
#ENV IMG_SIZE=640