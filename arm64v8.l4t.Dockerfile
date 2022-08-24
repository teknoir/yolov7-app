ARG BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3
FROM ${BASE_IMAGE} as base

RUN apt update && apt install --no-install-recommends -y zip htop screen libgl1-mesa-glx

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip wheel
##RUN python3 -m pip uninstall -y Pillow torchtext  # torch torchvision
#RUN python3 -m pip install --no-cache -r requirements.txt albumentations wandb gsutil notebook \
##    Pillow>=9.1.0 \
#    'opencv-python-headless==4.5.5.62'
##    --extra-index-url https://download.pytorch.org/whl/cu113
RUN python3 -m pip install --no-cache Pillow paho.mqtt numpy torch pandas requests torchvision opencv-python-headless tqdm PyYAML matplot seaborn

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

ADD . /usr/src/app

ENV OMP_NUM_THREADS=8
ENV WANDB_MODE=disabled
ENV DEVICE=0

CMD ["python3", "app.py"]

FROM base

RUN wget https://raw.githubusercontent.com/onnx/models/main/vision/object_detection_segmentation/yolov4/dependencies/coco.names
ENV CLASS_NAMES=/usr/src/app/coco.names
RUN wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
ENV WEIGHTS=/usr/src/app/yolov7.pt
ENV IMG_SIZE=640