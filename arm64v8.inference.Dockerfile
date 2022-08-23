FROM balenalib/generic-aarch64-debian:bullseye

RUN [ "cross-build-start" ]

RUN apt-get update && \
    apt-get install -y --no-install-recommends --allow-downgrades \
    git wget tar \
    python3-dev \
    python3 \
    python3-pip \
    python3-opencv \
    python3-pil \
    python3-setuptools \
    python3-numpy \
    ffmpeg \
    protobuf-compiler \
    libopenexr-dev libdc1394-22-dev libeigen3-dev \
    libatlas-base-dev

RUN apt update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y tzdata
RUN apt install --no-install-recommends -y python3-pip git zip curl htop gcc \
    libgl1-mesa-glx libglib2.0-0 libpython3.8-dev

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip wheel
RUN pip install --no-cache -r requirements.txt gsutil tensorflow-aarch64
    # onnx onnx-simplifier onnxruntime \

RUN mkdir -p /app
WORKDIR /app

ADD utils /app/utils
ADD models /app/models
COPY ./app.py /app/app.py

ENV WANDB_MODE=disabled

