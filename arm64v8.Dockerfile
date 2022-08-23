FROM balenalib/generic-aarch64-debian:bullseye

RUN [ "cross-build-start" ]
WORKDIR /app
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

RUN python3 -m pip install pip --upgrade
RUN python3 -m pip install protobuf==3.20.* mediapipe paho.mqtt
COPY app.py .

RUN [ "cross-build-end" ]

CMD ["python3", "app.py"]
