FROM arm64v8/ubuntu:20.04

RUN apt update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y tzdata
RUN apt install --no-install-recommends -y python3-pip git zip curl htop gcc \
    libgl1-mesa-glx libglib2.0-0 libpython3.8-dev

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip wheel
RUN pip install --no-cache -r requirements.txt gsutil notebook \
    tensorflow-aarch64
    # tensorflowjs \
    # onnx onnx-simplifier onnxruntime \
    # coremltools openvino-dev \

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

RUN git clone clone https://github.com/WongKinYiu/yolov7.git /usr/src/app

ENV WANDB_MODE=disabled

