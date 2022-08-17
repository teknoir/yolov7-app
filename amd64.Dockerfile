#FROM nvcr.io/nvidia/pytorch:22.07-py3
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
#FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
RUN rm -rf /opt/pytorch  # remove 1.2GB dir

RUN apt update && apt install --no-install-recommends -y zip htop screen libgl1-mesa-glx

COPY requirements.txt .
RUN python -m pip install --upgrade pip wheel
RUN pip uninstall -y Pillow torchtext  # torch torchvision
RUN pip install --no-cache -r requirements.txt albumentations wandb gsutil notebook Pillow>=9.1.0 \
    'opencv-python-headless==4.5.5.62' \
    --extra-index-url https://download.pytorch.org/whl/cu113

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

RUN git clone https://github.com/WongKinYiu/yolov7.git /usr/src/app

ENV OMP_NUM_THREADS=8
ENV WANDB_MODE=disabled

ENTRYPOINT ["/bin/bash"]
