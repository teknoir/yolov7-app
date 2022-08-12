FROM nvcr.io/nvidia/pytorch:22.07-py3
RUN rm -rf /opt/pytorch  # remove 1.2GB dir

RUN apt update && apt install --no-install-recommends -y zip htop screen libgl1-mesa-glx

COPY yolov7/requirements.txt .
RUN python -m pip install --upgrade pip wheel
RUN pip uninstall -y Pillow torchtext  # torch torchvision
RUN pip install --no-cache -r requirements.txt albumentations wandb gsutil notebook Pillow>=9.1.0 \
    'opencv-python<4.6.0.66' \
    --extra-index-url https://download.pytorch.org/whl/cu113

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

RUN git clone https://github.com/WongKinYiu/yolov7.git /usr/src/app

ENV OMP_NUM_THREADS=8
ENV WANDB_MODE=disabled

