import time
import os
import sys
import logging

import json
import base64
from io import BytesIO
from PIL import Image

import paho.mqtt.client as mqtt

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from utils.torch_utils import select_device, time_synchronized
from utils.plots import plot_one_box


#  MQTT_IN_0="camera/images" MQTT_SERVICE_HOST=192.168.68.104 MQTT_SERVICE_PORT=31883 WEIGHTS=weights/best_weights.pt IMG_SIZE=640 CLASS_NAMES=ppe-bbox-clean-20220821000000146/dataset/object.names python3 app.py 


APP_NAME = os.getenv('APP_NAME', 'yolov7')

args = {
    'NAME': APP_NAME,
    'MQTT_SERVICE_HOST': os.getenv('MQTT_SERVICE_HOST', 'mqtt.kube-system'),
    'MQTT_SERVICE_PORT': int(os.getenv('MQTT_SERVICE_PORT', '1883')),
    'MQTT_IN_0': os.getenv("MQTT_IN_0", "camera/images"),
    'MQTT_OUT_0': os.getenv("MQTT_OUT_0", f"{APP_NAME}/events"),
    'WEIGHTS': os.getenv("WEIGHTS", ""),
    'CLASS_NAMES': os.getenv("CLASS_NAMES", ""),
    'TRAINING_DATASET': os.getenv("TRAINING_DATASET",""), # define from model config - better to use a registry
    'CLASSES': os.getenv("CLASSES", ""),
    'IMG_SIZE': int(os.getenv("IMG_SIZE", 416)),
    'CONF_THRESHOLD': float(os.getenv("CONF_THRESHOLD", 0.25)),
    'IOU_THRESHOLD': float(os.getenv("IOU_THRESHOLD", 0.45)),
    'DEVICE': os.getenv("DEVICE", 'cpu'),  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    'AUGMENTED_INFERENCE': os.getenv("AUGMENTED_INFERENCE", ""),
    'AGNOSTIC_NMS': os.getenv("AGNOSTIC_NMS", ""),
    'MODEL_NAME': 'yolov7',  # define from model config - better to use a registry
}

if args["AUGMENTED_INFERENCE"] == "":
    args["AUGMENTED_INFERENCE"] = False
else:
    args["AUGMENTED_INFERENCE"] = True    

if args["AGNOSTIC_NMS"] == "":
    args["AGNOSTIC_NMS"] = False
else:
    args["AGNOSTIC_NMS"] = True    

if args["CLASS_NAMES"] != "":
    class_names = []
    with open(args["CLASS_NAMES"],"r") as names_file:
        for line in names_file:
            if line != "" and line != "\n":
                class_names.append(line.strip())
    args["CLASSES"] = class_names
else:
    print("You must specify 'CLASS_NAMES'")
    sys.exit(1)

if args["CLASSES"] == "":
    args["CLASSES"] = None

logger = logging.getLogger(args['NAME'])
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.INFO)

logger.info("TΞꓘN01R")
logger.info("TΞꓘN01R")
logger.info("TΞꓘN01R")


def error_str(rc):
    return '{}: {}'.format(rc, mqtt.error_string(rc))


def on_connect(_client, _userdata, _flags, rc):
    logger.info('Connected to MQTT broker {}'.format(error_str(rc)))


def base64_encode(ndarray_image):
    buff = BytesIO()
    Image.fromarray(ndarray_image).save(buff, format='JPEG')
    string_encoded = base64.b64encode(buff.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{string_encoded}"


# def base64_decode(utf8_image):
#     image_mime = str(utf8_image.decode("utf-8", "ignore"))
#     _, image_base64 = image_mime.split(',', 1)
#     image = Image.open(BytesIO(base64.b64decode(image_base64)))
#     return image, image_mime


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.array):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# Initialize
set_logging()
device = select_device(args["DEVICE"])
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(args["WEIGHTS"], map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = args["IMG_SIZE"]
if isinstance(imgsz, (list, tuple)):
    assert len(imgsz) == 2
    "height and width of image has to be specified"
    imgsz[0] = check_img_size(imgsz[0], s=stride)
    imgsz[1] = check_img_size(imgsz[1], s=stride)
else:
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
# names = model.module.names if hasattr(model, 'module') else model.names  # get class names
if half:
    model.half()  # to FP16

# Run inference
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
        next(model.parameters())))  # run once

args["MODEL"] = model
args["STRIDE"] = stride


def detect(userdata, im0, image_mime):

    # Padded resize
    img = letterbox(im0, userdata["IMG_SIZE"], stride=userdata["STRIDE"])[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = np.expand_dims(img, axis=0)

    t0 = time.time()
    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.half() if half else img_tensor.float()  # uint8 to fp16/32
    img_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
    # if img.ndimension() == 3:
    #     img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = userdata['MODEL'](img_tensor, augment=userdata["AUGMENTED_INFERENCE"])[0]
    print(pred[..., 4].max())

    # Apply NMS
    pred = non_max_suppression(pred,
                               userdata["CONF_THRESHOLD"],
                               userdata["IOU_THRESHOLD"],
                               classes=userdata["CLASSES"],
                               agnostic=userdata["AGNOSTIC_NMS"])
    t2 = time_synchronized()

    annotated = im0.copy()

    # Process detections
    detections = []
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(img_tensor.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, confidence, detected_label in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                conf = confidence.item()
                
                label_index = int(detected_label.item())                
                label=None
                if label_index >= 0 and label_index < len(userdata["CLASS_NAMES"]):
                    label = userdata["CLASS_NAMES"][label_index]

                if label:
                    detections.append({'label': label, 
                                        'x': xywh[0],
                                        'y': xywh[1],
                                        'width': xywh[2],
                                        'height': xywh[3],
                                        'confidence': conf})
                    plot_one_box(xyxy, annotated, color=(0,255,0), label=f'{label} {conf:.2f}', line_thickness=1)

    payload = {
        "model": userdata["MODEL_NAME"],
        "image": base64_encode(annotated),
        "load_time": t1 - t0,
        "inference_time": t2 - t1,
        "total_time": t2 - t0,
        "training_dataset": userdata["TRAINING_DATASET"],
        "detected_objects": detections
    }

    msg = json.dumps(payload, cls=NumpyEncoder)
    client.publish(userdata['MQTT_OUT_0'], msg)
    payload["image"] = "%s... - truncated for logs" % payload["image"][0:32]
    logger.info(payload)


def on_message(c, userdata, msg):
    try:
        image_mime = str(msg.payload.decode("utf-8", "ignore"))
        _, image_base64 = image_mime.split(',', 1)
        image = Image.open(BytesIO(base64.b64decode(image_base64)))
        detect(userdata, im0=np.array(image), image_mime=image_mime)

    except Exception as e:
        logger.error('Error:', e)
        exit(1)


client = mqtt.Client(args['NAME'], clean_session=True, userdata=args)
client.on_connect = on_connect
client.on_message = on_message
client.connect(args['MQTT_SERVICE_HOST'], args['MQTT_SERVICE_PORT'])
client.subscribe(args['MQTT_IN_0'], qos=1)
# This runs the network code in a background thread and also handles reconnecting for you.
client.loop_forever()
