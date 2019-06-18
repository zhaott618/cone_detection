# -*- coding:utf-8 -*-

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image,ImageFont,ImageDraw

from yolo3.model import yolo_eval,yolo_body,tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
from yolo import YOLO
from yolo import detect_video
import cv2
# import pyzed.camera as zcam
# import pyzed.defines as sl
# import pyzed.types as tp
# import pyzed.core as core
# import math
# import sys

# def zed_detect_video(yolo,video_path,output_path="")
# #     import cv2
# #     zed=zcam.PyZEDCamera()
# #
# #     init_paras=zcam.PyInitParameters()
# #     init_paras.depth_mode=sl.PyDEPTH_MODE.PyDEPTH_MODE_PERFORMANCE
# #     init_paras.coordinate_system=sl.PyUNIT.PyUNIT_MILLIMETER
# #
# #     err=zed.open(init_paras)
# #     if err!=tp.PyERROR_CODE.PySUCCESS:
# #         exit(1)
# #
# #     runtime_paras=zcam.PyRuntimeParameters
# #     runtime_paras.sensing_mode=sl.PySENSING_MODE.PySENSING_MODE_STANDARD
# #
# #     #i=0
# #     image=core.Pymat()
# #     depth=core.Pymat()
# #     point_cloud=core.Pymat()
# #
# #     while True:
# #         if zed.grab(runtime_paras)==tp.PyERROR_CODE.PySUCCESS
# #             zed.retrieve_image(image,sl.PyVIEW.PyVIEW_LEFT)
# #             zed.retrieve_measure(depth,sl.PyMEASURE.PyMEASURE_DEPTH)
# #             zed.retrieve_measure(point_cloud,sl.PyMEASURE.PyMEASURE_XYZRGBA)
# #             frame=image
# #             x=round()
# #
# #
# # if __name__='__main__':
# #     yolo=YOLO()
# #     video_path='model_data/Redundant Perception and State Estimation for Reliable Autonomous Racing_Trim.mp4'
# #     detect_video(yolo,video_path,output_path)


cap=cv2.VideoCapture(1)

# DATASET_DIR=./ panda_tfrecord
# TRAIN_DIR=./panda_model
# CHECKPOINT_PATH=./model_pre_train/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt
# python3 SSD-Tensorflow-master/train_ssd_network.py \
#     --train_dir=${TRAIN_DIR} \
#     --dataset_dir=${DATASET_DIR} \
#     --dataset_name=pascalvoc_2007 \
#     --dataset_split_name=train \
#     --model_name=ssd_300_vgg \
#     --checkpoint_path=${CHECKPOINT_PATH} \
#     --save_summaries_secs=60 \
#     --save_interval_secs=600 \
#     --weight_decay=0.0005 \
#     --optimizer=adam \
#     --learning_rate=0.0001 \
#     --batch_size=16
