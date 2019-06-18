# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
from keras.utils import multi_gpu_model
import pyzed.camera as zcam
import pyzed.defines as sl
import pyzed.types as tp
import pyzed.core as core
import sys
import math

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/trained_weights_final.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/my_class.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        from imageai.Prediction.Custom import CustomImagePrediction
        execution_path=os.getcwd()
        prediction=CustomImagePrediction()
        prediction.setModelTypeAsResNet()
        prediction.setModelPath(os.path.join(execution_path,"cone_color_keras/model_ex-100_acc-1.000000.h5"))
        prediction.setJsonPath(os.path.join(execution_path,"cone_color_keras/model_class_json"))
        prediction.loadModel(num_objects=3)
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size))) #letterbox()标准化尺寸？？
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            global Class
            Class=c
            box = out_boxes[i]
            score = out_scores[i]

            #label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            #label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            #print(label, (left, top), (right, bottom))
            
            image_to_detect_color=np.array(image)
            image_to_detect_color=image_to_detect_color[left:right,top:bottom]
            predictions,probabilities=prediction.predictImage(image_to_detect_color,result_count=3)
            label='{} {:.2f}'.format('blue cone',probabilities)
            label_size=draw.textsize(label,font)
            print(label,(left,top),(right,bottom))

            #calculate distance to the cones
            global locX,locY
            locX=round((left+right)/2)
            locY=round(top+(bottom-top)*0.8)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.ellipse(((left+right)/2,(top+(bottom-top)*0.8) , 5, 5), fill=self.colors[c])
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])

            #draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            if predictions==0:
                label='{} {:.2f}'.format('blue cone',probabilities)
                draw.text(text_origin,label,fill=(0,0,0),font=font)
            elif predictions==1:
                label='{} {:.2f}'.format('red cone',probabilities)
                draw.text(text_origin,label,fill=(0,0,0),font=font)
            else:
                label='{} {:.2f}'.format('yellow cone',probabilities)
                draw.text(text_origin,label,fill=(0,0,0),font=font)
            del draw

        end = timer()
        print(end - start)
        #import cv2
        #cv2.imshow("detected",image)
        return image

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    #import socket
    import cv2
    #BUFSIZE=1024
    #client=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    #ip_port=('192.168.11.174',8888)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    zed=zcam.PyZEDCamera()

    init_params=zcam.PyInitParameters()
    init_params.camera_resolution=sl.PyRESOLUTION.PyRESOLUTION_VGA
    init_params.coordinate_units=sl.PyUNIT.PyUNIT_MILLIMETER

    err=zed.open(init_params)
    if err!=tp.PyERROR_CODE.PySUCCESS:
        exit(-1)

    runtime_params=zcam.PyRuntimeParameters()
    runtime_params.sensing_mode=sl.PySENSING_MODE.PySENSING_MODE_STANDARD

    image=core.PyMat()
    depth=core.PyMat()
    point_cloud=core.PyMat()
    #count=0

    while True:
        if zed.grab(runtime_params)==tp.PyERROR_CODE.PySUCCESS:
            zed.retrieve_image(image,sl.PyVIEW.PyVIEW_LEFT)
            #zed.retrieve_measure(depth,sl.PyMEASURE.PyMEASURE_DEPTH)
            #zed.retrieve_measure(point_cloud,sl.PyMEASURE.PyMEASURE_XYZRGBA)
            #frame=np.asarray(frame,image.size,image.type)
            print(type(image))
            #frame0=core.slMat2cvMat(image)
            frame0=image.get_data()
            #cloud_img=point_cloud.get_data()

            print(type(frame0))
            frame=Image.fromarray(frame0)
            imageA=yolo.detect_image(frame)
            #x=round(image.get_width()/2)
            #y=round(image.get_height()/2)
            global locX,locY
            #err,point_cloud_value=point_cloud.get_value(locX,locY)
            #distance=math.sqrt(point_cloud_value[0]*point_cloud_value[0]+point_cloud_value[1]*point_cloud_value[1]+point_cloud_value[2]*point_cloud_value[2])
        #msg=StringIO.StringIO()
            """if not np.isnan(distance) and not np.isinf(distance):
                distance=round(distance)
                global Class
                if Class==0:
                    sprintf(msg,"Distance to Blue Cone at(%d，%d):%d mm\n"，locX,locY,distance)
                elif Class==1:
                    sprintf(msg,"Distance to Red Cone at(%d, %d):%d mm\n",locX,locY,distance)
                else:
                    sprintf(msg,"Distance to Yellow Cone at(%d, %d):%d mm\n",locX,locY,distance)
                #print(msg.getvalue())
                #msg1=buf+str(count)
                #client.sendto(msg1.encode('utf-8'),ip_port)
                #data,server_addr=client.recvfrom(BUFSIZE)
                #count+=1
                
            else:
                print("Can't estimate distance at this position!\n")
            sys.stdout.flush()"""
            
            

            #imageL,locXL,locYL=yolo.detect_image(imageL)
            resultA=np.asarray(imageA)
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(resultA, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.namedWindow("result_win", cv2.WINDOW_NORMAL)
            cv2.imshow("result_win", resultA)
            #printf("the")
            # if isOutput:
            #     out.write(resultL)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()


if __name__ == '__main__':
    yolo=YOLO()
    Class=0
    locX=0
    locY=0
    video_path='model_data/Redundant Perception and State Estimation for Reliable Autonomous Racing_Trim.mp4'
    detect_video(yolo, video_path, output_path="model_data/detected_cone4.mp4")
    """path='D:\Downloads\keras-yolo3-master\keras-yolo3-master\VOCdevkit\VOC2007\JPEGImages/raw1.jpg'
    image=Image.open(path)
    r_image,_=yolo.detect_image(image)
    r_image.show()"""
    yolo.close_session()


