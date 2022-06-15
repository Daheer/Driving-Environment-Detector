#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
from PIL import ImageFont, ImageDraw, Image
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

from tensorflow.keras.models import load_model
from yad2k.models.keras_yolo import yolo_head
from yad2k.utils.utils import draw_boxes, get_colors_for_classes, scale_boxes, read_classes, read_anchors, preprocess_image

# In[2]:


def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = .6):
    
    box_scores = np.multiply(box_class_probs, box_confidence)
    box_classes = tf.math.argmax(box_scores, axis = -1)
    box_class_scores = tf.math.reduce_max(box_scores, axis = -1)
    
    filtering_mask = box_class_scores >= threshold
    
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)    
    
    return scores, boxes, classes


# In[3]:


def iou(box1, box2):
    
    # IOU = Intersection Over Union
    
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2
    
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    
    inter_width = xi2 - xi1
    inter_height =  yi2 - yi1
    
    inter_area = max(inter_width, 0) * max(inter_height, 0)
    
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area
    
    return iou


# In[4]:


def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    
    max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')  

    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)
    
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)

    
    return scores, boxes, classes


# In[5]:


def yolo_boxes_to_corners(box_xy, box_wh):
    
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return tf.keras.backend.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])


# In[6]:


def yolo_eval(yolo_outputs, image_shape = (720, 1280), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    
    scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs)
    
    boxes = scale_boxes(boxes, image_shape)
    
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)
    
    return scores, boxes, classes


# In[7]:


class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
model_image_size = (608, 608)


# In[8]:


yolo_model = load_model("model_data/", compile=False)


# In[9]:


"""yolo_model.summary()"""


# In[10]:


def predict(image_file):

    image, image_data = preprocess_image("frames/" + image_file, model_image_size = (608, 608))
    
    yolo_model_outputs = yolo_model(image_data)
    yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))
    
    out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, [image.size[1],  image.size[0]], 10, 0.3, 0.5)

    colors = get_colors_for_classes(len(class_names))

    draw_boxes(image, out_boxes, out_classes, class_names, out_scores)
    
    image.save(os.path.join("frames_out", image_file), quality=100)
    
    output_image = Image.open(os.path.join("frames_out", image_file))

    return output_image


# In[11]:


main_folder = os.getcwd()


# In[12]:


import cv2

os.chdir(main_folder)
video_path = ''
for filename in os.listdir(main_folder):
    if filename.endswith(".mov") or filename.endswith(".mp4") or filename.endswith(".mkv") or filename.endswith(".avi"):
        video_path = filename
video_handle = cv2.VideoCapture(video_path)
frame_no = 0
steps = 1

while True:  
    eof, frame = video_handle.read()  
    if not eof:      
        break 
    if (frame_no % 5 == 0):
        cv2.imwrite("frames/0%d.jpg" % steps, frame)  
        steps += 1
    frame_no += 1

video_handle.release()

# In[13]:


from ipywidgets import IntProgress
from IPython.display import display

n_frames = frame_no
bar = IntProgress(min = 0, max = steps)
display(bar)

directory = 'frames/'
for filename in os.listdir(directory):
    if (bar.value < n_frames):
        bar.value += 1
    if filename.endswith(".jpg"):
        output_image = predict(filename)


# In[14]:


output_image.close()


# In[15]:


os.chdir(main_folder)
path_parent = os.path.dirname(os.getcwd())
os.chdir('frames_out/')  
path = os.getcwd()
  
mean_height = 0
mean_width = 0
  
num_of_images = len(os.listdir('.'))
  
for file in os.listdir('.'):
    if file.endswith(".jpg"):
        im = Image.open(os.path.join(path, file))

        width, height = im.size
        mean_width += width
        mean_height += height
im.close()
    
mean_width = int(mean_width / num_of_images)
mean_height = int(mean_height / num_of_images)

for file in os.listdir('.'):
    if file.endswith(".jpg"):

        im = Image.open(os.path.join(path, file))
   
        width, height = im.size   

        imResize = im.resize((mean_width, mean_height), Image.Resampling.LANCZOS) 
        imResize.save( file, 'JPEG', quality = 95)


# In[16]:


def generate_video():
    image_folder = '.'
    video_name = os.path.join(main_folder, 'out/video_output.mp4')
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)
    os.chdir('frames_out/') 
        
    images = [img for img in os.listdir(image_folder)
              if img.endswith(".jpg")
            ]
    
    import re
    def sorted_alphanumeric(data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(data, key=alphanum_key)
    images = sorted_alphanumeric(images)

    video = cv2.VideoWriter(video_name, 0, 5, (mean_width, mean_height)) 
    
    for image in images: 
        video.write(cv2.imread(os.path.join(image_folder, image))) 
        
    cv2.destroyAllWindows() 
    video.release()
    
generate_video()


# In[17]:


os.chdir(main_folder)
os.chdir('frames')
for file in os.listdir('.'):
    if file.endswith(".jpg"):
        os.remove(file)
os.chdir(main_folder)
os.chdir('frames_out')
for file in os.listdir('.'):
    if file.endswith(".jpg"):
        os.remove(file)


# In[ ]:




