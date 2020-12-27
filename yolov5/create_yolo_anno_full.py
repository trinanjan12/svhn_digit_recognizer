import json
import datetime
import pickle
from glob import glob as glob
import h5py
import numpy as np
import os
from tqdm import tqdm as tqdm
import random
import shutil
from PIL import Image

mode = "combined"
with open('../../combined_train_extra_annotations_processed.pkl', 'rb') as f:
    annotations = pickle.load(f)

output_dir = './svhn2020full'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(output_dir + '/images'):
    os.makedirs(output_dir+ '/images')
if not os.path.exists(output_dir + '/images/train2020'):
    os.makedirs(output_dir+ '/images/train2020')
if not os.path.exists(output_dir + '/images/val2020'):
    os.makedirs(output_dir+ '/images/val2020')

if not os.path.exists(output_dir + '/labels'):
    os.makedirs(output_dir+ '/labels')
if not os.path.exists(output_dir + '/labels/train2020'):
    os.makedirs(output_dir+ '/labels/train2020')
if not os.path.exists(output_dir + '/labels/val2020'):
    os.makedirs(output_dir+ '/labels/val2020')
    
random.seed(2020)
img_name_keys = list(annotations.keys())
random.shuffle(img_name_keys)

split = int(.9 * len(img_name_keys))
train_img_name_keys = img_name_keys[0:split+1]
val_img_name_keys = img_name_keys[split:-1]

date = str(datetime.datetime.now())

for image_name in tqdm(train_img_name_keys):
    img_index = int(image_name.split('.')[0])
    img_path = '../../' + mode + '/' + image_name
    img = Image.open(img_path)
    width, height = img.size[0], img.size[1]

    img_bbox = annotations[image_name]
    filename = output_dir + '/labels/train2020/' + str(img_index) + '.txt'
    with open(filename, 'a') as f:
        for i in range(0, len(img_bbox['label'])):
            x0, y0, w, h = img_bbox['left'][i], img_bbox['top'][i], img_bbox[
                'width'][i], img_bbox['height'][i]
            x_center, y_center, bbox_width, bbox_height = x0 + w / 2, y0 + h / 2, w, h
            x_center, y_center = x_center / width, y_center / height,
            bbox_width, bbox_height = bbox_width / width, bbox_height / height
            class_id = img_bbox['label'][i]
            bbox_mess = ' '.join([str(class_id), str(x_center), str(y_center), str(bbox_width), str(bbox_height)]) + '\n'
            f.write(bbox_mess)
    shutil.copyfile(img_path, output_dir + '/images/train2020/' + image_name)

for image_name in tqdm(val_img_name_keys):
    img_index = int(image_name.split('.')[0])
    img_path = '../../' + mode + '/' + image_name
    img = Image.open(img_path)
    width, height = img.size[0], img.size[1]

    img_bbox = annotations[image_name]
    filename = output_dir + '/labels/val2020/' + str(img_index) + '.txt'
    with open(filename, 'a') as f:
        for i in range(0, len(img_bbox['label'])):
            x0, y0, w, h = img_bbox['left'][i], img_bbox['top'][i], img_bbox[
                'width'][i], img_bbox['height'][i]
            x_center, y_center, bbox_width, bbox_height = x0 + w / 2, y0 + h / 2, w, h
            x_center, y_center = x_center / width, y_center / height,
            bbox_width, bbox_height = bbox_width / width, bbox_height / height
            class_id = img_bbox['label'][i]
            bbox_mess = ' '.join([str(class_id), str(x_center), str(y_center), str(bbox_width), str(bbox_height)]) + '\n'
            f.write(bbox_mess)
    shutil.copyfile(img_path, output_dir + '/images/val2020/' + image_name)