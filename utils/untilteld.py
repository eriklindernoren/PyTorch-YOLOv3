#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:28:03 2019

@author: ron
"""
import os
from tqdm import tqdm
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

import cv2
import glob


THERMAL___BIT = "thermal_8_bit"

LABELS = "labels"

json_file = "/home/ron/Downloads/flir/FLIR_ADAS_1_3/train/thermal_annotations.json"
output_path = "/tmp/ron/tmp"

def plot_single_image(image_path, boxes):

    img = np.array(Image.open(image_path).convert('RGB'))
    img_h, img_w, _ = img.shape

    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for cls_pred, x1, y1, box_w, box_h in boxes:
        box_w = img_w* box_w
        box_h = img_h * box_h
        x1 = x1* img_w- box_w/2
        y1 = y1* img_h - box_h/2
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(bbox)
        plt.text(x1,y1,
                    s=classes[cls_pred],
                    color="white",
                    verticalalignment="top",
                    #bbox={"color": color, "pad": 0},
                )
    plt.show()
    plt.close()

classes = {1:"person", 2: "bicycle", 3: "car"}

def write_labels_file(json_file, output_path):

    with open(json_file, 'r') as f:
        data = json.load(f)
        converted_results = {}
        annotations = data['annotations']
        ids = {}
        for i in data["images"]:
            ids[i["id"]] = i
        for ann in tqdm(annotations):
            height = ids[ann["image_id"]]["height"]
            width = ids[ann["image_id"]]["width"]
    
            cat_id = int(ann['category_id'])-1
            if cat_id <= 2: # why?
                left, top, bbox_width, bbox_height = map(float, ann['bbox'])
                x_center, y_center = (left + bbox_width / 2, top + bbox_height / 2)
                # darknet expects relative values wrt image width&height
                x_rel, y_rel = (x_center / width, y_center / height)
                w_rel, h_rel = (bbox_width / width, bbox_height / height)
    
                converted_results_id = converted_results.get(ann["image_id"], [])
                converted_results_id.append((cat_id, x_rel, y_rel, w_rel, h_rel))
                converted_results[ann["image_id"]] = converted_results_id
    
        os.mkdir(os.path.join(output_path, LABELS))
        for k,v in converted_results.items():
            file_name = ids[k]["file_name"]

            file_out = os.path.join(output_path, file_name.replace("jpeg", 'txt').replace(THERMAL___BIT, LABELS))
            with open(file_out, 'w+') as fp:
                fp.write('\n'.join('%d %.6f %.6f %.6f %.6f' %  res for res in v))
        print("total files", os.listdir(output_path))
        for file_ in list(map(lambda x: x['file_name'], data["images"])):
            file_out = os.path.join(output_path, file_.replace("jpeg", 'txt').replace(THERMAL___BIT, LABELS))
            if not os.path.exists(file_out):
                print(file_out, "exisit")
                Path(file_out).touch()


def create_train_and_val_txt(image_path, prefix, output_path):
    list_dir = os.listdir(image_path)
    list_dir = list(map(lambda x:  os.path.join(prefix, x), list_dir))
    list_dir_train, list_dir_test = train_test_split(list_dir, test_size=0.1)
    with open(os.path.join(output_path, "train.txt"), "w") as f:
        for listitem in list_dir_train:
            f.write('%s\n' % listitem)

    with open(os.path.join(output_path, "val.txt"), "w") as f:
        for listitem in list_dir_test:
            f.write('%s\n' % listitem)


def plot_images(idx):
    l = os.listdir("/tmp/ron/labels")
    l.sort()
    f1 = l[idx]
    dest = "/home/ron/Downloads/flir/FLIR_ADAS_1_3/train/thermal_8_bit/"
    with open(os.path.join("/tmp/ron/labels", f1)) as f:
        image_path = os.path.join(dest, f1.replace(".txt", ""))
        boxes = []
        for l in f:
            classs, x1, y1, box_w, box_h = l.split(" ")
            boxes.append([int(classs), float(x1), float(y1), float(box_w), float(box_h)])
        plot_single_image(image_path, boxes)

def remove_file_no_label(txt_file, label_dir):
    labels = set(map(lambda x: x.replace(".txt", ""), os.listdir(label_dir)))

    with open(txt_file.replace(".txt", "2.txt"), "w") as filew:
        with open(txt_file, "r") as file:
    
            for line in file:
                file_name = line.split("/")[-1].split(".")[0]
                if file_name in labels:
                    filew.write(line)



def create_sorted_img_array(file_path):
    l = os.listdir(file_path)
    l = list(map(lambda x: os.path.join(file_path, x), l))
    l.sort()
    img_array = []
    for filename in tqdm(l):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    return img_array, size

def create_video_from_images(file_path, output):
    img_array, size = create_sorted_img_array(file_path)
    out = cv2.VideoWriter(output,cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def get_concat_v(img1, img2):
    vis = np.concatenate((img1, img2), axis=1)
    return vis


def create_2_video_from_images(file_path1, file_path2, output):
    img_array1, size = create_sorted_img_array(file_path1)
    img_array2, size = create_sorted_img_array(file_path2)
    print(len(img_array1), len(img_array2))
    img_array1 = img_array1[1:1300]
    img_array2 = img_array2[1:1300]
    all_array = []
    for img1, img2 in zip(img_array1, img_array2):
        all_array.append(get_concat_v(img1, img2))

    height, width, layers = all_array[0].shape
    size = (width,height)
    out = cv2.VideoWriter(output,cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
     
    for i in range(len(all_array)):
        out.write(all_array[i])
    out.release()



def create_video_from_images(file_path, output):
    img_array, size = create_sorted_img_array(file_path)
    out = cv2.VideoWriter(output,cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


create_video_from_images("/home/ubuntu/PyTorch-YOLOv3/output", "/tmp/a3.avi")
# =============================================================================
# r = os.listdir("/home/ubuntu/PyTorch-YOLOv3/data/thermal/images")
# r = list(map(lambda x: os.path.join("data/thermal/images", x), r))
# with open("/home/ubuntu/PyTorch-YOLOv3/data/thermal/train.txt", "w") as f:
#     for listitem in r:
#         f.write('%s\n' % listitem)
# 
# with open("/tmp/ron/train.txt", "r") as f:
#     with open("/tmp/ron/train2.txt", "w") as g:
#         for ff in f:
#             g.write(ff.replace("data/thermal/images/", "/tmp/ron/images/"))
#
# 
# =============================================================================

