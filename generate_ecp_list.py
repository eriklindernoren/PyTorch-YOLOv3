from __future__ import division

import os
from natsort import natsorted

import sys
import time
import datetime
import argparse

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def get_imgs_labels_lists(path):
    ecp_img_train = os.path.join(path, "img/train")
    ecp_labels_train = os.path.join(path, "labels/train")
    ecp_img_val = os.path.join(path, "img/val")
    ecp_labels_val = os.path.join(path, "labels/val")
    ecp_img_test = os.path.join(path, "img/test")
    
    img_train_list = []
    labels_train_list = []
    img_val_list = []
    labels_val_list = []
    img_test_list = []

    for location in os.listdir(ecp_img_train):
        folder = os.path.join(ecp_img_train, location)
        for file in os.listdir(folder):
            img_train_list.append(os.path.join(ecp_img_train, location, file))
    for location in os.listdir(ecp_labels_train):
        folder = os.path.join(ecp_labels_train, location)
        for file in os.listdir(folder):
            labels_train_list.append(os.path.join(ecp_labels_train, location, file))

    for location in os.listdir(ecp_img_val):
        folder = os.path.join(ecp_img_val, location)
        for file in os.listdir(folder):
            img_val_list.append(os.path.join(ecp_img_val, location, file))
    for location in os.listdir(ecp_labels_val):
        folder = os.path.join(ecp_labels_val, location)
        for file in os.listdir(folder):
            labels_val_list.append(os.path.join(ecp_labels_val, location, file))

    for location in os.listdir(ecp_img_test):
        folder = os.path.join(ecp_img_test, location)
        for file in os.listdir(folder):
            img_test_list.append(os.path.join(ecp_img_test, location, file))
    img_train_list = natsorted(img_train_list)
    labels_train_list = natsorted(labels_train_list)
    img_val_list = natsorted(img_val_list)
    labels_val_list = natsorted(labels_val_list)
    img_test_list = natsorted(img_test_list)
    return img_train_list, labels_train_list, img_val_list, labels_val_list, img_test_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_ecp", type=str, default="/home/dataset/ECP/ECP/day", help="path to ecp dataset directory. day or night.")
    parser.add_argument("--path_out", type=str, default="data/ecp", help="output path for .txt files")
    opt = parser.parse_args()
    print(opt)
    if "day" in os.listdir(opt.path_ecp):
        print("Use both day and night dataset...")
        img_train_list, labels_train_list, img_val_list, labels_val_list, img_test_list = get_imgs_labels_lists(os.path.join(opt.path_ecp,"day"))
        night1, night2, night3, night4, night5 = get_imgs_labels_lists(os.path.join(opt.path_ecp,"night"))
        img_train_list.extend(night1)
        labels_train_list.extend(night2)
        img_val_list.extend(night3)
        labels_val_list.extend(night4)
        img_test_list.extend(night5)
        img_train_list = natsorted(img_train_list)
        labels_train_list = natsorted(labels_train_list)
        img_val_list = natsorted(img_val_list)
        labels_val_list = natsorted(labels_val_list)
        img_test_list = natsorted(img_test_list)
    else:
        img_train_list, labels_train_list, img_val_list, labels_val_list, img_test_list = get_imgs_labels_lists(opt.path_ecp)

    path_w = os.path.join(opt.path_out, './train.txt')
    with open(path_w, mode='w') as f:
        for file in img_train_list:
            f.write(file)
            f.write("\n")
    path_w = os.path.join(opt.path_out, './train_labels.txt')
    with open(path_w, mode='w') as f:
        for file in labels_train_list:
            f.write(file)
            f.write("\n")
    path_w = os.path.join(opt.path_out, './valid.txt')
    with open(path_w, mode='w') as f:
        for file in img_val_list:
            f.write(file)
            f.write("\n")
    path_w = os.path.join(opt.path_out, './valid_labels.txt')
    with open(path_w, mode='w') as f:
        for file in labels_val_list:
            f.write(file)
            f.write("\n")
    path_w = os.path.join(opt.path_out, './test.txt')
    with open(path_w, mode='w') as f:
        for file in img_test_list:
            f.write(file)
            f.write("\n")
