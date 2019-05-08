import glob
import random
import os
import numpy as np
import lycon
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.transform import resize

import sys


def pad_to_square(img, pad_value):
    h, w, _ = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1 = dim_diff // 2
    pad2 = dim_diff - pad1
    # Determine padding
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    # Add padding
    img = np.pad(img, pad, "constant", constant_values=pad_value)

    return img, pad

# for rescaling traning 
def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = np.array(Image.open(img_path))
        input_img, _ = pad_to_square(img, 127.5)
        # Resize
        input_img = lycon.resize(
            input_img, height=self.img_size, width=self.img_size, interpolation=lycon.Interpolation.NEAREST
        )
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float() / 255.0

        return img_path, input_img

    def __len__(self):
        return len(self.files)




# the images and the labels in the data is not scaled or normalized   
class ListDataset(Dataset):
    def __init__(self, data_folder, img_size=416, training=True, val = False,split_at = 800):
        if not val : self.img_files = [data_folder + '/' + im for im in os.listdir(data_folder) if im.endswith('.jpg')][:split_at]
        else : self.img_files = [data_folder + '/' + im for im in os.listdir(data_folder) if im.endswith('.jpg')][split_at:]
        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 50
        self.is_training = training

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = lycon.load(img_path)

        # Handles images with less than three channels
        if len(img.shape) != 3:
            img = np.expand_dims(img, -1)
            img = np.repeat(img, 3, -1)

        h, w, _ = img.shape
        img, pad = pad_to_square(img, 127.5)
        padded_h, padded_w, _ = img.shape
        # Resize to target shape
        img = lycon.resize(img, height=self.img_size, width=self.img_size)
        # Channels-first and normalize
        input_img = torch.from_numpy(img / 255).float().permute((2, 0, 1))

        # ---------
        #   Label
        #   the labels of the data set contains 6 unnormalized numbers.
        #   the object number
        #   x,y : the center of the oriented BBox (x from the left limit, y from upper limit)
        #   lenght (longest dimension)
        #   width  (shortest dimension)
        #   orientation (- 0, / 45 , \ -45 , | -90 or 90)
        #   return target_label, x,y (center), w, le, theta. (all normalized)
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path, delimiter=' ', skiprows=1).reshape(-1, 6)
            
            if len(labels.shape) == 1:  # only 1 object
                labels = labels.reshape(1,-1)
            
            # get the 4 vertices for the OBB of the unpadded + unscaled image.
            # TODO : the theory behind the equations 
            p1_x = labels[:,1] + labels[:,3] * np.cos(np.radians(labels[:,5]     )) / 2.0 + \
                                 labels[:,4] * np.cos(np.radians(90 + labels[:,5])) / 2.0
            p1_y = labels[:,2] - labels[:,3] * np.sin(np.radians(labels[:,5]     )) / 2.0 - \
                                 labels[:,4] * np.sin(np.radians(90 + labels[:,5])) / 2.0

            p2_x = labels[:,1] - labels[:,3] * np.cos(np.radians(labels[:,5]     )) / 2.0 + \
                                 labels[:,4] * np.cos(np.radians(90 + labels[:,5])) / 2.0
            p2_y = labels[:,2] + labels[:,3] * np.sin(np.radians(labels[:,5]     )) / 2.0 - \
                                 labels[:,4] * np.sin(np.radians(90 + labels[:,5])) / 2.0

            p3_x = labels[:,1] - labels[:,3] * np.cos(np.radians(labels[:,5]     )) / 2.0 - \
                                 labels[:,4] * np.cos(np.radians(90 + labels[:,5])) / 2.0
            p3_y = labels[:,2] + labels[:,3] * np.sin(np.radians(labels[:,5]     )) / 2.0 + \
                                 labels[:,4] * np.sin(np.radians(90 + labels[:,5])) / 2.0

            p4_x = labels[:,1] + labels[:,3] * np.cos(np.radians(labels[:,5]     )) / 2.0 - \
                                 labels[:,4] * np.cos(np.radians(90 + labels[:,5])) / 2.0
            p4_y = labels[:,2] - labels[:,3] * np.sin(np.radians(labels[:,5]     )) / 2.0 + \
                                 labels[:,4] * np.sin(np.radians(90 + labels[:,5])) / 2.0

            # Adjust for added padding
            p1_x += pad[1][0]
            p2_x += pad[1][0]
            p3_x += pad[1][0]
            p4_x += pad[1][0]
            p1_y += pad[0][0]
            p2_y += pad[0][0]
            p3_y += pad[0][0]
            p4_y += pad[0][0]

            # Returns (x, y, w, h)
            # get the center of the scaled image and normalize it [0, 1]
            labels[:, 1] = ((p1_x+p2_x+p3_x+p4_x) / 4) / padded_w
            labels[:, 2] = ((p1_y+p2_y+p3_y+p4_y) / 4) / padded_h
            # normalize the width and lenght 
            diagonal_lenght = np.sqrt(padded_h**2 + padded_w**2)
            # get the width and lenght after padding and scalling
            # normalize width and lenght with the diagonal[0, 1] 
            dim1 = np.sqrt((p2_x-p1_x)**2 + (p2_y-p1_y)**2)
            dim2 = np.sqrt((p3_x-p2_x)**2 + (p3_y-p2_y)**2)                
            labels[:, 3] = np.min([dim1, dim2], axis=0) / diagonal_lenght # width
            labels[:, 4] = np.max([dim1, dim2], axis=0) / diagonal_lenght # lenght
            # normalize theta [-1, 1]
            labels[:, 5] /= 90
        
        # Fill matrix
        labels = torch.from_numpy( labels)
        filled_labels = torch.zeros((self.max_objects, 6)) # label, x,y , w, le, theta 
        if labels is not None:
            labels = labels[: self.max_objects]
            filled_labels[: len(labels)] = labels

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)
