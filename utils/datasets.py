import glob
import random
import os
import numpy as np

import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.transform import resize

import sys

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_shape = (img_size, img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = np.array(Image.open(img_path))
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        return img_path, input_img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, data_folder, img_size=416):
        self.img_files = [data_folder + '/' + s for s in os.listdir(data_folder) if s.endswith('.jpg')]
        self.label_files = [path.replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50

    def __getitem__(self, index):

        index = 1

        #---------
        #  Image
        #---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = np.array(Image.open(img_path))

        # Handles images with less than three channels
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))

        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        image_before_tensor = input_img[:]  # used for visualiztion
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        #---------
        #  Label
        #---------

        # In cu-obb-roadway-features dataset (1K images train and test):
        #   class_id,
        #   x (centre, starts from image left limit),
        #   y (centre, starting from image upper limit),
        #   height (longest dimension),
        #   width (smallest dimension),
        #   orientation (/ 45 , \ -45 , | -90 or 90)
        # In coco 2014 dataset (82K+40K images train and test): class_id,x,y,width,height (all normalized from 0 to 1)

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path, delimiter=' ', skiprows=1)

            # Get OBB 4 vertices
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

            # Adjust vertices for added padding
            p1_x += pad[1][0]
            p2_x += pad[1][0]
            p3_x += pad[1][0]
            p4_x += pad[1][0]
            p1_y += pad[0][0]
            p2_y += pad[0][0]
            p3_y += pad[0][0]
            p4_y += pad[0][0]

            # Normalize origin for yolo GT to be from 0 to 1
            labels[:, 1] = (p1_x+p2_x+p3_x+p4_x) / (4*padded_w)
            labels[:, 2] = (p1_y+p2_y+p3_y+p4_y) / (4*padded_h)

            # Get height and width after padding, and normalize it, TODO: what OBB should give height normalization of 1?
            val1 = np.sqrt(((p2_x - p1_x) ** 2) + ((p2_y - p1_y) ** 2))
            val2 = np.sqrt(((p3_x - p2_x) ** 2) + ((p3_y - p2_y) ** 2))
            labels[:, 3] = np.min([val1, val2], axis=0) / (1.5*max(padded_h, padded_w))  # width
            labels[:, 4] = np.max([val1, val2], axis=0) / (1.5*max(padded_h, padded_w))  # height

            # For debugging
            visualize = False
            if visualize:
                import matplotlib as mpl
                import matplotlib.pyplot as plt
                import matplotlib.collections as collections
                from matplotlib.path import Path
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.imshow((image_before_tensor * 255.).astype(np.uint8))

                patches = []
                for i in range(labels.shape[0]):
                    verts = [(self.img_shape[0]*p1_x[i]/padded_w, self.img_shape[1]*p1_y[i]/padded_h),
                             (self.img_shape[0]*p2_x[i]/padded_w, self.img_shape[1]*p2_y[i]/padded_h),
                             (self.img_shape[0]*p3_x[i]/padded_w, self.img_shape[1]*p3_y[i]/padded_h),
                             (self.img_shape[0]*p4_x[i]/padded_w, self.img_shape[1]*p4_y[i]/padded_h),
                             (0., 0.), ]  # ignored
                    codes = [Path.MOVETO,
                             Path.LINETO,
                             Path.LINETO,
                             Path.LINETO,
                             Path.CLOSEPOLY, ]
                    path = Path(verts, codes)
                    patches.append(mpl.patches.PathPatch(path, linewidth=1, edgecolor='r', facecolor='none'))

                ax.add_collection(collections.PatchCollection(patches))
                plt.show(block=False)

        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 6))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        # filled_labels = torch.from_numpy(filled_labels[:,:5])  # to ignore orientation
        filled_labels = torch.from_numpy(filled_labels) # class_id,x,y,w,h,theta

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)
