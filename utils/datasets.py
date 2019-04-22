import glob
import random
import os
import numpy as np
import lycon
import torch

from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.transform import resize

import sys


def pad_to_square(image_np, constant_value):
    h, w = image_np.shape[:2]
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    new_shape = (max([h, w]), max([h, w]), 3)
    padded_image = np.full(new_shape, constant_value, dtype=np.float32)
    if h < w:
        padded_image[pad1 : pad1 + h, :] = image_np
        padding = (0, pad1, 0, pad2)
    else:
        padded_image[:, pad1 : pad1 + w] = image_np
        padding = (pad1, 0, pad2, 0)
    return padded_image, padding


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
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
        input_img = np.pad(img, pad, "constant", constant_values=127.5) / 255.0
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode="reflect")
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        return img_path, input_img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()
        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = lycon.load(img_path)

        # Handles images with less than three channels
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = lycon.load(img_path)

        h, w, _ = img.shape
        # Pad to square resolution
        img, pad = pad_to_square(img, 127.5)
        padded_h, padded_w, _ = img.shape
        # Resize to target shape
        img = lycon.resize(img, height=self.img_shape[0], width=self.img_shape[1])
        # Channels-first and normalize
        input_img = torch.from_numpy(img).float().permute((2, 0, 1)) / 255

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3] / 2)
            y1 = h * (labels[:, 2] - labels[:, 4] / 2)
            x2 = w * (labels[:, 1] + labels[:, 3] / 2)
            y2 = h * (labels[:, 2] + labels[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[1]
            x2 += pad[2]
            y2 += pad[3]
            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h

        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[: self.max_objects]] = labels[: self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)
