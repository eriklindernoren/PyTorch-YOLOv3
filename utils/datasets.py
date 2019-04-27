import glob
import random
import os
import sys
import numpy as np
import lycon
import torch
import torch.nn.functional as F
from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms


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
        img = lycon.load(img_path)
        img, _ = pad_to_square(img, 127)
        # Resize
        img = lycon.resize(img, height=self.img_size, width=self.img_size, interpolation=lycon.Interpolation.NEAREST)
        # Channels-first
        img = np.transpose(img, (2, 0, 1))
        # As pytorch tensor
        img = torch.from_numpy(img).float() / 255.0

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, training=True, augment=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.is_training = training
        self.augment = augment and training

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
        img = torch.from_numpy(img).float().permute((2, 0, 1)) / 255.0

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            labels = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3] / 2)
            y1 = h * (labels[:, 2] - labels[:, 4] / 2)
            x2 = w * (labels[:, 1] + labels[:, 3] / 2)
            y2 = h * (labels[:, 2] + labels[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][1]
            y2 += pad[0][1]

            if self.is_training:
                # Returns (x, y, w, h)
                labels[:, 1] = ((x1 + x2) / 2) / padded_w
                labels[:, 2] = ((y1 + y2) / 2) / padded_h
                labels[:, 3] *= w / padded_w
                labels[:, 4] *= h / padded_h
            else:
                # Returns (x1, y1, x2, y2)
                labels[:, 1] = x1 * (self.img_size / padded_w)
                labels[:, 2] = y1 * (self.img_size / padded_h)
                labels[:, 3] = x2 * (self.img_size / padded_w)
                labels[:, 4] = y2 * (self.img_size / padded_h)

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, labels = horisontal_flip(img, labels)

        # Add dummy label if there are none
        num_labels = 1 if labels is None else len(labels)
        boxes = torch.zeros((num_labels, 6))
        if labels is not None:
            boxes[:, 1:] = labels

        return img_path, img, boxes

    @staticmethod
    def collate_fn(batch):
        paths, imgs, labels = list(zip(*batch))
        for i, boxes in enumerate(labels):
            boxes[:, 0] = i
        imgs = torch.stack(imgs, 0)
        labels = torch.cat(labels, 0)
        return paths, imgs, labels

    def __len__(self):
        return len(self.img_files)
