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
    def __init__(self, list_path, img_size=416):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50

    def __getitem__(self, index):

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
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        #---------
        #  Label
        #---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)
            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)
    

    
    
class TransformVOCDetectionAnnotation(object):
    def __init__(self, keep_difficult=False):
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        res = []
        
        for obj in target.iter('object'):
            tmp = []
            #difficult = int(obj.find('difficult').text) == 1
            #if not self.keep_difficult and difficult:
            #    continue
            #name = obj.find('name').text
            name = obj[0].text.lower().strip()
            #print(type(name))
            #bb = obj.find('bndbox')
            bbox = obj[1]
            #print(bbox)
            #bndbox = [bb.find('xmin').text, bb.find('ymin').text,
            #    bb.find('xmax').text, bb.find('ymax').text]
            # supposes the order is xmin, ymin, xmax, ymax
            # attention with indices
            bndbox = [int(bb.text)-1 for bb in bbox]
            if name == 'seacucumber':
                name = 0
            elif name == 'seaurchin':
                name = 1
            else:
                name = 2
            tmp.append(copy.deepcopy(float(name)))
            tmp.append(copy.deepcopy(float(bndbox[0])))
            tmp.append(copy.deepcopy(float(bndbox[1])))
            tmp.append(copy.deepcopy(float(bndbox[2])))
            tmp.append(copy.deepcopy(float(bndbox[3])))
            res.append(copy.deepcopy(tmp))
        return res

    
    
# new code here, here is a pascal_voc dataloader implementation
class VOCDetection(data.Dataset):
    def __init__(self, root, image_set, transform=None, target_transform=None, img_size = 416):
        """
            root: refers to the path you save the pascal_voc liked data;
            imagenet: the subset such as 'train', 'val', 'test'
        """
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform

        dataset_name = 'VOC2007'
        self._annopath = os.path.join(self.root, dataset_name, 'Annotations', '%s.xml')
        self._imgpath = os.path.join(self.root, dataset_name, 'JPEGImages', '%s.jpg')
        self._imgsetpath = os.path.join(self.root, dataset_name, 'ImageSets', 'Main', '%s.txt')
 
        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]
        self.max_objects = 100
        self.img_shape = (img_size, img_size)

    def __getitem__(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = np.array(Image.open(self._imgpath % img_id).convert('RGB'))
        #print(self._imgpath % img_id)
        
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        if self.target_transform is not None:
            target = self.target_transform(target)
        labels = None
        """
        if len(target) == 1:
            print('{} only contain one object'.format(img_id))
            label =  np.array(target)
            cls = label[0,0]
            x1 = label[0,1]
            y1 = label[0,2]
            x2 = label[0,3]
            y2 = label[0,4]

            labels = np.zeros((1,5))
            labels[0,0] = cls
            labels[0,1] = (x1 + x2) / (2 * w)
            labels[0,2] = (y1 + y2) / (2 * h)
            labels[0,3] = (x2 - x1) / w
            labels[0,4] = (y2 - y1) / h



            x1 = w * (labels[0, 1] - labels[0, 3]/2)
            y1 = h * (labels[0, 2] - labels[0, 4]/2)
            x2 = w * (labels[0, 1] + labels[0, 3]/2)
            y2 = h * (labels[0, 2] + labels[0, 4]/2)

            
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]

            labels[0,1] = ((x1 + x2) / 2) / padded_w
            labels[0,2] = ((y1 + y2) / 2) / padded_h
            labels[0,3] *= float(w) / float(padded_w)
            labels[0,4] *= float(h) / float(padded_h)
            
        """ 
        if len(target) > 0:
            target = np.array(target)
            x1 = target[:,1]
            y1 = target[:,2]
            x2 = target[:,3]
            y2 = target[:,4]
            # transform the pascal form label to coco form
            labels = np.zeros((target.shape))
            labels[:,0] = target[:,0]
            labels[:,1] = (x1 + x2) / (2 * w)
            labels[:,2] = (y1 + y2) / (2 * h)
            labels[:,3] = (x2 - x1) / w
            labels[:,4] = (y2 - y1) / h
            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)
               
            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= float(w) / float(padded_w)
            labels[:, 4] *= float(h) / float(padded_h)
        
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        else:
            print('no object')
        filled_labels = torch.from_numpy(filled_labels)

        return input_img, filled_labels

    def __len__(self):
        return len(self.ids)
