from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--image_folder', type=str, default='data/samples', help='path to dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of each image batch')
parser.add_argument('--model_config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
parser.add_argument('--data_config_path', type=str, default='config/coco.data', help='path to data config file')
parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
parser.add_argument('--iou_thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available else False

# Get data configuration
data_config     = parse_data_config(opt.data_config_path)
test_path       = data_config['valid']

# Initiate model
model = Darknet(opt.model_config_path)
model.load_weights(opt.weights_path)

if cuda:
    model = model.cuda()

model.eval()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(test_path),
    batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
num_classes = 80
num_batches = len(dataloader)

# template to output progress
num_batch_digits = len(str(num_batches))
output_template = 'Batch [{0: ' + str(num_batch_digits) + '}/{1}]'

annotations = []
detections  = []
for batch_i, (_, imgs, targets) in enumerate(dataloader):
    imgs = Variable(imgs.type(Tensor))
    targets = targets.type(Tensor)

    batch_size = imgs.shape[0]

    with torch.no_grad():
        output = model(imgs)
        output = non_max_suppression(output, num_classes, conf_thres=0.2)

    for sample_i in range(batch_size):
        # add targets to annotations:
        # 1. convert from (center, size) to (top-left, bottom-right)
        # 2. convert from unnormalized coordinates
        # 3. split into groups corresponding to class labels
        target_sample = targets[sample_i, targets[sample_i, :, 3] != 0]
        if target_sample.size(0) == 0: # no annotated objects
            annotations.append([np.empty((0, 5)) for label in range(num_classes)])
        else:
            target_sample[:, 1:3]  = target_sample[:, 1:3] - target_sample[:, 3:5] / 2
            target_sample[:, 3:5] += target_sample[:, 1:3]
            target_sample[:, 1:5] *= opt.img_size
            annotations.append([target_sample[target_sample[:, 0] == label, 1:5]
                    for label in range(num_classes)])
        # add output to detections:
        # 1. split into groups corresponding to class labels
        sample_pred = output[sample_i]
        if sample_pred is None: # no detected objects
            detections.append([np.empty((0, 5)) for label in range(num_classes)])
        else:
            detections.append([sample_pred[sample_pred[:, -1] == label, :5]
                    for label in range(num_classes)])
    print(output_template.format(batch_i, num_batches), end='\r')
print()

ap = evaluate(detections, annotations, opt.iou_thres)
ap_list = [v for k, v in ap.items()]
print ('Mean Average Precision: {:.5f}'.format(sum(ap_list) / num_classes))
