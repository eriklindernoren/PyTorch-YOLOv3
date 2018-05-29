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

n_gt = 0
correct = 0
for batch_i, (_, imgs, targets) in enumerate(dataloader):
    imgs = Variable(imgs.type(Tensor))
    targets = targets.type(Tensor)

    with torch.no_grad():
        output = model(imgs)
        output = non_max_suppression(output, 80, conf_thres=0.2)

    for sample_i in range(targets.size(0)):
        # Get labels for sample where width is not zero (dummies)
        target_sample = targets[sample_i, targets[sample_i, :, 3] != 0]
        for obj_cls, tx, ty, tw, th in target_sample:
            # Get rescaled gt coordinates
            tx1, tx2 = opt.img_size * (tx - tw / 2), opt.img_size * (tx + tw / 2)
            ty1, ty2 = opt.img_size * (ty - th / 2), opt.img_size * (ty + th / 2)
            n_gt += 1
            box_gt = torch.cat([coord.unsqueeze(0) for coord in [tx1, ty1, tx2, ty2]]).view(1, -1)
            sample_pred = output[sample_i]
            if sample_pred is not None:
                # Iterate through predictions where the class predicted is same as gt
                for x1, y1, x2, y2, conf, obj_conf, obj_pred in sample_pred[sample_pred[:, 6] == obj_cls]:
                    box_pred = torch.cat([coord.unsqueeze(0) for coord in [x1, y1, x2, y2]]).view(1, -1)
                    iou = bbox_iou(box_pred, box_gt)
                    if iou >= opt.iou_thres:
                        correct += 1
                        break

    if n_gt:
        print ('Batch [%d/%d] mAP: %.5f' % (batch_i, len(dataloader), float(correct / n_gt)))


print ('Mean Average Precision: %.5f' % float(correct / n_gt))
