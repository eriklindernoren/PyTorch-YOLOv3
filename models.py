from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from PIL import Image

from utils import *

def parse_config(path):
    """Parses the yolo-v3 layer configuration file and returns block definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    blocks = []
    for line in lines:
        if line.startswith('['):        # This marks the start of a new block
            blocks.append({})           # re-init the block
            blocks[-1]['type'] = line[1:-1].rstrip()
            if blocks[-1]['type'] == 'convolutional':
                blocks[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            blocks[-1][key.rstrip()] = value.lstrip()

    return blocks

class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""
    def __init__(self):
        super(EmptyLayer, self).__init__()

class YOLOLayer(nn.Module):
    """Detection layer"""
    def __init__(self, anchors, num_classes, image_dim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_dim = image_dim

    def forward(self, x, cuda=True):
        batch_size = x.size(0)
        x_dim = x.size(2)
        stride =  self.image_dim // x_dim

        prediction = x.view(batch_size, self.bbox_attrs * self.num_anchors, x_dim * x_dim)
        prediction = prediction.transpose(1, 2).contiguous()
        prediction = prediction.view(batch_size, x_dim * x_dim * self.num_anchors, self.bbox_attrs)

        # Sigmoid
        prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])    # Center X
        prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])    # Center Y
        prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])    # Object conf.

        grid = np.arange(x_dim)
        a, b = np.meshgrid(grid, grid)

        x_offset = torch.FloatTensor(a).view(-1, 1)
        y_offset = torch.FloatTensor(b).view(-1, 1)

        if cuda:
            x_offset = x_offset.cuda()
            y_offset = y_offset.cuda()

        x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, self.num_anchors).view(-1, 2).unsqueeze(0)
        # Add offsets to center x and center y
        prediction[:, :, :2] += x_y_offset

        anchors = torch.FloatTensor([(a_h / stride, a_w / stride) for a_h, a_w in self.anchors])
        if cuda:
            anchors = anchors.cuda()

        # Scale width and height by anchors
        anchors = anchors.repeat(x_dim * x_dim, 1).unsqueeze(0)
        prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors
        # Apply sigmoid to class scores
        prediction[:, :, 5:self.bbox_attrs] = torch.sigmoid((prediction[:, :, 5:self.bbox_attrs]))
        # Rescale output dimension to image size
        prediction[:, :, :4] *= stride

        return prediction

def create_modules(blocks):
    """
    Constructs module list of layer blocks from block configuration in 'blocks'
    """
    hyperparams = blocks.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    for i, block in enumerate(blocks):
        modules = nn.Sequential()

        if block['type'] == 'convolutional':
            bn = int(block['batch_normalize'])
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            pad = (kernel_size - 1) // 2 if int(block['pad']) else 0
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(block['stride']),
                                                        padding=pad,
                                                        bias=not bn))
            if bn:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
            if block['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1, inplace=True))

        elif block['type'] == 'upsample':
            upsample = nn.Upsample(scale_factor=int(block['stride']), mode='bilinear')
            modules.add_module('upsample_%d' % i, upsample)

        elif block['type'] == 'route':
            layers = [int(x) for x in block["layers"].split(',')]
            filters = sum([output_filters[layer_i] for layer_i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif block['type'] == 'shortcut':
            filters = output_filters[int(block['from'])]
            modules.add_module("shortcut_%d" % i, EmptyLayer())

        elif block["type"] == "yolo":
            mask = [int(x) for x in block["mask"].split(",")]
            # Extract anchors corresponding to mask indices
            anchors = [int(x) for x in block["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, int(block['classes']), int(hyperparams['height']))
            modules.add_module('yolo_%d' % i, yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)
    return hyperparams, module_list


class Darknet(nn.Module):
    """YOLO-V3 object detection model"""
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.blocks = parse_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.blocks)
        self.img_size = img_size

    def forward(self, x, cuda=True):
        detections = None
        outputs = []
        for i, (module_def, module) in enumerate(zip(self.blocks, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample']:
                x = module(x)
            elif module_def['type'] == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                if len(layer_i) == 1:    # Output
                    x = outputs[layer_i[0]]
                else:                   # Skip-connection
                    x = torch.cat((outputs[layer_i[0]], outputs[layer_i[1]]), 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = outputs[-1] + outputs[layer_i]
            elif module_def['type'] == 'yolo':
                x = module[0](x, cuda)
                # Add to detections
                detections = x if detections is None else torch.cat((detections, x), 1)

            outputs.append(x)

        return detections

    def load_weights(self, weights_path):
        """Parses and copies the weights stored in 'weights_path' to the model"""

        #Open the weights file
        fp = open(weights_path, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)   # First five are header values
        weights = np.fromfile(fp, dtype=np.float32)         # The rest are weights
        fp.close()

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.blocks, self.module_list)):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                if module_def['batch_normalize']:
                    bn_layer = module[1]
                    # Number of biases
                    num_b = bn_layer.bias.numel()
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Number of biases
                    num_b = conv_layer.bias.numel()
                    # Conv Bias
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr = ptr + num_b
                # Number of weights
                num_w = conv_layer.weight.numel()
                # Conv Weights
                conv_w = torch.from_numpy(weights[ptr:ptr+num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr = ptr + num_w
