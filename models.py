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
    lines = file.read().split('\n')                        # store the lines in a list
    lines = [x for x in lines if len(x) > 0]               # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']              # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line.startswith('['):        # This marks the start of a new block
            if block:                   # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)    # add it the blocks list
                block = {}              # re-init the block
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    """Placeholder for the 'yolo' layer"""
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def create_modules(blocks):
    """
    Constructs module list of layer blocks from block configuration in 'blocks'
    """
    hyperparams = blocks[0]
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    for i, block in enumerate(blocks[1:]):
        modules = nn.Sequential()

        if block['type'] == 'convolutional':
            bn = True if 'batch_normalize' in block else False
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
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

        elif block['type'] == 'upsample':
            upsample = nn.Upsample(scale_factor=int(block['stride']), mode='bilinear')
            modules.add_module('upsample_%d' % i, upsample)

        elif block['type'] == 'route':
            layers = [int(x) for x in block["layers"].split(',')]
            filters = output_filters[layers[0]]
            # If skip-connection add filters of second output
            if len(layers) > 1:
                filters += output_filters[layers[1]]
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
            detection = DetectionLayer(anchors)
            modules.add_module('Detection_%d' % i, detection)
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

    def load_weights(self, weights_path):
        """Parses and copies the weights stored in 'weights_path' to the model"""

        #Open the weights file
        fp = open(weights_path, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)   # First five are header values
        weights = np.fromfile(fp, dtype=np.float32)         # The rest are weights

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.blocks[1:], self.module_list)):
            if module_def['type'] == 'convolutional':

                conv_layer = module[0]
                if 'batch_normalize' in module_def:
                    bn_layer = module[1]
                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn_layer.bias.numel()
                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn_layer.bias.data)
                    bn_weights = bn_weights.view_as(bn_layer.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn_layer.running_mean)
                    bn_running_var = bn_running_var.view_as(bn_layer.running_var)
                    # Copy the data to model
                    bn_layer.bias.data.copy_(bn_biases)
                    bn_layer.weight.data.copy_(bn_weights)
                    bn_layer.running_mean.copy_(bn_running_mean)
                    bn_layer.running_var.copy_(bn_running_var)
                else:
                    # Number of biases
                    num_biases = conv_layer.bias.numel()
                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv_layer.bias.data)
                    # Finally copy the data
                    conv_layer.bias.data.copy_(conv_biases)

                # Let us load the weights for the Convolutional layers
                num_weights = conv_layer.weight.numel()
                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                # Copy data to model
                conv_weights = conv_weights.view_as(conv_layer.weight.data)
                conv_layer.weight.data.copy_(conv_weights)

    def forward(self, x, cuda=True):
        detections = None
        outputs = []
        for i, (module_def, module) in enumerate(zip(self.blocks[1:], self.module_list)):
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
                anchors = module[0].anchors
                #Get the number of classes
                num_classes = int(module_def["classes"])
                # Rescale detection to image dim
                x = rescale_detections(x, self.img_size, anchors, num_classes, cuda)
                # Register detection
                detections = x if detections is None else torch.cat((detections, x), 1)
            # Register output
            outputs.append(x)

        return detections
