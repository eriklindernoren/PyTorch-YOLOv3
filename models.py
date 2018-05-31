from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from PIL import Image

from utils.parse_config import *
from utils.utils import build_targets

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        padding=pad,
                                                        bias=not bn))
            if bn:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

        elif module_def['type'] == 'upsample':
            upsample = nn.Upsample( scale_factor=int(module_def['stride']),
                                    mode='nearest')
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def["layers"].split(',')]
            filters = sum([output_filters[layer_i] for layer_i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module("shortcut_%d" % i, EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def['classes'])
            img_height = int(hyperparams['height'])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_height)
            modules.add_module('yolo_%d' % i, yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list

class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""
    def __init__(self):
        super(EmptyLayer, self).__init__()

class YOLOLayer(nn.Module):
    """Detection layer"""
    def __init__(self, anchors, num_classes, image_dim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.scaled_anchors = None
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_dim = image_dim
        self.ignore_thres = 0.5
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.seen = 0

        self.mse_loss = nn.MSELoss(size_average=False)
        self.ce_loss = nn.CrossEntropyLoss(size_average=False)

    def forward(self, x, targets=None):
        bs = x.size(0)
        g_dim = x.size(2)
        stride =  self.image_dim / g_dim
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        if x.is_cuda:
            self.mse_loss = self.mse_loss.cuda()
            self.ce_loss = self.ce_loss.cuda()

        prediction = x.view(bs, self.bbox_attrs * self.num_anchors, g_dim * g_dim)
        prediction = prediction.transpose(1, 2).contiguous()
        prediction = prediction.view(bs, g_dim * g_dim * self.num_anchors, self.bbox_attrs)

        # Get outputs
        x = torch.sigmoid(prediction[:, :, 0])          # Center x
        y = torch.sigmoid(prediction[:, :, 1])          # Center y
        w = prediction[:, :, 2]                         # Width
        h = prediction[:, :, 3]                         # Height
        conf = torch.sigmoid(prediction[:, :, 4])       # Conf
        pred_cls = torch.sigmoid(prediction[:, :, 5:])  # Cls pred.

        # Get x and y offsets for each grid
        grid = np.arange(g_dim)
        a, b = np.meshgrid(grid, grid)
        x_offset = FloatTensor(a).view(-1, 1)
        y_offset = FloatTensor(b).view(-1, 1)
        x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, self.num_anchors).view(-1, 2).unsqueeze(0)

        # Scale anchors
        scaled_anchors = [(a_w / stride, a_h / stride) for a_w, a_h in self.anchors]
        anchors = FloatTensor(scaled_anchors).repeat(g_dim * g_dim, 1).unsqueeze(0)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[:, :, :4].shape)
        pred_boxes[:, :, 0] = x.data + x_y_offset[:, :, 0]
        pred_boxes[:, :, 1] = y.data + x_y_offset[:, :, 1]
        pred_boxes[:, :, 2] = torch.exp(w.data) * anchors[:, :, 0]
        pred_boxes[:, :, 3] = torch.exp(h.data) * anchors[:, :, 1]

        self.seen += prediction.size(0)

        # Training
        if targets is not None:

            nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls = build_targets(pred_boxes[:, :, :4].cpu().data,
                                                                                                        targets.cpu().data,
                                                                                                        scaled_anchors,
                                                                                                        self.num_anchors,
                                                                                                        self.num_classes,
                                                                                                        g_dim,
                                                                                                        g_dim,
                                                                                                        self.noobject_scale,
                                                                                                        self.object_scale,
                                                                                                        self.ignore_thres,
                                                                                                        self.image_dim,
                                                                                                        self.seen)


            nProposals = int((conf > 0.25).sum().item())

            tx    = Variable(tx.type(FloatTensor), requires_grad=False)
            ty    = Variable(ty.type(FloatTensor), requires_grad=False)
            tw    = Variable(tw.type(FloatTensor), requires_grad=False)
            th    = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls  = Variable(tcls[cls_mask == 1].type(LongTensor), requires_grad=False)
            coord_mask = Variable(coord_mask.type(FloatTensor), requires_grad=False)
            conf_mask  = Variable(conf_mask.type(FloatTensor), requires_grad=False)

            pred_cls = pred_cls[cls_mask.view(bs, -1) == 1]

            loss_x = self.coord_scale * self.mse_loss(x.view_as(tx)*coord_mask, tx*coord_mask) / 2
            loss_y = self.coord_scale * self.mse_loss(y.view_as(ty)*coord_mask, ty*coord_mask) / 2
            loss_w = self.coord_scale * self.mse_loss(w.view_as(tw)*coord_mask, tw*coord_mask) / 2
            loss_h = self.coord_scale * self.mse_loss(h.view_as(th)*coord_mask, th*coord_mask) / 2
            loss_conf = self.mse_loss(conf.view_as(tconf)*conf_mask, tconf*conf_mask)
            loss_cls = self.class_scale * self.ce_loss(pred_cls, tcls)
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            print('%d: nGT %d, recall %d, AP %.2f%% proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (self.seen, nGT, nCorrect, 100*float(nCorrect/nGT), nProposals, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_cls.item(), loss.item()))

            return loss

        else:
            # If not in training phase return predictions
            output = torch.cat((pred_boxes * stride, conf.unsqueeze(-1), pred_cls), -1)
            return output


class Darknet(nn.Module):
    """YOLOv3 object detection model"""
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size

    def forward(self, x, targets=None):
        output = [] if targets is None else 0   # Model outputs
        layer_outputs = []                      # Layer outputs
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample']:
                x = module(x)
            elif module_def['type'] == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'yolo':
                x = module[0](x, targets)
                # If predictions: concatenate / if loss: add to total loss
                output = output + [x] if targets is None else output + x
            layer_outputs.append(x)

        return torch.cat(output, 1) if targets is None else output


    def load_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        #Open the weights file
        fp = open(weights_path, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)   # First five are header values

        # Needed to write header when saving weights
        self.header_info = header

        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)         # The rest are weights
        fp.close()

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                if module_def['batch_normalize']:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel() # Number of biases
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
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    """
        @:param path    - path of the new weights file
        @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
    """
    def save_weights(self, path, cutoff=-1):

        # Load only cutoff layers, if cutoff != -1
        if cutoff:
            num_layers = len(self.module_list)
        else:
            num_layers = cutoff

        fp = open(path, 'wb')
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]

                # If batch norm, load bn first
                if module_def['batch_normalize']:
                    bn_layer = module[1]

                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)

                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()