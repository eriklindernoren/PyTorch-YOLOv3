from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def filter_detections(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (image_index, object_confidence, x1, y1, x2, y2, class_score, class_pred)
    """
    # If no detection with confidence over threshold => exit
    if (prediction[:, :, 4] >= conf_thres).squeeze().sum() == 0:
        return

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = None
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1,  keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        #Get the various classes detected in the image
        unique_classes = torch.from_numpy(np.unique(detections[:, -1].data.cpu().numpy())).cuda()
        for c in unique_classes:
            #get the detections with one particular class
            cls_mask = (detections[:, -1] == c)
            detections_class = detections[cls_mask]
            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            n_cls_detections = detections_class.size(0)   # Number of detections

            for i in range(n_cls_detections):
                # If we're at the last detection
                if i >= len(detections_class) - 1:
                    break
                # Get the IOUs of all boxes of lower confidence than box i
                ious = bbox_iou(detections_class[i].unsqueeze(0), detections_class[i+1:])
                # Remove detections with IoU > NMS threshold
                iou_mask = (ious < nms_thres)
                detections_class = torch.cat((detections_class[:i+1], detections_class[i+1:][iou_mask]))
            # Get index of image
            image_index = detections_class.new(detections_class.size(0), 1).fill_(image_i)
            # Repeat the batch_id for as many detections of the class cls in the image
            detections_class = torch.cat((image_index, detections_class), 1)
            # Add detection to outputs
            output = detections_class if output is None else torch.cat((output, detections_class), 0)

    return output
