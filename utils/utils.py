from __future__ import division
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area =    torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                    torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
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
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))

    return output

def build_targets(pred_boxes, target, anchors, num_anchors, num_classes, dim, ignore_thres, img_dim):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    dim = dim
    mask        = torch.zeros(nB, nA, dim, dim)
    tx         = torch.zeros(nB, nA, dim, dim)
    ty         = torch.zeros(nB, nA, dim, dim)
    tw         = torch.zeros(nB, nA, dim, dim)
    th         = torch.zeros(nB, nA, dim, dim)
    tconf      = torch.zeros(nB, nA, dim, dim)
    tcls       = torch.zeros(nB, nA, dim, dim, num_classes)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0:
                continue
            nGT = nGT + 1
            # Convert to position relative to box
            gx = target[b, t, 1] * dim
            gy = target[b, t, 2] * dim
            gw = target[b, t, 3] * dim
            gh = target[b, t, 4] * dim
            # Get grid box indices
            gi = int(gx)
            gj = int(gy)
            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # Calculate iou between gt and anchor shape
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)
            best_iou = anch_ious[best_n]
            # Get the ground truth box and corresponding best prediction
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)

            # Masks
            mask[b, best_n, gj, gi] = 1
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw/anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh/anchors[best_n][1] + 1e-16)
            # One-hot encoding of label
            tcls[b, best_n, gj, gi, int(target[b, t, 0])] = 1
            # Calculate iou between ground truth and best matching prediction
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            tconf[b, best_n, gj, gi] = 1

            if iou > 0.5:
                nCorrect = nCorrect + 1

    return nGT, nCorrect, mask, tx, ty, tw, th, tconf, tcls

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.from_numpy(np.eye(num_classes, dtype='uint8')[y])

def compare_boxes(detections, annotations, iou_threshold):
    """ Evaluate detections for a single class on an image

    # Arguments
        detections    : The constructed detections to evaluate.
        annotations   : The ground truth annotations to compare with.
        iou_threshold : The threshold used to consider when a detection is positive or negative.
    # Returns
        The binary mask of true positive detections
    """
    num_annotations = annotations.shape[0]
    num_detections  = detections.shape[0]
    true_positives  = np.zeros(num_detections, dtype=np.bool)

    if num_annotations == 0:
        # all detections are false positives
        return true_positives

    if num_detections == 0:
        return true_positives

    # binary mask of detected annotations
    detected_annotations = np.zeros(num_annotations, dtype=np.bool)

    # process detections in order of decreasing scores
    scores = detections[:, 4]
    order  = np.argsort(-scores)
    detections = detections[order]

    # construct reverse mapping to restore the original ordering
    rev_order = np.empty(num_detections, dtype=int)
    rev_order[order] = np.arange(num_detections)

    for i, d in enumerate(detections):
        overlaps            = bbox_iou(d[None, :], annotations)
        assigned_annotation = torch.argmax(overlaps)
        max_overlap         = overlaps[assigned_annotation]

        if max_overlap >= iou_threshold and not detected_annotations[assigned_annotation]:
            true_positives[i] = True
            detected_annotations[assigned_annotation] = True
    # return true positives in  ordering of the input
    return true_positives[rev_order]

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def evaluate(detections, annotations, iou_threshold=0.5):
    """ Evaluate a given dataset using a given model results.

    # Arguments
        detections      : The constructed detections to evaluate. The detections is a list of lists such that the size is: detections[num_images][num_labels] = image_detections[num_detections][4 + 1].
        annotations     : The ground truth annotations as list of lists such that the size is: annotations[num_images][num_labels] = image_annotations[num_annotations][4]
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
    # Returns
        A dict mapping class to mAP scores.
    """
    average_precisions = dict()

    num_images  = len(detections)
    assert len(annotations) == num_images,\
            "Number of images in detections and annotations does not match ({0} vs {1})".format(num_images, len(annotations))
    if num_images == 0:
        return average_precisions

    num_classes = len(detections[0])
    assert all([len(el) == num_classes for el in detections]),\
            "Number of detected classes is not constant among images"
    assert all([len(el) == len(annotations[0]) for el in annotations]),\
            "Number of annotated classes is not constant among images"
    assert len(annotations[0]) == num_classes,\
            "Number of classes in detections and annotations does not match ({0} vs {1})".format(num_classes, len(annotations[0]))

    # process detections and annotations
    for label in range(num_classes):
        true_positives  = np.hstack(list(map(
            lambda d, a: compare_boxes(d[label], a[label], iou_threshold),
            detections, annotations)))
        scores          = np.hstack([el[label][:, 4]
            if el[label].shape[0] > 0
            else []
            for el in detections])
        num_annotations = sum([el[label].shape[0] for el in annotations])
        num_detections  = true_positives.size

        # no image_annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            #average_precisions[label] = 1 if num_detections == 0 else 0
            average_precisions[label] = 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        true_positives  = np.cumsum(true_positives)
        false_positives = np.cumsum(np.logical_not(true_positives))

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precisions[label] = _compute_ap(recall, precision)

    return average_precisions
