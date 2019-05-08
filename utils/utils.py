from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
# used to compute IOU for OBB
import shapely.geometry
import shapely.affinity

import matplotlib.pyplot as plt
from descartes import PolygonPatch
import matplotlib.patches as patches

# Class for handling OBB calculations 
class OBB: 
    def __init__(self, cx, cy, w, le, theta):
        self.cx = cx
        self.cy = -cy  # minus because y is defined downside
        self.w = w
        self.le = le
        self.theta = theta

    def get_contour(self):
        #box(minx, miny, maxx, maxy)
        c = shapely.geometry.box(-self.w/2, -self.le/2, self.w/2, self.le/2)
        rc = shapely.affinity.rotate(c, self.theta.copy())
        obb = shapely.affinity.translate(rc, self.cx.copy(), self.cy.copy()) 
        return obb
    
    def intersection(self, other):
        return self.get_contour().intersection(other.get_contour())

    def union(self, other):
        return self.get_contour().union(other.get_contour())

    def iou(self, other, visualize=False):
        intersect_area = self.intersection(other).area
        union_area = self.union(other).area

        if visualize:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlim(-30, 30)
            ax.set_ylim(-30, 30)
            ax.add_patch(PolygonPatch(self.get_contour(), fc='#990000', alpha=0.7))
            ax.add_patch(PolygonPatch(other.get_contour(), fc='#000099', alpha=0.7))
            ax.add_patch(PolygonPatch(self.intersection(other), fc='#009900', alpha=1))
            # plt.show()
            plt.show(block=False)

        return intersect_area / (union_area + 1e-16)




def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
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
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ 
    Compute true positives, predicted scores and predicted labels per sample 
    output (N)(x,y, w, le, theta, score, pred)
    targets (N)(label, x,y, w, le, theta)
    """

    batch_metrics = []
    for sample_i in range(len(outputs)):
        annotations = to_cpu(targets[sample_i][targets[sample_i][:, -1] > 0]).numpy() # clean the zeros palceholders from dataset.py
        target_labels = annotations[:, 0] if len(annotations) else []

        if outputs[sample_i] is None:
            continue

        output = to_cpu(outputs[sample_i]).numpy()
        pred_boxes = output[:, :5]
        pred_scores = output[:, 5]
        pred_labels = output[:, -1]
        # this is done by finding the IOU of each prediction with all the tagets 
        # and the biggest IOU is assigned to the prediction
        # if the IOU is bigger than the threshould then it's considered to be TP sampel
        true_positives = np.zeros(pred_boxes.shape[0])
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]
            # unnormalize target output
            target_boxes[:, :2] *= 416
            target_boxes[:, 2:4] *= 416 * np.sqrt(2)
            target_boxes[:, 4] *=90

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                ious = obbox_iou(np.expand_dims(pred_box, 0), target_boxes).unsqueeze(0).numpy() # TODO if used convert it to IOU_OBB
                iou, box_index = ious.max(1), ious.argmax(1)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def obbox_iou(box1, box2, visualize=False): # box format is: x,y,w,l,theta(degrees): 
    """
    Returns the IoU of two orianted bounding boxes
    """
    ious = torch.empty(box2.shape[0])
    obb1 = OBB(box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3], box1[:, 4])
    for i in range(box2.shape[0]):
        obb2 = OBB(box2[i, 0], box2[i, 1], box2[i, 2], box2[i, 3], box2[i, 4])
        ious[i] = obb1.iou(obb2, visualize=visualize)
    return ious


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
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def bbox_iou_numpy(box1, box2):
    """Computes IoU between bounding boxes.
    Parameters
    ----------
    box1 : ndarray
        (N, 4) shaped array with bboxes
    box2 : ndarray
        (M, 4) shaped array with bboxes
    Returns
    -------
    : ndarray
        (N, M) shaped array with IoUs
    """
    area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iw = np.minimum(np.expand_dims(box1[:, 2], axis=1), box2[:, 2]) - np.maximum(
        np.expand_dims(box1[:, 0], 1), box2[:, 0]
    )
    ih = np.minimum(np.expand_dims(box1[:, 3], axis=1), box2[:, 3]) - np.maximum(
        np.expand_dims(box1[:, 1], 1), box2[:, 1]
    )
    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)
    ua = np.expand_dims((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]), axis=1) + area - iw * ih
    ua = np.maximum(ua, np.finfo(float).eps)
    intersection = iw * ih

    return intersection / ua


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    prediction shape is: 16 (batch size) X 10647 (52X52+26X26+13X13 feature maps outputs from YOLOv3 X 3 anchors)
                                         X 26 (x,y,w,l,theta,objectiveness, 20 classes)
    
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred) 
        XXX: modified to ( x,y,w,le,theta, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    # prediction[..., :4] = xywh2xyxy(prediction[..., :4]) TODO : removed as OBB accept (x,y,w,le)
    
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 5] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 5] * image_pred[:, 6:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 6:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :6], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            # obbox_iou takes OBB as (x,y,w,le,theta) # don't good
            # TODO:there is problem with computing IOU for OBB as small change in angel can lead to 
            large_overlap = bbox_iou(detections[0, :5].unsqueeze(0), detections[:, :5], x1y1x2y2=False) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 5:6] # object detection score
            # Merge overlapping bboxes by order of confidence (Weighted avrage)
            detections[0, :5] = (weights * detections[invalid, :5]).sum(0) / weights.sum() #TODO : remove theta form this ?
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def build_targets(pred_boxes, pred_cls, target, anchors, num_anchors, num_classes, grid_size, ignore_thres):


    # target size is batch_size X max_number_of_objects_in_image X 6 (class_id,x,y,w,l,theta) (normalized GT)
    # pred_boxes size is nBatch X nAnchors X featuremap X 5 (x,y,w,l,theta), after demoralizing with respect to anchors
    # pred_cls size is nBatch X nAnchors X featuremap X nClasses
    # pred_conf size is nBatch X nAnchors X featuremap X 1
    # anchors size is nAnchors X 3 (w,l,theta)

    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    nG = grid_size
    obj_mask = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = torch.ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = torch.zeros(nB, nA, nG, nG).float()
    iou_scores = torch.zeros(nB, nA, nG, nG).float()
    tx = torch.zeros(nB, nA, nG, nG)
    ty = torch.zeros(nB, nA, nG, nG)
    tw = torch.zeros(nB, nA, nG, nG)
    th = torch.zeros(nB, nA, nG, nG)
    ttheta = torch.zeros(nB, nA, nG, nG)
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)

    num_targets = 0
    num_correct = 0 # TODO: why not used
    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0:
                continue
            num_targets += 1
            # Convert to position relative to box (consider cell size is 1)
            gx = target[b, t, 1] * nG
            gy = target[b, t, 2] * nG
            gw = target[b, t, 3] * np.sqrt(2) * nG # as we normalized it by diagonal
            gh = target[b, t, 4] * np.sqrt(2) * nG
            gtheta = target[b, t, 5] * 90 # didn't depend on the cell
            # Get grid box indices
            gi = int(gx)
            gj = int(gy)
            # Get the shape of the gt box (centered at (100, 100))
            gt_shape = torch.FloatTensor([100, 100, gw, gh, gtheta]).unsqueeze(0)
            # Get shape of the anchor boxes (centered at (100, 100))
            anchor_shapes = torch.ones((len(anchors), 5)).float() * 100
            anchor_shapes[:, 2:] = anchors
            # Compute iou between gt and anchor shapes
            # TODO : better to use Orinted version or not to find the suitabl anchor box
            anch_ious = bbox_iou(gt_shape, anchor_shapes, x1y1x2y2=False)
            # Where the overlap is larger than threshold set mask to zero (ignore)
            # TODO:why ignoring anchor boxes with anch_ious > ignore_thres (only considring w, h)
            noobj_mask[b, anch_ious > ignore_thres, gj, gi] = 0
            # Find the best matching anchor box
            best_n = torch.argmax(anch_ious)
            # Get ground truth box
            gt_box = torch.FloatTensor([gx, gy, gw, gh, gtheta]).unsqueeze(0)
            # Get the prediction at best matching anchor box
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            # Masks
            obj_mask[b, best_n, gj, gi] = 1
            noobj_mask[b, best_n, gj, gi] = 0
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
            ttheta[b, best_n, gj, gi] = gtheta / 90
            # One-hot encoding of label
            target_label = int(target[b, t, 0])
            tcls[b, best_n, gj, gi, target_label] = 1
            tconf[b, best_n, gj, gi] = 1
            # Calculate iou between ground truth and best matching prediction
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            class_mask[b, best_n, gj, gi] = pred_label == target_label # id the target is the same class as prediction
            iou_scores[b, best_n, gj, gi] = obbox_iou(gt_box.numpy(), pred_box.numpy(), visualize=False)

    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, ttheta, tconf, tcls
