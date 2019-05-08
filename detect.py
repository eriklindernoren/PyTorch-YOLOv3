from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
parser.add_argument("--config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
parser.add_argument("--weights_path", type=str, default="weights/yolov3_WithAngel.pth", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/All Anotated Photos/classes.txt", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available()

os.makedirs("output", exist_ok=True)

# Set up model
model = Darknet(opt.config_path, img_size=opt.img_size)

if opt.weights_path.endswith("WithoutAngel.pth"):
        
    model_dict = model.state_dict() # state of the current model 
    pretrained_dict = torch.load(opt.weights_path) # state of the pretrained model
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if ('81' not in k) and ('93' not in k) and ('105' not in k)} # remove the classifier from the state
    classifier_dict = {k: v for k, v in model_dict.items() if ('81' in k) or ('93' in k) or ('105' in k)} # get the classifier weight from new model
    pretrained_dict.update(classifier_dict)
    model_dict.update(pretrained_dict) # update without classifier 
    model.load_state_dict(pretrained_dict) # the model know has the wights of the model without angel but the classifier part is intialized
    
else:
    model.load_state_dict(torch.load(opt.weights_path))


if cuda:
    model.cuda()

model.eval()  # Set in evaluation mode

dataloader = DataLoader(
    ListDataset(opt.image_folder, img_size=opt.img_size),
    batch_size=opt.batch_size,
    shuffle=False,
)

classes = load_classes(opt.class_path)  # Extracts class labels from file

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

imgs = []  # Stores image paths
img_detections = []  # Stores detections for each image index

print("\nPerforming object detection:")
prev_time = time.time()
for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))

    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

    # Save image and detections
    imgs.extend(img_paths)
    img_detections.extend(detections)

# Bounding-box colors
cmap = plt.get_cmap("tab20b")
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

print("\nSaving images:")
# Iterate through images and save plot of detections
for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

    print("(%d) Image: '%s'" % (img_i, path))

    # Create plot
    img = np.array(Image.open(path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    if detections is not None:
        # get the vertcies of the OBB
        p1_x = detections[:, 0] + detections[:, 3] * np.cos(np.radians(detections[:, 4])) / 2.0 + \
            detections[:, 2] * np.cos(np.radians(90 + detections[:, 4])) / 2.0
        p1_y = detections[:, 1] - detections[:, 3] * np.sin(np.radians(detections[:, 4])) / 2.0 - \
            detections[:, 2] * np.sin(np.radians(90 + detections[:, 4])) / 2.0

        p2_x = detections[:, 0] - detections[:, 3] * np.cos(np.radians(detections[:, 4])) / 2.0 + \
            detections[:, 2] * np.cos(np.radians(90 + detections[:, 4])) / 2.0
        p2_y = detections[:, 1] + detections[:, 3] * np.sin(np.radians(detections[:, 4])) / 2.0 - \
            detections[:, 2] * np.sin(np.radians(90 + detections[:, 4])) / 2.0

        p3_x = detections[:, 0] - detections[:, 3] * np.cos(np.radians(detections[:, 4])) / 2.0 - \
            detections[:, 2] * np.cos(np.radians(90 + detections[:, 4])) / 2.0
        p3_y = detections[:, 1] + detections[:, 3] * np.sin(np.radians(detections[:, 4])) / 2.0 + \
            detections[:, 2] * np.sin(np.radians(90 + detections[:, 4])) / 2.0

        p4_x = detections[:, 0] + detections[:, 3] * np.cos(np.radians(detections[:, 4])) / 2.0 - \
            detections[:, 2] * np.cos(np.radians(90 + detections[:, 4])) / 2.0
        p4_y = detections[:, 1] - detections[:, 3] * np.sin(np.radians(detections[:, 4])) / 2.0 + \
            detections[:, 2] * np.sin(np.radians(90 + detections[:, 4])) / 2.0
        
        # The amount of padding that was added
        pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
        # Image height and width after padding is removed
        unpad_h = opt.img_size - pad_y
        unpad_w = opt.img_size - pad_x

        # Rescale coordinates to original dimensions
        p1_x = ((p1_x - pad_x // 2) / unpad_w) * img.shape[1]
        p2_x = ((p2_x - pad_x // 2) / unpad_w) * img.shape[1]
        p3_x = ((p3_x - pad_x // 2) / unpad_w) * img.shape[1]
        p4_x = ((p4_x - pad_x // 2) / unpad_w) * img.shape[1]
        p1_y = ((p1_y - pad_y // 2) / unpad_h) * img.shape[0]
        p2_y = ((p2_y - pad_y // 2) / unpad_h) * img.shape[0]
        p3_y = ((p3_y - pad_y // 2) / unpad_h) * img.shape[0]
        p4_y = ((p4_y - pad_y // 2) / unpad_h) * img.shape[0]

        # Draw bounding boxes and labels of detections
    
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for i, (x, y, w, le, theta, conf, cls_conf, cls_pred) in enumerate(detections):

            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

            # Rescale coordinates to original dimensions
            y = ((y - pad_y // 2) / unpad_h) * img.shape[0]
            x = ((x - pad_x // 2) / unpad_w) * img.shape[1]

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a polygon patch
            verts = [(p1_x[i], p1_y[i]), (p2_x[i], p2_y[i]), (p3_x[i], p3_y[i]), (p4_x[i], p4_y[i]), (0., 0.), ]
            codes = [Path.MOVETO,        Path.LINETO,        Path.LINETO,        Path.LINETO,        Path.CLOSEPOLY, ]
            path = Path(verts, codes)
            obbox = patches.PathPatch(path, linewidth=1, edgecolor=color, facecolor='none')
            # Add the bbox to the plot
            ax.add_patch(obbox)
            # Add label
            plt.text(
                x,
                y,
                s=classes[int(cls_pred)],
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )

    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig("output/%d.png" % (img_i), bbox_inches="tight", pad_inches=0.0)
    plt.close()
