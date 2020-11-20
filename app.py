from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image
import json
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import boto3
import sys
import json
import time
from flask import *
import cv2
import numpy as np
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, default="/tmp/images/", help="path to dataset")
parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
opt = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#os.makedirs("output", exist_ok=True)

# Set up model
model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

if opt.weights_path.endswith(".weights"):
    # Load darknet weights
    model.load_darknet_weights(opt.weights_path)
else:
    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.weights_path))

model.eval()  # Set in evaluation mode

classes = load_classes(opt.class_path)  # Extracts class labels from file

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

app = Flask(__name__)

@app.route('/', methods=['POST'])
def main(_argv=None):
    requests = request.json

    shutil.rmtree("/tmp/images", ignore_errors=True)
    os.makedirs("/tmp/images/", exist_ok=True)
    # s3ストレージからダウンロード
    s3 = boto3.client('s3')
    for v in requests.values():
      result_bucket = v['upload_bucketname']
      s3.download_file(v['download_bucketname'],
                       v['download_imagepath'],
                       '/tmp/images/{}'.format(v['download_bucketimage']))
    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )
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

    data = {'results': []}
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        print("(%d) Image: '%s'" % (img_i, path))
        img = np.array(Image.open(path))

        # Create plot

        image_dict = {
            "Images": path,
            "Conf": []
        }

        data['results'].append(image_dict)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:


                print("\t Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                
                box_w = x2 - x1
                box_h = y2 - y1
                # bboxは高さ、幅、左、上のピクセルを表示しています
                conf_dict = {
                    "Name": classes[int(cls_pred)],
                    "Confidence": cls_conf.item(),
                    "Instances": [
                    {
                        "BoundingBox": {
                            "Width": int(box_w),
                            "Height": int(box_h),
                            "Left": int(x1),
                            "Top": int(y2)
                        },
                        "Confidence": cls_conf.item()
                    }   
                ],
                        "Parents": [
                            {
                                "Name": classes[int(cls_pred)]
                            }
                    ]
                }
                data['results'][img_i]['Conf'].append(conf_dict)
                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
    json_file = json.dumps(data)
    os.makedirs('./results', exist_ok=True)
    with open(os.path.join('results', 'output.json'), 'w') as f:
        f.write(json_file)
    # 指定したS3にアップロード
    s3.upload_file(
        'results/output.json',
        result_bucket,
        requests['image1']['upload_bucketfile']
        )
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
