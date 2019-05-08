from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

from terminaltables import AsciiTable

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
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/road.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="weights/yolov3_WithoutAngel.pth", help="if specified starts from checkpoint model")
parser.add_argument("--class_path", type=str, default="data/All Anotated Photos/classes.txt", help="path to class label file")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model weights")
#parser.add_argument("--multi_scale", default=True, help="allow for multi-scale training")
opt = parser.parse_args()
print(opt)

logger = Logger("logs")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
train_path = data_config["train"]

# Initiate model
model = Darknet(opt.model_config_path).to(device)
model.apply(weights_init_normal)

# If specified we start from checkpoint
if opt.weights_path:
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

model.train()

# Get dataloader
dataset = ListDataset(train_path)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True
)

optimizer = torch.optim.Adam(model.parameters())

metrics = [
    "grid_size",
    "loss",
    "x",
    "y",
    "w",
    "h",
    "conf",
    "cls",
    "cls_acc",
    "recall50",
    "recall75",
    "precision",
    "conf_obj",
    "conf_noobj",
]

for epoch in range(opt.epochs):
    start_time = time.time()
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        batches_done = len(dataloader) * epoch + batch_i

        # Enables multi-scale training
        #if opt.multi_scale and batches_done % 2 == 0:
        #   min_size = opt.img_size - 3 * 32
        #   max_size = opt.img_size + 3 * 32
        #   imgs = random_resize(imgs, min_size, max_size)

        imgs = Variable(imgs.to(device))
        targets = Variable(targets.to(device), requires_grad=False)

        loss, outputs = model(imgs, targets)
        loss.backward()

        if batches_done % opt.gradient_accumulations:
            # Accumulates gradient before each step
            optimizer.step()
            optimizer.zero_grad()

        # ----------------
        #   Log progress
        # ----------------

        log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

        metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

        # Log metrics at each YOLO layer
        for i, metric in enumerate(metrics):
            formats = {m: "%.6f" for m in metrics}
            formats["grid_size"] = "%2d"
            formats["cls_acc"] = "%.2f%%"
            row_metrics = [formats[metric] % yolo.metrics[metric] for yolo in model.yolo_layers]
            metric_table += [[metric, *row_metrics]]

            # Tensorboard logging
            tensorboard_log = []
            for j, yolo in enumerate(model.yolo_layers):
                for name, metric in yolo.metrics.items():
                    if name != "grid_size":
                        tensorboard_log += [(f"{name}_{j+1}", metric)]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)

        log_str += AsciiTable(metric_table).table

        global_metrics = [("Total Loss", loss.item())]

        # Print mAP and other global metrics
        log_str += "\n" + ", ".join([f"{metric_name} {metric:f}" for metric_name, metric in global_metrics])

        # Determine approximate time left for epoch
        epoch_batches_left = len(dataloader) - (batch_i + 1)
        time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
        log_str += f"\n---- ETA {time_left}"

        print(log_str)

        model.seen += imgs.size(0)

    if epoch % opt.checkpoint_interval == 0:
        torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
