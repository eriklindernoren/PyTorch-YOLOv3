from __future__ import division

from models import *
from utils.logger import *
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
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, help="if specified starts from checkpoint model")
parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
opt = parser.parse_args()
print(opt)

logger = Logger("logs")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
train_path = data_config["train"]

# Get hyper parameters
hyperparams = parse_model_config(opt.model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# Initiate model
model = Darknet(opt.model_config_path).to(device)
model.apply(weights_init_normal)

# If specified we start from checkpoint
if opt.weights_path:
    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)
    else:
        model.load_state_dict(torch.load(opt.weights_path))

model.train()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu
)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

for epoch in range(opt.epochs):
    start_time = time.time()
    for batch_i, (_, imgs, targets) in enumerate(dataloader):

        batches_done = len(dataloader) * epoch + batch_i

        imgs = Variable(imgs.to(device))
        targets = Variable(targets.to(device), requires_grad=False)

        optimizer.zero_grad()

        loss = model(imgs, targets)

        loss.backward()
        optimizer.step()

        # ----------------
        #   Log progress
        # ----------------

        print("\n---- [Epoch %d/%d, Batch %d/%d] ----" % (epoch, opt.epochs, batch_i, len(dataloader)))
        for i in range(4):
            print(
                "[%s] [loss %f, x %f, y %f, w %f, h %f, conf %f, cls %f, cls_acc: %.2f%%, recall: %.5f, precision: %.5f]"
                % (
                    "Total" if i + 1 == 4 else "YOLO Layer %d" % (i + 1),
                    model.losses[i]["loss"],
                    model.losses[i]["x"],
                    model.losses[i]["y"],
                    model.losses[i]["w"],
                    model.losses[i]["h"],
                    model.losses[i]["conf"],
                    model.losses[i]["cls"],
                    100 * model.losses[i]["cls_acc"],
                    model.losses[i]["recall"],
                    model.losses[i]["precision"],
                )
            )

            # Tensorboard logging
            for name, loss in model.losses[i].items():
                loss_name = f"{name}_total" if i + 1 == 4 else f"{name}_{i+1}"
                logger.scalar_summary(loss_name, loss, batches_done)

        # Determine approximate time left for epoch
        epoch_batches_left = len(dataloader) - (batch_i + 1)
        time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
        print(f"---- ETA {time_left}")

        model.seen += imgs.size(0)

        torch.cuda.empty_cache()

    if epoch % opt.checkpoint_interval == 0:
        torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
