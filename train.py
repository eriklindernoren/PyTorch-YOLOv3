from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *
from utils.parse_config import *
from utils.loss import compute_loss
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--verbose", "-v", default=False, action='store_true', help="Makes the training more verbose")
    parser.add_argument("--logdir", type=str, default="logs", help="Defines the directory where the training log files are stored")
    opt = parser.parse_args()
    print(opt)

    logger = Logger(opt.logdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, multiscale=opt.multiscale_training, img_size=opt.img_size, transform=AUGMENTATION_TRANSFORMS)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size= model.hyperparams['batch'] // model.hyperparams['subdivisions'],
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    if (model.hyperparams['optimizer'] in [None, "adam"]):
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
            )
    elif (model.hyperparams['optimizer'] == "sgd"):
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
            momentum=model.hyperparams['momentum'])
    else:
        print("Unknown optimizer. Please choose between (adam, sgd).")

    for epoch in range(opt.epochs):
        print("\n---- Training Model ----")
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)

            outputs = model(imgs)

            loss, loss_components = compute_loss(outputs, targets, model)

            loss.backward()

            ###############
            # Run optimizer
            ###############

            if batches_done % model.hyperparams['subdivisions'] == 0:
                # Adapt learning rate
                # Get learning rate defined in cfg
                lr = model.hyperparams['learning_rate']
                if batches_done < model.hyperparams['burn_in']:
                    # Burn in
                    lr *= (batches_done / model.hyperparams['burn_in'])
                else:
                    # Set and parse the learning rate to the steps defined in the cfg
                    for threshold, value in model.hyperparams['lr_steps']:
                        if batches_done > threshold:
                            lr *= value
                # Log the learning rate
                logger.scalar_summary("train/learning_rate", lr, batches_done)
                # Set learning rate
                for g in optimizer.param_groups:
                        g['lr'] = lr

                # Run optimizer
                optimizer.step()
                # Reset gradients
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------
            log_str = ""
            log_str += AsciiTable(
                [
                    ["Type", "Value"],
                    ["IoU loss", float(loss_components[0])],
                    ["Object loss", float(loss_components[1])], 
                    ["Class loss", float(loss_components[2])],
                    ["Loss", float(loss_components[3])],
                    ["Batch loss", to_cpu(loss).item()],
                ]).table

            if opt.verbose: print(log_str)

            # Tensorboard logging
            tensorboard_log = [
                    ("train/iou_loss", float(loss_components[0])),
                    ("train/obj_loss", float(loss_components[1])), 
                    ("train/class_loss", float(loss_components[2])),
                    ("train/loss", to_cpu(loss).item())]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            metrics_output = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.1,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=model.hyperparams['batch'] // model.hyperparams['subdivisions'],
            )
            
            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                evaluation_metrics = [
                ("validation/precision", precision.mean()),
                ("validation/recall", recall.mean()),
                ("validation/mAP", AP.mean()),
                ("validation/f1", f1.mean()),
                ]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)

                if opt.verbose:
                    # Print class APs and mAP
                    ap_table = [["Index", "Class name", "AP"]]
                    for i, c in enumerate(ap_class):
                        ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
                    print(AsciiTable(ap_table).table)
                    print(f"---- mAP {AP.mean()}")                
            else:
                print( "---- mAP not measured (no detections found by model)")

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
