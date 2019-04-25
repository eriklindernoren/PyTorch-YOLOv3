import torch
import torch.nn.functional as F
import numpy as np

# from datasets import ListDataset
# from parse_config import parse_data_config
# from utils import xywh2xyxy
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 1] = 1 - targets[:, 1]
    return images, targets


# if __name__ == "__main__":
#     # Get data configuration
#     data_config = parse_data_config("config/coco.data")
#     train_path = data_config["train"]
#     dataset = ListDataset(train_path)
#
#     _, image, target = dataset.__getitem__(1)
#
#     image, target = horisontal_flip(image, target)
#
#     target = target[target[..., -1] > 0]
#
#     target[:, 1:] = xywh2xyxy(target[:, 1:]) * 416
#
#     plt.figure(figsize=(20, 20))
#     plt.imshow(image.permute(1, 2, 0))
#     for x1, y1, x2, y2 in target[:, 1:]:
#         box_w, box_h = x2 - x1, y2 - y1
#         # Create a Rectangle patch
#         bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor="w", facecolor="none")
#         # Add the bbox to the plot
#         ax = plt.gca()
#         ax.add_patch(bbox)
#         print(image.shape, target.shape)
#     plt.show()
#     plt.close()
