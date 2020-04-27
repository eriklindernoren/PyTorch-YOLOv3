
from torch.utils.data import DataLoader, Dataset
import torch
from zipfile import ZipFile
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
import io
from PIL import Image


def show_box2d(sample):
    image = sample['image']
    labels = sample['labels']

    for child in labels['children']:
        start_point = (child['x0'], child['y0'])
        end_point = (child['x1'], child['y1'])
        color = (255, 0, 0)
        thickness = 3
        image = cv2.rectangle(image, start_point, end_point, color, thickness)

    plt.imshow(image)
    plt.pause(0.001)


class ToTarget(object):
    """ Convert raw data to target format to train network.
    Network specific implementation might be needed. """

    def __call__(self, sample):

        # load raw data
        image, labels = sample['image'], sample['labels']

        """ process input """
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        _, h, w = image.shape

        """ process labels """
        targets = torch.zeros((1, 1))
        # rescale box2d, do something else...

        return {'image': torch.from_numpy(image), 'targets': targets}


class EcpDataset(Dataset):
    """ ECP Dataset Class - inherits from std PyTorch Dataset class
    inspired by
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    """

    def __init__(self, root_dir_images, root_dir_labels, transform=None):

        # check given input
        assert os.path.isfile(path_images), 'Does not exist / not a file: %s' % path_images
        assert os.path.isfile(path_labels), 'Does not exist / not a file: %s' % path_labels
        # assert path_zip_images[-4:] == '.zip', 'File name extension does not indicate a file: %s' % path_zip_images[-4:]
        # assert path_zip_labels[-4:] == '.zip', 'File name extension does not indicate a file: %s' % path_zip_labels[-4:]
        self.root_dir_images = root_dir_images
        self.root_dir_labels = root_dir_labels

        # set transfrom
        self.transform = transform

        # read all png images from zip file (do not extract yet)
        # self.image_names = []
        # with ZipFile(self.path_zip_images, 'r') as zipFile:
        #     for name in zipFile.namelist():
        #         if name[-4:] == '.png':
        #             self.image_names.append(name)
        #         else:
        #             continue
        # sort image names alphabetically to assert same order as labels
        # self.image_names.sort()
        # print('%d images added to data loader.' % len(self.image_names))

        # read all json label files from zip file (do not extract yet)
        # self.label_names = []
        # with ZipFile(self.path_zip_labels, 'r') as zipFile:
        #     for name in zipFile.namelist():
        #         if name[-5:] == '.json':
        #             self.label_names.append(name)
        #         else:
        #             continue
        # sort label names alphabetically to assert same order as images
        # self.label_names.sort()
        # print('%d label files added to data loader.' % len(self.label_names))

        # assert same number of images and label files (one for each)
        # assert len(self.image_names) == len(self.label_names)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        # convert to list if idx is tensor
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir_images)
        # # extract and read image file
        # with ZipFile(self.path_zip_images, 'r') as zipFile:
        #     image = zipFile.read(self.image_names[idx])
        #     image = Image.open(io.BytesIO(image))
        #     image = np.array(image)

        # # extract and read label file
        # with ZipFile(self.path_zip_labels, 'r') as zipFile:
        #     labels = zipFile.read(self.label_names[idx])
        #     labels = labels.decode('utf-8')
        #     labels = json.loads(labels, 'r')

        # compose data sample as dict
        sample = {'image': image, 'labels': labels}

        # apply transform if given
        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":

    """ example use """

    # set paths to zip files
    path_zip_images = 'datasets/ECP_night_img_val.zip'
    path_zip_labels = 'datasets/ECP_night_labels_val.zip'

    # get dataset object
    ecpDataset = EcpDataset(path_zip_images, path_zip_labels)

    # plot sample image
    id = 123
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.tight_layout()
    ax.set_title('Sample %03d' % id)
    ax.axis('off')
    show_box2d(ecpDataset[id])
    plt.show()

    # apply transformations
    ecpDataset_transformed = EcpDataset(path_zip_images, path_zip_labels, transform=ToTarget())

    # create pytorch data loader object which can be used for training
    dataLoader = DataLoader(ecpDataset_transformed, batch_size=4, shuffle=False, num_workers=1)

    # load some batches
    for i_batch, sample_batched in enumerate(dataLoader):
        print('Loading batch %03d...' % i_batch)
        if i_batch > 5: break
