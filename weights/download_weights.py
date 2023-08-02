#!/usr/bin/env python3
import os
import urllib.request


WEIGHTS_FOLDER = os.path.abspath(os.path.dirname(__file__))

DOWNLOAD_LINKS = [
    'https://pjreddie.com/media/files/yolov3.weights',
    'https://pjreddie.com/media/files/yolov3-tiny.weights',
    'https://pjreddie.com/media/files/darknet53.conv.74',
]


def download(remote_location, folder):
    filename = os.path.basename(remote_location)
    local_location = os.path.join(folder, filename)
    print(f'Downloading \'{filename}\' to \'{folder}\' ... '
          f'This might take a some of time.')

    # Special stuff to fix 404 for some people
    # https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/526
    opener = urllib.request.build_opener()
    opener.addheaders = [('Referer', 'pjreddie.com')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(remote_location, local_location)
    print(f'Download of \'{filename}\' finished!')


def download_all():
    for weight_url in DOWNLOAD_LINKS:
        download(weight_url, WEIGHTS_FOLDER)


if __name__ == '__main__':
    download_all()
