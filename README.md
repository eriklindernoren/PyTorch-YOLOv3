# PyTorch-YOLOv3
A minimal PyTorch implementation of YOLOv3, with support for training, inference and evaluation.

## Installation
##### Clone and install requirements
    $ git clone git@github.com:fusic/PyTorch-YOLOv3.git
    $ cd PyTorch-YOLOv3/
    $ sudo pip3 install -r requirements.txt

##### Download pretrained weights
    $ cd weights/
    $ wget -c https://pjreddie.com/media/files/yolov3.weights

##### Download COCO
    $ cd data/
    $ git clone https://github.com/pdollar/coco
    $ cd coco
    $ wget -c https://pjreddie.com/media/files/instances_train-val2014.zip
    $ wget -c https://pjreddie.com/media/files/coco/5k.part
    $ wget -c https://pjreddie.com/media/files/coco/trainvalno5k.part
    $ wget -c https://pjreddie.com/media/files/coco/labels.tgz
    $ tar xzf labels.tgz
    $ unzip -q instances_train-val2014.zip

# Set Up Image Lists
    $ paste <(awk "{print \"$PWD\"}" <5k.part) 5k.part | tr -d '\t' > 5k.txt
    $ paste <(awk "{print \"$PWD\"}" <trainvalno5k.part) trainvalno5k.part | tr -d '\t' > trainvalno5k.txt    

## API start
    $ python3.6 app.py
