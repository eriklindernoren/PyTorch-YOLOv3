# PyTorch-YOLO-V3
Minimal implementation of YOLO-V3

## Table of Contents
- [PyTorch-YOLO-V3](#pytorch-yolo-v3)
  * [Table of Contents](#table-of-contents)
  * [Installation](#installation)
  * [Run Inference](#run-inference)
  * [Credit](#credit)

## Installation
    $ git clone https://github.com/eriklindernoren/PyTorch-YOLO-V3
    $ cd PyTorch-YOLO-V3/
    $ sudo pip3 install -r requirements.txt
    $ cd weights/
    $ bash download_weights.sh
    
## Run Inference
    $ python3 test.py --image_folder /data/samples
   
<p align="center"><img src="outputs/2_0.png" width="480"\></p>
<p align="center"><img src="outputs/3_0.png" width="480"\></p>
<p align="center"><img src="outputs/6_0.png" width="480"\></p>

## Credit
Inspired by https://github.com/ayooshkathuria/pytorch-yolo-v3
