#!/bin/bash

train="python3 -m torch.distributed.launch --nproc_per_node=4 --use_env src/train.py --train-dataset COCO --test-dataset COCO --model resnet18_msf_deeplabv3 --warmup --loss ce --epochs 30"

test="python3 src/train.py --test-only --model resnet18_msf_deeplabv3 --train-dataset COCO --test-dataset COCO"


