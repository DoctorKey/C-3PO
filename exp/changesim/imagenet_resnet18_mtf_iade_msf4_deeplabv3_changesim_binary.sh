#!/bin/bash

train="python3 -m torch.distributed.launch --nproc_per_node=4 --use_env src/train.py --train-dataset ChangeSim_Binary --test-dataset ChangeSim_Binary --input-size 256 --model resnet18_mtf_msf_deeplabv3 --mtf iade --msf 4 --warmup --loss ce --loss-weight"

test="python3 src/train.py --test-only --save-imgs --model resnet18_mtf_msf_deeplabv3 --mtf iade --msf 4  --train-dataset ChangeSim_Binary --test-dataset ChangeSim_Binary --input-size 256"


