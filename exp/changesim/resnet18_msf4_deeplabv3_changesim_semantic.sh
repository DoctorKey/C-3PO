#!/bin/bash

train="python3 -m torch.distributed.launch --nproc_per_node=4 --use_env src/train.py --train-dataset ChangeSim_Semantic --test-dataset ChangeSim_Semantic --input-size 256 --model resnet18_msf_deeplabv3 --warmup --loss ce --epochs 100"

test="python3 src/train.py --test-only --save-imgs --model resnet18_msf_deeplabv3 --train-dataset ChangeSim_Semantic --test-dataset ChangeSim_Semantic --input-size 256"


