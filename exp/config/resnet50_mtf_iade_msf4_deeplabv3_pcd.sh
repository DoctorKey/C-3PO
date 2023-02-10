#!/bin/bash

train="python3 -m torch.distributed.launch --nproc_per_node=4 --use_env src/train.py --train-dataset PCD_CV --test-dataset GSV --test-dataset2 TSUNAMI --data-cv 4 --input-size 256 --model resnet50_mtf_msf_deeplabv3 --mtf iade --msf 4 --batch-size 4 --warmup --loss bi --loss-weight"

test="python3 src/train.py --test-only --save-imgs --model resnet50_mtf_msf_deeplabv3 --mtf iade --msf 4 --train-dataset PCD_CV --test-dataset TSUNAMI --data-cv 0 --input-size 256 --resume output/deeplabv3_tri_fpn4_resnet18_PCD_CV_0_256/2021-09-23_14:48:25/checkpoint.pth"


