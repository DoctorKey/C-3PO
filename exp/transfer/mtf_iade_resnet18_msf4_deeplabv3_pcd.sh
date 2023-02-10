#!/bin/bash

train="python3 -m torch.distributed.launch --nproc_per_node=4 --use_env src/train.py --train-dataset PCD_CV  --test-dataset GSV --test-dataset2 TSUNAMI --data-cv 4 --input-size 256 --model mtf_resnet18_msf_deeplabv3 --mtf iade --warmup --loss bi --loss-weight --pretrained output/resnet18_msf4_deeplabv3_COCO_0_448/2021-10-19_09:02:03/best.pth"

test="python3 src/train.py --test-only --save-imgs --model mtf_resnet18_msf_deeplabv3 --mtf iade --train-dataset PCD_CV --test-dataset TSUNAMI --data-cv 0 --input-size 256 --resume output/deeplabv3_bi_fpn4_bdae_vgg16bn_PCD_CV_0_256/2021-10-02_10:58:34/best.pth"


