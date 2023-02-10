#!/bin/bash

train="python3 -m torch.distributed.launch --nproc_per_node=4 --use_env src/train.py --train-dataset VL_CMU_CD  --test-dataset VL_CMU_CD --input-size 512 --model resnet18_msf_deeplabv3_mtf --mtf id --warmup --loss bi --loss-weight --pretrained output/resnet18_msf4_deeplabv3_COCO_0_448/2021-10-19_09:02:03/best.pth"

test="python3 src/train.py --test-only --save-imgs --model resnet18_msf_deeplabv3_mtf --mtf id --train-dataset PCD_CV --test-dataset TSUNAMI --data-cv 0 --input-size 256 --resume output/deeplabv3_bi_fpn4_bdae_vgg16bn_PCD_CV_0_256/2021-10-02_10:58:34/best.pth"


