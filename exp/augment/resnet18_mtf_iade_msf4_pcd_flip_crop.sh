#!/bin/bash

train="python3 -m torch.distributed.launch --nproc_per_node=4 --use_env src/train.py --train-dataset PCD_CV_woRot --test-dataset GSV --test-dataset2 TSUNAMI --data-cv 0 --input-size 256 --randomflip 0.5 --randomcrop --model resnet18_mtf_msf_fcn --mtf iade --msf 4 --warmup --loss bi --loss-weight"

test="python3 src/train.py --test-only --save-imgs --model resnet18_mtf_msf_fcn --mtf iade --msf 4 --train-dataset PCD_CV --test-dataset TSUNAMI --data-cv 0 --input-size 256 --resume output/deeplabv3_bi_fpn4_bdae_vgg16bn_PCD_CV_0_256/2021-10-02_10:58:34/best.pth"


