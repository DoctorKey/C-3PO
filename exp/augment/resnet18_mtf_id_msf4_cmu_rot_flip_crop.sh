#!/bin/bash

train="python3 -m torch.distributed.launch --nproc_per_node=4 --use_env src/train.py --train-dataset VL_CMU_CD --test-dataset VL_CMU_CD --input-size 512 --randomflip 0.5 --randomcrop --model resnet18_mtf_msf_fcn --mtf id --msf 4 --warmup --loss bi --loss-weight"

test="python3 src/train.py --test-only --save-imgs --model resnet18_mtf_msf_fcn --mtf id --msf 4 --train-dataset VL_CMU_CD --test-dataset VL_CMU_CD --input-size 512 --resume output/deeplabv3_bi_fpn4_bdae_vgg16bn_PCD_CV_0_256/2021-10-02_10:58:34/best.pth"


