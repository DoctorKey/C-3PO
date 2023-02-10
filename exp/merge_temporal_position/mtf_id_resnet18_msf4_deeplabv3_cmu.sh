#!/bin/bash

train="python3 -m torch.distributed.launch --nproc_per_node=4 --use_env src/train.py --train-dataset VL_CMU_CD --test-dataset VL_CMU_CD --input-size 512 --model mtf_resnet18_msf_deeplabv3 --mtf id --warmup --loss bi --loss-weight"

test="python3 src/train.py --test-only --model mtf_resnet18_msf_deeplabv3 --mtf id --train-dataset VL_CMU_CD --test-dataset VL_CMU_CD --input-size 512 --resume output/deeplabv3_tri_mulitfeature_resnet18_VL_CMU_CD_0_512/2021-09-15_09:59:00/checkpoint.pth"
