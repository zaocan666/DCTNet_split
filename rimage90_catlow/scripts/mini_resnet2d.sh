#!/usr/bin/env bash
subset=24
# subset=${subset:-0}
echo "subset: $subset"
python main/mini_imagenet_resnet2d.py -j 64 --gpu-id 0,1,2,3 --arch 50 --subset $subset --data /data --train_batch 128 --lr 0.5 --epoch 120