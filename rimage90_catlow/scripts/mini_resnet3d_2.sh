#!/usr/bin/env bash
subset=$1
subset=${subset:-0}
echo "subset: $subset"
python main/mini_imagenet_resnet3d_2.py -j 64 --gpu-id 4,5,6,7 --arch 101 --subset $subset --data /data --train_batch 128 --lr 0.05 --w 4 --epoch 90 --channels 12 --depth 8