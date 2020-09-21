#!/usr/bin/env bash
# subset_high=24
# subset=${subset:-0}
echo "subset_high: $subset_high"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main/mini_imagenet_resnet_partlow.py -j 16 --arch 50 --bound 0.9 --data /data --train_batch 128 --lr 0.5 --epoch 90 \
--all_checkpoint ../rimage88_cat --lr_scheduler cosine