#!/usr/bin/env bash
subset_high=24
# subset=${subset:-0}
echo "subset_high: $subset_high"
CUDA_VISIBLE_DEVICES=4,5,6,7 python main/mini_imagenet_resnet_part.py -j 16 --arch 50 --bound 0.9 --subset_high $subset_high --data /data --train_batch 128 --lr 0.5 --epoch 90 \
--low_checkpoint None --lr_scheduler cosine