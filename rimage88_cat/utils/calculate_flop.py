import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchsummaryX import summary
from models.imagenet.part_model import Part_model
import torch
import torch.nn as nn

subset_low = 6
subset_high = 24
low_reduce_factor = 0.5
mode = 'high'

model = Part_model(low_checkpoint=None, subset_low=subset_low, subset_high=subset_high, 
        low_reduce_factor=low_reduce_factor, num_classes=100, bound=1)

if mode == 'normal':
    # compute output
    summary(model, torch.rand([1, int(subset_low), 56, 56]), torch.rand([1, int(subset_high)-int(subset_low), 56, 56]), True)
elif mode == 'low':
    part_model = nn.Sequential(
        model.low_model_feature,
        nn.Flatten(),
        model.low_model_fc,
    )
    summary(part_model, torch.rand([1, int(subset_low), 56, 56]))
elif mode == 'high':
    part_model = nn.Sequential(
        model.high_model_feature,
        nn.Flatten(),
        nn.Linear(in_features=int(2048*low_reduce_factor), out_features=100),
    )
    summary(part_model, torch.rand([1, int(subset_high)-int(subset_low), 56, 56]))
elif mode == 'both':
    summary(model, torch.rand([1, int(subset_low), 56, 56]), torch.rand([1, int(subset_high)-int(subset_low), 56, 56]), False)
