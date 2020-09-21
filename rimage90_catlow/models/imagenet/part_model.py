import sys
import os
import pdb
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(".")
import torch.nn as nn
import torch.nn.functional as F
import torch

from models.imagenet.resnet import ResNetDCT_Upscaled_Static

def conv_block(in_channel, out_channel):
    return nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=True),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True)
            )

class Part_model(nn.Module):
    def __init__(self, low_checkpoint, subset_low, subset_high, low_reduce_factor, num_classes, bound, first_stride, cat_pos):
        super(Part_model, self).__init__()
        
        self.bound = bound
        self.cat_pos = cat_pos

        self.low_input_layer = conv_block(in_channel=int(subset_low), out_channel=int(64*low_reduce_factor))
        low_model = ResNetDCT_Upscaled_Static(channels=int(64*low_reduce_factor), pretrained=False, classes=100, reduce_factor=low_reduce_factor, first_stride=first_stride, 
                                                cat_factor=0, cat_pos=[])
        if low_checkpoint:
            low_model.load_state_dict({k.replace('module.',''):v for k,v in low_checkpoint.items()})
        self.low_model_feature = low_model.model
        self.low_model_fc = low_model.fc
        low_model_feature_num = self.low_model_feature[-2][-1].conv3.weight.shape[0]

        high_reduce_factor = 1-low_reduce_factor
        cat_factor = low_reduce_factor/high_reduce_factor
        self.high_input_layer = conv_block(in_channel=int(subset_high)-int(subset_low), out_channel=int(64*high_reduce_factor))
        high_model = ResNetDCT_Upscaled_Static(channels=int(64*high_reduce_factor*(1+cat_factor)), pretrained=False, classes=100, reduce_factor=high_reduce_factor, first_stride=first_stride,
                                                cat_factor=cat_factor, cat_pos=cat_pos)
        self.high_model_feature = high_model.model
        high_model_feature_num = self.high_model_feature[-2][-1].conv3.weight.shape[0]

        # self.fc_fuse = nn.Linear(low_model_feature_num+high_model_feature_num, num_classes)
        self.fc_high = nn.Linear(high_model.fc.weight.shape[1], num_classes)

    def forward(self, image_low, image_high, eval_flag=0):
        if eval_flag==2:
            return self.low_forward_pure(image_low)
            
        low_features = self.low_forward(image_low)
        # low_feature = low_feature.detach() #[bn, low_model_feature_num, 1, 1]
        # low_feature_re = low_features[-1].reshape(low_features[-1].size(0), -1) #[bn, low_model_feature_num]
        
        if eval_flag==0:
            high_input = image_high
            # high_input = torch.cat([image_low, image_high], dim=1)
            high_feature = self.high_forward(high_input, low_features)
            high_feature_re = high_feature.reshape(high_feature.size(0), -1) #[bn, high_model_feature_num]
            # fuse_feature = torch.cat([low_feature_re, high_feature_re], dim=1) #[bn, low_model_feature_num + high_model_feature_num]
            
            return self.fc_high(high_feature_re)
            # return self.fc_fuse(fuse_feature)
        elif eval_flag==1:
            low_feature_re = low_features[-1].reshape(low_features[-1].size(0), -1) #[bn, low_model_feature_num]
            # low_feature_re = low_feature.reshape(low_features[-1].size(0), -1) #[bn, low_model_feature_num]
            low_fc_score = self.low_model_fc(low_feature_re) #[bn, num_classes]
            low_fc_softmax = F.softmax(low_fc_score, dim=1) #[bn, num_classes]

            max_indexes = torch.argmax(low_fc_softmax, dim=1) #[bn]
            max_value = low_fc_softmax[[i for i in range(low_fc_softmax.shape[0])], max_indexes] #[bn]
            bound_flag = (max_value > self.bound)
            
            if (~bound_flag).any():
                # low_qualified_score = low_fc_score[bound_flag] #[x, num_classes]
                low_unqualified_feature_re = low_feature_re[~bound_flag] #[bn-x, low_model_feature_num]
                high_unqualified_image = image_high[~bound_flag] #[bn-x, subset_high, h, w]

                low_unqualified_features = []
                for low_f in low_features[:len(self.cat_pos)+1]:
                    low_unqualified_features.append(low_f[~bound_flag])

                high_unqualified_feature = self.high_forward(high_unqualified_image, low_unqualified_features)
                high_unqualified_feature_re = high_unqualified_feature.reshape(high_unqualified_feature.size(0), -1) #[bn-x, high_model_feature_num]

                # fuse_unqualified_feature = torch.cat([low_unqualified_feature_re, high_unqualified_feature_re], dim=1) #[bn-x, low_model_feature_num + high_model_feature_num]
                # unqualified_score = self.fc_fuse(fuse_unqualified_feature) #[bn-x, num_classes]
                unqualified_score = self.fc_high(high_unqualified_feature_re) #[bn-x, num_classes]
                
                low_fc_score[~bound_flag]=unqualified_score

            return low_fc_score

    def low_forward(self, image_low):
        # [bs, in_c, h, w]
        image_input = self.low_input_layer(image_low)
        middle_feature0_0 = self.low_model_feature[0][0](image_input)
        middle_feature0 = self.low_model_feature[0][1:](middle_feature0_0) # [bs, 128, h, w]
        middle_feature1 = self.low_model_feature[1](middle_feature0) # [bs, 256, h/2, w/2]
        middle_feature2 = self.low_model_feature[2](middle_feature1) # [bs, 512, h/4, w/4]
        middle_feature3 = self.low_model_feature[3](middle_feature2) # [bs, 1024, h/8, w/8]
        output_low = self.low_model_feature[4](middle_feature3) # [bs, 1024, 1, 1]

        return [image_input, middle_feature0_0, middle_feature0, middle_feature1, middle_feature2, middle_feature3, output_low]

    def low_forward_pure(self, image_low):
        low_input = self.low_input_layer(image_low)
        low_input = low_input.detach()
        feature_low = self.low_model_feature(low_input)
        feature_low_re = feature_low.reshape(feature_low.size(0), -1)
        output = self.low_model_fc(feature_low_re)
        return output

    def high_forward(self, image_high, low_features):
        # [bs, in_c, h, w]
        image_input = self.high_input_layer(image_high)
        image_input_plus = torch.cat([image_input, low_features[0]], dim=1)
        
        middle_feature0_0 = self.high_model_feature[0][0](image_input_plus) #[bs, 128, 56, 56]
        if [0,0] in self.cat_pos:
            middle_feature0_0_plus = torch.cat([middle_feature0_0, low_features[1]], dim=1)
        else:
            middle_feature0_0_plus = middle_feature0_0

        middle_feature0 = self.high_model_feature[0][1:](middle_feature0_0_plus) # [bs, 128, h, w]
        # middle_feature0_plus = middle_feature0 + low_features[0]
        if 0 in self.cat_pos:
            middle_feature0_plus = torch.cat([middle_feature0, low_features[2]], dim=1)
        else:
            middle_feature0_plus = middle_feature0

        middle_feature1 = self.high_model_feature[1](middle_feature0_plus) # [bs, 256, h/2, w/2]
        if 1 in self.cat_pos:
            middle_feature1_plus = torch.cat([middle_feature1, low_features[3]], dim=1)
        else:
            middle_feature1_plus = middle_feature1
        
        middle_feature2 = self.high_model_feature[2](middle_feature1_plus) # [bs, 512, h/4, w/4]
        if 2 in self.cat_pos:
            middle_feature2_plus = torch.cat([middle_feature2, low_features[4]], dim=1)
        else:
            middle_feature2_plus = middle_feature2

        middle_feature3 = self.high_model_feature[3](middle_feature2_plus) # [bs, 1024, h/8, w/8]
        if 3 in self.cat_pos:
            middle_feature3_plus = torch.cat([middle_feature3, low_features[5]], dim=1)
        else:
            middle_feature3_plus = middle_feature3
        
        output_high = self.high_model_feature[4](middle_feature3_plus) # [bs, 1024, 1, 1]
        
        return output_high

if __name__ == '__main__':
    subset_low = '6'
    subset_high = '24'
    input_low = torch.randn([2, int(subset_low), 56, 56])
    input_high = torch.randn([2, int(subset_high), 56, 56])

    model = Part_model(low_checkpoint=None, subset_low=subset_low, subset_high=subset_high, 
        low_reduce_factor=0.5, num_classes=100, bound=0.9, first_stride=2)
    output = model(input_low, input_high, eval_flag=False)