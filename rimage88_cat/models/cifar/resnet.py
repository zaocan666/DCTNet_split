'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from models.utils import *
from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        channel_increase=1
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16*channel_increase, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16*channel_increase)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16*channel_increase, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32*channel_increase, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64*channel_increase, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64*channel_increase, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56(classes=10):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=classes)


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])

class ResNet50DCT(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50DCT, self).__init__()
        model = resnet50(pretrained=pretrained)
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = norm_layer(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        in_ch, out_ch = 64, 96
        self.model = nn.Sequential(*list(model.children())[5:-1])
        self.fc = list(model.children())[-1]
        self.deconv1 = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, dct_y, dct_cb, dct_cr):
        dct_cb = self.deconv1(dct_cb)
        dct_cr = self.deconv2(dct_cr)
        x = torch.cat((dct_y, dct_cb, dct_cr), dim=1)

        x = self.model(x)
        x = x.reshape(x.size(0), -1)  # 2048
        x = self.fc(x)
        return x

        # x = self.layer1(x)  # 256x56x56
        # x = self.layer2(x)  # 512x28x28
        # x = self.layer3(x)  # 1024x14x14
        # x = self.layer4(x)  # 2048x7x7
        #
        # x = self.avgpool(x)  # 2048x1x1
        # x = x.reshape(x.size(0), -1)  # 2048
        # x = self.fc(x)  # 1000

        # return x

class ResNet50DCT_Upscaled(nn.Module):
    def __init__(self):
        super(ResNet50DCT_Upscaled, self).__init__()
        model = resnet50(pretrained=True)
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = norm_layer(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.model = nn.Sequential(*list(model.children())[4:-1])
        self.fc = list(model.children())[-1]

        upscale_factor_y, upscale_factor_cb, upscale_factor_cr = 1, 2, 2
        self.upconv_y = nn.Conv2d(in_channels=64, out_channels=22*upscale_factor_y*upscale_factor_y,
                                  kernel_size=3, stride=1, padding=1, bias=False)
        self.upconv_cb = nn.Conv2d(in_channels=64, out_channels=21*upscale_factor_cb*upscale_factor_cb,
                                  kernel_size=3, stride=1, padding=1, bias=False)
        self.upconv_cr = nn.Conv2d(in_channels=64, out_channels=21*upscale_factor_cr*upscale_factor_cr,
                                   kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_y = nn.BatchNorm2d(22)
        self.bn_cb = nn.BatchNorm2d(21)
        self.bn_cr = nn.BatchNorm2d(21)
        # self.pixelshuffle_y = nn.PixelShuffle(1)
        self.pixelshuffle_cb = nn.PixelShuffle(upscale_factor_cb)
        self.pixelshuffle_cr = nn.PixelShuffle(upscale_factor_cr)
        self.relu = nn.ReLU(inplace=True)
        self.init_weights()

    def init_weights(self):
        # initialize input layers
        for name, m in self.named_modules():
            if any(s in name for s in ['upconv_y', 'upconv_cb', 'upconv_cr', 'bn_y', 'bn_cb', 'bn_cr']):
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)

    def forward(self, dct_y, dct_cb, dct_cr):
        # dct_y = self.relu(self.bn_y(self.pixelshuffle_y(self.upconv_y(dct_y))))
        dct_y = self.relu(self.bn_y(self.upconv_y(dct_y)))
        dct_cb = self.relu(self.bn_cb(self.pixelshuffle_cb(self.upconv_cb(dct_cb))))
        dct_cr = self.relu(self.bn_cr(self.pixelshuffle_cr(self.upconv_cr(dct_cr))))
        x = torch.cat((dct_y, dct_cb, dct_cr), dim=1)

        x = self.model(x)
        x = x.reshape(x.size(0), -1)  # 2048
        x = self.fc(x)
        return x

class ResNetDCT_Upscaled_Static(nn.Module):
    def __init__(self, channels=0, pretrained=True, input_gate=False, classes=1000):
        super(ResNetDCT_Upscaled_Static, self).__init__()

        self.input_gate = input_gate

        model = resnet56(classes=classes)

        self.model = nn.Sequential(*list(model.children())[3:-1]) #same with paper
        self.fc = list(model.children())[-1]
        self.relu = nn.ReLU(inplace=True)
        
        if channels == 0 or channels == 192:
            out_ch = self.model[0][0].conv1.out_channels #16
            self.model[0][0].conv1 = nn.Conv2d(channels, out_ch, kernel_size=1, stride=1, bias=False)
            kaiming_init(self.model[0][0].conv1)

            # if len(self.model[0][0].shortcut)>0:
            # out_ch = self.model[0][0].shortcut[0].out_channels
            self.model[0][0].shortcut = nn.Conv2d(channels, out_ch, kernel_size=1, stride=1, bias=False)
            kaiming_init(self.model[0][0].shortcut)

            # temp_layer = conv3x3(channels, out_ch, 1)
            # temp_layer = nn.Conv2d(channels, out_ch, kernel_size=1, stride=1, bias=False)
            # temp_layer.weight.data = self.model[0][0].conv1.weight.data.repeat(1, 3, 1, 1)
            # self.model[0][0].conv1 = temp_layer

            # out_ch = self.model[0][0].downsample[0].out_channels
            # temp_layer = nn.Conv2d(channels, out_ch, kernel_size=1, stride=1, bias=False)
            # temp_layer.weight.data = self.model[0][0].downsample[0].weight.data.repeat(1, 3, 1, 1)
            # self.model[0][0].downsample[0] = temp_layer
        elif channels < 64:
            out_ch = self.model[0][0].conv1.out_channels
            # temp_layer = conv3x3(channels, out_ch, 1)
            temp_layer = nn.Conv2d(channels, out_ch, kernel_size=3, stride=1, bias=False)
            temp_layer.weight.data = self.model[0][0].conv1.weight.data[:, :channels]
            self.model[0][0].conv1 = temp_layer

            out_ch = self.model[0][0].shortcut[0].out_channels
            temp_layer = nn.Conv2d(channels, out_ch, kernel_size=1, stride=1, bias=False)
            temp_layer.weight.data = self.model[0][0].shortcut[0].weight.data[:, :channels]
            self.model[0][0].shortcut[0] = temp_layer

        if input_gate:
            self.inp_GM = GateModule192()
            self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if 'inp_gate_l' in str(name):
                m.weight.data.normal_(0, 0.001)
                m.bias.data[::2].fill_(0.1)
                m.bias.data[1::2].fill_(2)
            elif 'inp_gate' in str(name):
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)

    def forward(self, x):
        if self.input_gate:
            x, inp_atten = self.inp_GM(x)
        
        x = self.model(x)
        x = x.reshape(x.size(0), -1)  # 2048
        x = self.fc(x)
        if self.input_gate:
            return x, inp_atten
        else:
            return x

def main():
    import numpy as np

    channels = 192
    model_resnet50dct = ResNetDCT_Upscaled_Static(channels=channels)
    input = torch.from_numpy(np.random.randn(16, channels, 56, 56)).float()
    x = model_resnet50dct(input)
    print(x.shape)

    model_resnet50dct = resnet50(pretrained=True)
    input = torch.from_numpy(np.random.randn(16, 3, 224, 224)).float()
    x = model_resnet50dct(input)
    print(x.shape)

    # ResNet50DCT
    model_resnet50dct = ResNet50DCT()

    dct_y  = torch.from_numpy(np.random.randn(16, 64, 28, 28)).float()
    dct_cb = torch.from_numpy(np.random.randn(16, 64, 14, 14)).float()
    dct_cr = torch.from_numpy(np.random.randn(16, 64, 14, 14)).float()

    x = model_resnet50dct(dct_y, dct_cb, dct_cr)
    print(x.shape)

    # SE-ResNet50DCT
    model_seresnet50dct = SE_ResNet50DCT()
    x = model_seresnet50dct(dct_y, dct_cb, dct_cr)
    print(x.shape)

if __name__ == '__main__':
    main()
