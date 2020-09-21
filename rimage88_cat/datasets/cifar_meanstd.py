import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datasets.cvtransforms as transforms
import torch
import torchvision
import numpy as np
import cv2
from tensorboardX import SummaryWriter
from turbojpeg import TurboJPEG
from jpeg2dct.numpy import load, loads

transform = transforms.Compose([
                transforms.Upscale(upscale_factor=2),
                transforms.TransformUpscaledDCT(),
                transforms.ToTensorDCT(),
                # transforms.SubsetDCT(channels=args.subset),
                transforms.Aggregate(),
            ])

class myToTensor(object):
    def __call__(self, img):
        img_cv2_bgr = cv2.cvtColor(np.asarray(img.convert('RGB')),cv2.COLOR_RGB2BGR)
        img_cv2_rgb = cv2.cvtColor(img_cv2_bgr,cv2.COLOR_BGR2RGB)
        # cv2.imwrite('test_cvbgr.png', img_cv2_bgr)
        # cv2.imwrite('test_cvrgb.png', img_cv2_rgb)
        # img.save('test_pil.png')
        # imgYCC = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2YCrCb)
        imgcv2bgr_torch = torch.from_numpy(img_cv2_bgr.transpose((2, 0, 1))).float()
        imgcv2rgb_torch = torch.from_numpy(img_cv2_rgb.transpose((2, 0, 1))).float()
        
        return imgcv2bgr_torch, imgcv2rgb_torch

pre_trans = torchvision.transforms.Compose([torchvision.transforms.Resize(256), myToTensor()])

train_data=torchvision.datasets.CIFAR10(root='../cifar10/', train=True, transform=pre_trans, download=False,)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=1, shuffle=False)

# writer = SummaryWriter()
mean_all = []
std_all = []

for batch_idx, (img, target) in enumerate(train_loader):
    imgcv2bgr_torch, imgcv2rgb_torch = img
    img_tran = transform(imgcv2bgr_torch[0].numpy().transpose((1, 2, 0))) #[192 16 16]
    
    # jpeg = TurboJPEG('/usr/lib/libturbojpeg.so')
    # img_encode = jpeg.encode(imgcv2bgr_torch[0].numpy().transpose((1, 2, 0)), quality=100, jpeg_subsample=2)
    # dct_y, dct_cb, dct_cr = loads(img_encode)  # 28

    # out_file = open('output.jpg', 'wb')
    # out_file.write(jpeg.encode(imgcv2bgr_torch[0].numpy().transpose((1, 2, 0))))
    # out_file.close()

    img_tran_l = img_tran.view([192, -1]).numpy()
    mean_all.append(img_tran_l.mean(axis=1))
    std_all.append(img_tran_l.std(axis=1))

    each=2500
    if batch_idx%each==0:
        print(batch_idx)
    #     writer.add_image('input_img', imgcv2bgr_torch[0].long(), batch_idx//each)

mean_s = np.mean(np.array(mean_all), axis=0)
std_s = np.mean(np.array(std_all), axis=0)
print('mean_s\n', mean_s)
print('std_s\n', std_s)
import pdb
pdb.set_trace()

# writer.close()