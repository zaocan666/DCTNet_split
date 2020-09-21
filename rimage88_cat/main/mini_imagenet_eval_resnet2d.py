import os
import sys
import numpy as np
import time
import pdb
import copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from models.imagenet.resnet import ResNetDCT_Upscaled_Static
from datasets.dataloader_mini_imagenet_dct import valloader_upscaled_static, trainloader_upscaled_static
from utils import Bar, AverageMeter, accuracy, cal_his_acc

def get_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    print('check point path: ', checkpoint_path)
    print('epoch: ', checkpoint['epoch'])
    print('best prec1: ', checkpoint['best_prec1'])

    conv1_name = [name for name in checkpoint['state_dict'].keys() if '.0.0.conv1.weight' in name][0]
    subset = checkpoint['state_dict']['module.model.0.0.conv1.weight'].shape[1]
    print('subset: ', subset)

    model = ResNetDCT_Upscaled_Static(channels=subset, pretrained=False, classes=100)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, subset

def test(val_loader, model, criterion, epoch):
    bar = Bar('Test Processing', max=len(val_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    output_all = []
    target_all = []
    with torch.no_grad():
        end = time.time()
        for batch_idx, (image, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # ([200, 48, 56, 56])  ([200])
            image, target = image.cuda(), target.cuda()
            # data = image.reshape(image.size(0),args.channels,args.depth,56,56)

            # compute output
            output = model(image)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))
            top5.update(prec5.item(), image.size(0))

            output_all.append(torch.softmax(output, dim=1).cpu().data.numpy())
            target_all.append(target.cpu().data.numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            record  = '({epoch} {batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .2f} | top5: {top5: .2f}'.format(
                        epoch=epoch,
                        batch=batch_idx + 1,
                        size=len(val_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )

            bar.suffix = record
            bar.next()
            
        bar.finish()
    
    return (losses.avg, top1.avg, top5.avg, np.concatenate(output_all, axis=0), np.concatenate(target_all, axis=0))

class dumpy_args():
    def __init__(self, subset, data, test_batch, train_batch, workers=8):
        self.subset=subset
        self.data = data
        self.test_batch = test_batch
        self.train_batch = train_batch
        self.workers = workers

def prob_histogram(output_all, target_all):
    bounds_updown = [[(i-1)/10.0, i/10.0] for i in range(1, 11)]
    histogram_correct, value_acc, _, _ = cal_his_acc(bounds_updown, output_all, target_all)

    return histogram_correct, value_acc

if __name__=='__main__':
    checkpoint_path = [p for p in os.listdir('./') if p.startswith('best_') and p.endswith('.pth')][0]
    model, subset = get_model(checkpoint_path)

    args = dumpy_args(subset=str(subset), data='/data', test_batch=50, train_batch=128)
    # train_loader = trainloader_upscaled_static(args, model='resnet')
    val_loader = valloader_upscaled_static(args, model='resnet')

    criterion = nn.CrossEntropyLoss().cuda()

    losses_avg, top1_avg, top5_avg, output_all, target_all = test(val_loader, model, criterion, 0) #[10000, 100], [10000,]
    print('val. losses avg: %.4f\t top1 avg: %.2f\t top5 avg: %.2f'%(losses_avg, top1_avg, top5_avg))
    histogram, value_acc = prob_histogram(output_all, target_all)
    print('histogram 3: ', list(map(lambda x: '%.3f'%x, histogram)))
    #['0.000', '0.001', '0.007', '0.014', '0.019', '0.028', '0.032', '0.042', '0.070', '0.787']
    print('value acc 3: ', list(map(lambda x: '%.3f'%x, value_acc)))
    #['nan', '0.058', '0.242', '0.304', '0.366', '0.442', '0.515', '0.628', '0.702', '0.957']
    pdb.set_trace()
