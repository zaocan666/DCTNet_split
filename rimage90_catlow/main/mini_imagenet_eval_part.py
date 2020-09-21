import os
import sys
import numpy as np
import time
import pdb
import copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from models.imagenet.part_model import Part_model
from datasets.dataloader_mini_imagenet_dct import valloader_upscaled_static, trainloader_upscaled_static
from utils import Bar, AverageMeter, accuracy, cal_his_acc

def get_model(checkpoint_path, bound):
    all_checkpoint = torch.load(checkpoint_path)
    print('check point path: ', checkpoint_path)
    print('epoch: ', all_checkpoint['epoch'])
    print('best prec1: ', all_checkpoint['best_prec1'])

    subset_low = str(all_checkpoint['state_dict']['module.low_input_layer.0.weight'].shape[1])
    subset_high = str(all_checkpoint['state_dict']['module.high_input_layer.0.weight'].shape[1]+int(subset_low))
    low_reduce_factor = all_checkpoint['state_dict']['module.low_model_feature.0.0.conv1.weight'].shape[0]/64.0

    print('subset_low: ', subset_low)
    print('subset_high: ', subset_high)
    print('low_reduce_factor: ', low_reduce_factor)

    model = Part_model(low_checkpoint=None, subset_low=subset_low, subset_high=subset_high, 
        low_reduce_factor=low_reduce_factor, num_classes=100, bound=bound, first_stride=2, cat_pos=[])
    # model = Part_model(low_checkpoint=None, subset_low=subset_low, subset_high=subset_high, 
    #     low_reduce_factor=low_reduce_factor, num_classes=100, bound=bound)
    # model = ResNetDCT_Upscaled_Static(channels=subset, pretrained=False, classes=100)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(all_checkpoint['state_dict'])
    model.eval()
    return model, subset_low, subset_high

def test(val_loader, model, criterion, epoch, mode):
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
        for batch_idx, (image_low, image_high, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # ([200, 48, 56, 56])  ([200])
            image_low, image_high, target = image_low.cuda(), image_high.cuda(), target.cuda()

            if mode == 'normal':
                # compute output
                output = model(image_low, image_high, eval_flag=1)
            elif mode == 'low':
                output = model(image_low, image_high, eval_flag=2)
            elif mode == 'both':
                output = model(image_low, image_high, eval_flag=0)

            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            losses.update(loss.item(), image_low.size(0))
            top1.update(prec1.item(), image_low.size(0))
            top5.update(prec5.item(), image_low.size(0))

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
    def __init__(self, subset_low, subset_high, data, test_batch, train_batch, workers=8):
        self.subset_low = subset_low
        self.subset_high = subset_high
        self.data = data
        self.test_batch = test_batch
        self.train_batch = train_batch
        self.workers = workers

def prob_histogram(output_all, target_all, histogram_all_flag=False):
    bounds_updown = [[(i-1)/10.0, i/10.0] for i in range(1, 11)]
    histogram_correct, value_acc, _, _ = cal_his_acc(bounds_updown, output_all, target_all, histogram_all_flag)

    return histogram_correct, value_acc

def test_dataloader(val_loader, output_all, target_all, model):
    image_low, image_high, target = val_loader.dataset.__getitem__(0)
    image_low, image_high = image_low.cuda(), image_high.cuda()
    output = model(image_low.unsqueeze(0), image_high.unsqueeze(0), eval_flag=2)
    output_softmax = output[0].softmax(0)

def low_model_output_visualize(val_loader, output_all, target_all, model):
    max_indexes = np.argmax(output_all, axis=1) #[10000]
    max_value = output_all[[i for i in range(output_all.shape[0])], max_indexes] #[10000]

    image_path = 'max_min_image'
    if not os.path.exists(image_path):
        os.mkdir(image_path)

    max_value_sort = np.argsort(max_value) #[10000], small 2 large
    num = 10
    for i in range(num):
        max_index_i = max_value_sort[-i-1]
        min_index_i = max_value_sort[i]

        max_path = val_loader.dataset.images[max_index_i]
        min_path = val_loader.dataset.images[min_index_i]

        max_max_value = max_value[max_index_i]
        min_max_value = max_value[min_index_i]

        suffix = max_path.split('.')[-1]
        os.system('cp '+max_path+' '+os.path.join(image_path, 'max_%02d_%.4f.'%(i, max_max_value)+suffix))
        os.system('cp '+min_path+' '+os.path.join(image_path, 'min_%02d_%.4f.'%(i, min_max_value)+suffix))

if __name__=='__main__':
    checkpoint_path = [p for p in os.listdir('./') if p.startswith('best_') and p.endswith('.pth')][0]
    bound = 0.85
    model, subset_low, subset_high = get_model(checkpoint_path, bound=bound)

    args = dumpy_args(subset_low=subset_low, subset_high=subset_high, data='/data', test_batch=50, train_batch=128)
    # train_loader = trainloader_upscaled_static(args, model='resnet')
    val_loader = valloader_upscaled_static(args, model='resnet_part')

    criterion = nn.CrossEntropyLoss().cuda()

    losses_avg, top1_avg, top5_avg, output_all, target_all = test(val_loader, model, criterion, 0, mode='low') #[10000, 100], [10000,]
    print('only low model val. losses avg: %.4f\t top1 avg: %.2f\t top5 avg: %.2f'%(losses_avg, top1_avg, top5_avg))
    # losses_avg, top1_avg, top5_avg, _, _ = test(val_loader, model, criterion, 0, mode='normal')
    # print('normal val. losses avg: %.4f\t top1 avg: %.2f\t top5 avg: %.2f'%(losses_avg, top1_avg, top5_avg))
    # losses_avg, top1_avg, top5_avg, _, _ = test(val_loader, model, criterion, 0, mode='both')
    # print('both val. losses avg: %.4f\t top1 avg: %.2f\t top5 avg: %.2f'%(losses_avg, top1_avg, top5_avg))
    
    low_model_output_visualize(val_loader, output_all, target_all, model)
    histogram_correct, value_acc, _, _ = cal_his_acc([[bound, 1.0]], output_all, target_all, histogram_all_flag=True)
    # histogram 3:  ['0.680']
    # value acc 3:  ['0.954']
    # histogram, value_acc = prob_histogram(output_all, target_all, histogram_all_flag=True)
    print('histogram 3: ', list(map(lambda x: '%.3f'%x, histogram_correct)))
    print('value acc 3: ', list(map(lambda x: '%.3f'%x, value_acc)))
    # histogram 3:  ['0.000', '0.009', '0.028', '0.035', '0.046', '0.057', '0.054', '0.056', '0.075', '0.640']
    # value acc 3:  ['0.000', '0.114', '0.203', '0.321', '0.378', '0.486', '0.585', '0.639', '0.774', '0.964']
    pdb.set_trace()
