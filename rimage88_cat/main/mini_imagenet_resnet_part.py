from __future__ import print_function

import argparse
import os
import sys
import warnings
import shutil
import time
import random
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
import torch.optim as optim
import torch.utils.data as data
# from models.imagenet.resnet3d_2 import generate_model
from models.imagenet.part_model import Part_model
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, save_checkpoint, BalancedDataParallel
from datasets.dataloader_mini_imagenet_dct import valloader_upscaled_static, trainloader_upscaled_static
from tensorboardX import SummaryWriter
import pdb
import copy


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--lr', default=1e-4, type=float, metavar='N',
                    help='learning rate')
parser.add_argument('--bound', default=0.9, type=float, metavar='N',
                    help='boundary of part model')                
parser.add_argument('--test_batch', default=128, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--train_batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--epoch', default=90, type=int, metavar='N',
                    help='total epoch')
parser.add_argument('--lr_scheduler', default='cosine', type=str, metavar='N',
                    help='lr_scheduler')

# Architecture
parser.add_argument('--arch', default=50, type=int, metavar='N',
                    help='architecture')
# parser.add_argument('--w', default=4, type=int, metavar='N',
#                     help='widen factor')
# parser.add_argument('--channels', default=12, type=int, metavar='N',
#                     help='channels')
# parser.add_argument('--depth', default=8, type=int, metavar='N',
#                     help='depth')

# Miscs
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--subset_high', default='24', type=str, help='subset_high of y, cb, cr')
#Device options
# parser.add_argument('--gpu-id', default='0,1,2,3,4,5,6,7', type=str,
#                     help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--pretrained', default='False', type=str2bool,
                    help='load pretrained model or not')
parser.add_argument('--low_checkpoint', type=str,
                    help='low model checkpoint')              

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
# args.subset = str(args.depth*args.channels)
args.test_batch = args.train_batch

if args.low_checkpoint=='None':
    args.subset_low = '6'
    low_reduce_factor = 0.5
    low_checkpoint = None
else:
    if not args.low_checkpoint.endswith('.pth'):
        args.low_checkpoint = [os.path.join(args.low_checkpoint, p) for p in os.listdir(args.low_checkpoint) if p.startswith('best_') and p.endswith('.pth')][0]
    print('low checkpoint:', args.low_checkpoint)
    low_checkpoint = torch.load(args.low_checkpoint)['state_dict']
    args.subset_low = str(low_checkpoint['module.model.0.0.conv1.weight'].shape[1])
    low_reduce_factor = low_checkpoint['module.model.0.0.conv1.weight'].shape[0]/64.0

model_name = 'fre_resnet_mini_arch_resnet_' + str(args.arch) + '_sgd_lr_' + str(args.lr) + '_bz_' + str(args.train_batch) + '_scheduler_' + str(args.lr_scheduler) + '_epoch_' + str(args.epoch) \
 + '_subset-high_' + str(args.subset_high) + '_subset-low_' + str(args.subset_high) + '_low-reduce-factor_' + str(low_reduce_factor) + '_bound_' + str(args.bound)
print(model_name)

path = './results/' + model_name + '/'
record_file = path + 'process.txt'

isExists = os.path.exists(path)
if not isExists:
    os.makedirs(path)

# Use CUDA
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_prec1 = 0  # best test accuracy

def main():
    global args, best_prec1

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # model = generate_model(args.arch, n_classes=100, n_input_channels=args.channels, widen_factor=args.w)
    model = Part_model(low_checkpoint=low_checkpoint, subset_low=args.subset_low, subset_high=args.subset_high, 
        low_reduce_factor=low_reduce_factor, num_classes=100, bound=args.bound, first_stride=2, cat_pos=[])
    
    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda()
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=False)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # Resume
    title = 'ImageNet-resnet' + str(args.arch)
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # if args.resume:
    #     # Load checkpoint.
    #     print('==> Resuming from checkpoint..')
    #     checkpoint = torch.load(args.resume)
    #     args.start_epoch = checkpoint['epoch']
    #     best_prec1 = checkpoint['best_prec1']
    #     model.load_state_dict(checkpoint['state_dict'])
    #     print("=> loaded checkpoint '{}' (epoch {})"
    #           .format(args.resume, checkpoint['epoch']))
    #     args.checkpoint = os.path.dirname(args.resume)

    model = torch.nn.DataParallel(model).cuda()
    # model = BalancedDataParallel(gpu0_bsz=1, module=model, dim=0).cuda()

    cudnn.benchmark = True
    # print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000.0))

    # Data loading code
    train_loader = trainloader_upscaled_static(args, model='resnet_part')
    val_loader = valloader_upscaled_static(args, model='resnet_part')

    # Define Tensorboard Summary
    writer = SummaryWriter()

    best_acc1=0
    best_train_acc1=0
    best_model_stateDic=model.state_dict()
    best_epoch=0
    for epoch in range(args.epoch):
        train_loss, train_acc_top1, train_acc_top5 = train(train_loader, model, criterion, optimizer, epoch)
        # scheduler.step()
        print(' Train Loss:  %.8f, Train Acc Top1:  %.2f, Train Acc Top5:  %.2f' % (train_loss, train_acc_top1, train_acc_top5))
        test_loss, test_acc_top1, test_acc_top5 = test(val_loader, model, criterion, epoch)
        print(' Test Loss:  %.8f, Test Acc Top1:  %.2f, Test Acc Top5:  %.2f' % (test_loss, test_acc_top1, test_acc_top5))
        writer.add_scalars(model_name+'/loss',{'train':train_loss,'valid':test_loss},epoch)
        writer.add_scalars(model_name+'/Acc Top1',{'train':train_acc_top1,'valid':test_acc_top1},epoch)
        writer.add_scalars(model_name+'/Acc Top5',{'train':train_acc_top5,'valid':test_acc_top5},epoch)
        writer.add_scalar(model_name+'/lr', optimizer.param_groups[0]['lr'], epoch)

        if test_acc_top1>best_acc1:
            best_acc1=test_acc_top1
            best_train_acc1=train_acc_top1
            best_model_stateDic=copy.deepcopy(model.state_dict())
            best_epoch=epoch

        # save_checkpoint(
        #     save_path=os.path.join(args.checkpoint, 'e%d_trA_%.2f_teA_%.2f.pth'%(epoch, train_acc_top1, test_acc_top1)), 
        #     model_stateDic=model.state_dict(), 
        #     best_prec1=best_acc1, 
        #     epoch=epoch
        #     )

    writer.close()
    print('best\tepoch:%d\ttrain top1 Acc:%.2f\ttest top1 Acc:%.2f'%(best_epoch, best_train_acc1, best_acc1))

    save_checkpoint(
            save_path='best_e%d_trA_%.2f_teA_%.2f.pth'%(best_epoch, best_train_acc1, best_acc1), 
            model_stateDic=best_model_stateDic,
            best_prec1=best_acc1,
            epoch=best_epoch
            )

def adjust_learning_rate_iter_warmup(optimizer, epoch, step, len_epoch):
    if args.lr_scheduler == 'multiStep':
        factor = epoch // 30
        if epoch >= 80:
            factor = factor + 1
        lr = args.lr*(0.1**factor)
        """Warmup"""
        if epoch < 5:
            lr = lr/4 + (lr - lr/4)*float(1 + step + epoch*len_epoch)/(5.*len_epoch)

    elif args.lr_scheduler == 'cosine':
        if epoch < 5:
            lr = args.lr * float(1 + step + epoch*len_epoch)/(5.*len_epoch)
        else:
            lr = 0.5 * args.lr * (1 + math.cos(math.pi * float((epoch-5)*len_epoch+step) / float((args.epoch-5)*len_epoch)))
    else:
        assert('lr_scheduler wrong')

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(train_loader, model, criterion, optimizer, epoch):
    bar = Bar('Train Processing', max=len(train_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.train()

    end = time.time()

    train_batch = len(train_loader)

    for batch_idx, (image_low, image_high, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        adjust_learning_rate_iter_warmup(optimizer, epoch, batch_idx, train_batch)

        # non_blocking=True
        image_low, image_high, target = image_low.cuda(), image_high.cuda(), target.cuda()  # ([200, 96, 56, 56])  ([200])

        # print(image.shape)
        # print(target.shape)
        # pdb.set_trace()
        # data = image.reshape(image.size(0),args.channels,args.depth,56,56)
        
        # compute output
        # output_all = model(image_low, image_high, eval_flag=False)
        # output_low = model.module.low_model_fc(model.module.low_model_feature(image_low).reshape(image_low.size(0), -1))

        # loss_all = criterion(output_all, target)
        # loss_low = criterion(output_low, target)
        # loss = loss_all + loss_low

        output = model(image_low, image_high, eval_flag=False)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        losses.update(loss.item(), image_low.size(0))
        top1.update(prec1.item(), image_low.size(0))
        top5.update(prec5.item(), image_low.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        record  = '({epoch} {batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .2f} | top5: {top5: .2f}'.format(
                    epoch=epoch,
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.suffix = record
        with open(record_file, 'a') as file_object:
            file_object.write(record+"\n")

        bar.next()
    bar.finish()

    return (losses.avg, top1.avg, top5.avg)


def test(val_loader, model, criterion, epoch):
    bar = Bar('Test Processing', max=len(val_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for batch_idx, (image_low, image_high, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # ([200, 48, 56, 56])  ([200])
            image_low, image_high, target = image_low.cuda(), image_high.cuda(), target.cuda()
            # data = image.reshape(image.size(0),args.channels,args.depth,56,56)

            # compute output
            output = model(image_low, image_high, eval_flag=False)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            losses.update(loss.item(), image_low.size(0))
            top1.update(prec1.item(), image_low.size(0))
            top5.update(prec5.item(), image_low.size(0))

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
            with open(record_file, 'a') as file_object:
                file_object.write(record+"\n")
            bar.next()
            
        bar.finish()
    return (losses.avg, top1.avg, top5.avg)

if __name__ == '__main__':
    main()
