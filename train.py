from __future__ import print_function
import sys
import argparse
import os
import shutil
import torch.nn as nn
import torch.nn.parallel
import time
import random
sys.path.append("D:/resden_final/models/cifar/")
import torch.backends.cudnn as cudnn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.utils.data as data
import models.cifar as models
from progress.bar import Bar as Bar
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from contextlib import redirect_stdout
from torchsummary import summary
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
from utils import AverageMeter, Cutout, accuracy
from utils.cutout import Cutout


parser = argparse.ArgumentParser(description='ResDen Model')
parser.add_argument('--dropo', '--dropout', default=0, type=float,metavar='Dropout', help='Dropout')
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('--cutout', action='store_true', default=False,help='apply cutout')
parser.add_argument('--length', type=int, default=16,help='length of the holes')
parser.add_argument('--schedule', type=int, nargs='+', default=[120, 200],help='learning rate decreases')
parser.add_argument('--n_holes', type=int, default=1, help='number of holes to cut out from image')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resden',choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resden)')
parser.add_argument('--depth', type=int, default=29, help='Model depth')
parser.add_argument('--manualSeed', type=int, help='ms')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model')
args = parser.parse_args()
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
state = {k: v for k, v in args._get_kwargs()}
torch.manual_seed(args.manualSeed)

# Use CUDA
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0
learn=0.1
assert args.dataset == 'cifar10', 'Only cifar-10 dataset'

def main():
    global best_acc
    global learn
    start_epoch = 0
    # Data
    print('Getting the Dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.cutout:
        transform_train.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataloader = datasets.CIFAR10
    trainset = dataloader(root='D:/resden_final/CIFAR/', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
    testset = dataloader(root='D:/resden_final/CIFAR/', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    print("Model '{}'".format(args.arch))
    model = models.__dict__[args.arch](
                num_classes=10,
                depth=args.depth,
                k1 = 12, 
                k2 = 12,
                dropRate=args.dropo,
            )
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    summary(model, (3, 32, 32))
    model_size = (sum(p.numel() for p in model.parameters())/1000000.0)
    print('Total Number of Parameters: %.2f Million' % model_size)
    with open('D:/resden_final/modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            summary(model, (3, 32, 32))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learn, momentum=0.9, weight_decay=1e-4)

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Accuracy:  %.2f' % (test_loss, test_acc))
        return

    tra=[]
    tea=[]
    trl=[]
    tel=[]
    # Train and val
    for epoch in range(start_epoch, 300):
        change_lr(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, 300, learn))
        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
        tra.append(train_acc)
        tea.append(test_acc)
        trl.append(train_loss)
        tel.append(test_loss)
        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

    print('Best acc:')
    print(best_acc)

    plt.figure(1)             
    plt.plot(tra)
    plt.title('Training Accuracy vs Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.savefig('D:/resden_final/train_acc.png')

    plt.figure(2)             
    plt.plot(tea)
    plt.title('Testing Accuracy vs Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.savefig('D:/resden_final/test_acc.png')

    plt.figure(3)             
    plt.plot(trl)
    plt.title('Training Loss vs Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.savefig('D:/resden_final/train_loss.png')

    plt.figure(4)             
    plt.plot(tel)
    plt.title('Testing Loss vs Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.savefig('D:/resden_final/test_loss.png')

def change_lr(optimizer, epoch):
    global state
    global learn
    if epoch in args.schedule:
        d=0
        learn=learn*0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = learn

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    data_time = AverageMeter()
    z=8
    batch_time = AverageMeter()
    ar=[]
    top1 = AverageMeter()
    a=0
    b=0
    c=0
    losses = AverageMeter()
    end = time.time()
    rs=[]
    top5 = AverageMeter()

    bar = Bar('Processing Train', max=len(trainloader))
    for bid, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        x=0
        cx=[]
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        er=0
        top5.update(prec5, inputs.size(0))
        f=0
        losses.update(loss.data, inputs.size(0))
        cb=0
        top1.update(prec1, inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size})| Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(batch=bid + 1,size=len(trainloader),bt=batch_time.avg,total=bar.elapsed_td, eta=bar.eta_td,loss=losses.avg, top1=top1.avg)
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    data_time = AverageMeter()
    c=0
    v=np.ones((1,2))
    batch_time = AverageMeter()
    top1 = AverageMeter()
    vv=[]
    losses = AverageMeter()
    x=1
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing Test', max=len(testloader))
    for bid, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        top1.update(prec1, inputs.size(0))
        losses.update(loss.data, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size})| Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(batch=bid + 1, size=len(testloader),bt=batch_time.avg,total=bar.elapsed_td, eta=bar.eta_td, loss=losses.avg, top1=top1.avg)
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

if __name__ == '__main__':
    main()