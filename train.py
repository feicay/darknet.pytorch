import torch as t
import argparse
import os
import sys
import cv2
import re
import model.network as net
import model.loss as loss
import model.data as dat
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from torch.autograd import Variable
import time

def parse_args():
    parser = argparse.ArgumentParser(description='train a network')
    parser.add_argument('--dataset',help='training set config file',default='dataset/coco.data',type=str)
    parser.add_argument('--netcfg',help='the network config file',default='cfg/yolov2.cfg',type=str)
    parser.add_argument('--weight',help='the network weight file',default='weight/yolov2_final.weight',type=str)
    parser.add_argument('--batch',help='training batch size',default=16,type=int)
    parser.add_argument('--vis',help='visdom the training process',default=1,type=int)
    parser.add_argument('--cuda',help='use the GPU',default=1,type=int)
    parser.add_argument('--ngpus',help='use mult-gpu',default=0,type=int)
    args = parser.parse_args()
    return args

def parse_dataset_cfg(cfgfile):
    with open(cfgfile,'r') as fp:
        p1 = re.compile(r'classes=\d')
        p2 = re.compile(r'train=')
        p3 = re.compile(r'names=')
        p4 = re.compile(r'backup=')
        for line in fp.readlines():
            a = line.replace(' ','').replace('\n','')
            if p1.findall(a):
                classes = re.sub('classes=','',a)
            if p2.findall(a):
                trainlist = re.sub('train=','',a)
            if p3.findall(a):
                namesdir = re.sub('names=','',a)
            if p4.findall(a):
                backupdir = re.sub('backup=','',a)
    return int(classes),trainlist,namesdir,backupdir

def parse_network_cfg(cfgfile):
    with open(cfgfile,'r') as fp:
        layerList = []
        layerInfo = ''
        p = re.compile(r'\[\w+\]')
        p1 = re.compile(r'#.+')
        for line in fp.readlines():
            if p.findall(line):
                if layerInfo:
                    layerList.append(layerInfo)
                    layerInfo = ''
            if line == '\n' or p1.findall(line):
                continue
            line = line.replace(' ','')
            layerInfo += line
        layerList.append(layerInfo)
    print('layer number is %d'%(layerList.__len__() - 1) )
    return layerList


if __name__ == '__main__':
    args = parse_args()
    print(args)
    classes, trainlist, namesdir, backupdir = parse_dataset_cfg(args.dataset)
    print('%d classes in dataset'%classes)
    print('trainlist directory is ' + trainlist)
    #step 1: parse the network
    layerList = parse_network_cfg(args.netcfg)
    layer = []
    print('the depth of the network is %d'%(layerList.__len__()-1))
    network = net.network(layerList)
    criterion = loss.CostYoloV2(network.layers[-1].flow[0])
    #step 2: load network parameters
    network.load_weights(args.weight)
    layerNum = network.layerNum
    if args.cuda:
        if args.ngpus:
            print('use mult-gpu')
            network = nn.DataParallel(network).cuda() 
        else:
            network = network.cuda()
    #step 3: load data 
    dataset = dat.YoloDataset(trainlist,416,416)
    dataloader = data.DataLoader(dataset, batch_size=args.batch, shuffle=1)
    dataIter = iter(dataloader)
    #step 4: define optimizer
    optimizer = optim.Adam(network.parameters())
    #step 5: start train
    print('start training...')
    t_start = time.time()
    for i in range(5):
    #for i in range(network.max_batches):
        imgs, labels = next(dataIter)
        imgs = Variable( imgs, requires_grad=True)
        labels =  Variable(labels)
        if args.cuda:
            imgs =  imgs.cuda()
            labels =  labels.cuda()
        t0 = time.time()
        pred = network.forward(imgs)
        t1 = time.time()
        cost = criterion(pred, labels)
        t2 = time.time() 
        optimizer.zero_grad()
        cost.backward()
        t3 = time.time()
        
        print('forward time: %f, loss time: %f, backward time: %f'%((t1-t0),(t2-t1),(t3-t2)))
