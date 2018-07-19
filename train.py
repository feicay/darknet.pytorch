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

def parse_args():
    parser = argparse.ArgumentParser(description='train a network')
    parser.add_argument('--dataset',help='training set config file',default='dataset/coco.data',type=str)
    parser.add_argument('--netcfg',help='the network config file',default='cfg/yolov2.cfg',type=str)
    parser.add_argument('--weight',help='the network weight file',default='weight/yolov2_final.weight',type=str)
    parser.add_argument('--batch',help='training batch size',default=64,type=int)
    parser.add_argument('--vis',help='visdom the training process',default=1,type=int)
    parser.add_argument('--cuda',help='use the GPU',default=1,type=int)
    parser.add_argument('--ngpus',help='use mult-gpu',default=1,type=int)
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
    print(network.layers[-1].flow[0].anchors)
    criterion = loss.lossYoloV2(network.layers[-1].flow[0])
    #step 2: load network parameters
    network.load_weights(args.weight)
    if args.cuda:
        for i in range(network.layerNum):
                if network.layers[i].name == 'conv' or network.layers[i].name == 'region' or network.layers[i].name == 'yolo':
                    network.layers[i].flow = network.layers[i].flow.cuda()
                network.layers[i] = network.layers[i].cuda()
        network = network.cuda()
        if args.ngpus:
            print('use mult-gpu')
            network = nn.DataParallel(network, device_ids=[0,1,2,3])
            
        criterion = criterion.cuda()
    #step 3: load data 
    dataset = dat.YoloDataset(trainlist,416,416)
    dataloader = data.DataLoader(dataset, batch_size=args.batch, shuffle=1)
    dataIter = iter(dataloader)
    
    '''
    for i in range(100):
        imgs, labels = next(dataIter)
        print(i)
        print(imgs.size())
        print(labels.size())
    '''
    #step 4: start train
    for i in range(10):
    #for i in range(network.max_batches):
        imgs, labels = next(dataIter)
        if args.cuda:
            imgs = imgs.cuda()
            labels = labels.cuda()
        pred = network.forward(imgs)
        loss = criterion(pred, labels)
