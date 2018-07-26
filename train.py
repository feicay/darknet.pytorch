import torch as t
import torch.cuda
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
import visdom
import numpy as np

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

def adjust_learning_rate(optimizer, batch, model, num_gpu):
    lr = model.lr
    for i in range(len(model.steps)):
        scale = model.scales[i] if i < len(model.scales) else 1
        if batch >= model.steps[i]:
            lr = model.lr * scale
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/model.batch*num_gpu
    print('learning rate: %f'%lr)
    return lr

if __name__ == '__main__':
    args = parse_args()
    print(args)
    classes, trainlist, namesdir, backupdir = parse_dataset_cfg(args.dataset)
    print('%d classes in dataset'%classes)
    print('trainlist directory is ' + trainlist)
    #step 1: parse the network
    layerList = parse_network_cfg(args.netcfg)
    netname = args.netcfg.split('.')[0].split('/')[-1]
    layer = []
    print('the depth of the network is %d'%(layerList.__len__()-1))
    network = net.network(layerList)
    criterion = loss.CostYoloV2(network.layers[-1].flow[0])
    max_batch = network.max_batches
    lr = network.lr / network.batch
    #step 2: load network parameters
    #network.load_weights(args.weight)
    network.init_weights()
    layerNum = network.layerNum
    if args.cuda:
        if args.ngpus:
            print('use mult-gpu')
            network = nn.DataParallel(network).cuda()   
            model = network.module
            num_gpu = torch.cuda.device_count()
            #criterion = nn.DataParallel(criterion).cuda()
        else:
            network = network.cuda()
            model = network
            num_gpu = 1
    #step 3: load data 
    dataset = dat.YoloDataset(trainlist,416,416)
    dataloader = data.DataLoader(dataset, batch_size=args.batch, shuffle=1)
    dataIter = iter(dataloader)
    #step 4: define optimizer
    optimizer = optim.Adam(network.parameters(),lr=lr*num_gpu)
    #step 5: start train
    print('start training...')
    t_start = time.time()
    #step 6 : initialize visdom board
    if args.vis:
        vis = visdom.Visdom(env=u'test1')
    for i in range(max_batch):
    #for i in range(network.max_batches):
        imgs, labels = next(dataIter)
        imgs = Variable( imgs )
        labels =  Variable(labels)
        if args.cuda:
            imgs =  imgs.cuda()
            #labels =  labels.cuda()
        #forward propagate
        optimizer.zero_grad()
        t0 = time.time()
        pred = network.forward(imgs)
        #calculate loss
        t1 = time.time()
        pred = pred.cpu()
        cost = criterion(pred, labels)
        cost = cost.cuda()
        #back propagate
        t2 = time.time() 
        cost.backward()
        #update parameters
        t3 = time.time()
        optimizer.step()
        t4 = time.time()
        model.seen += model.batch
        print('forward time: %f, loss time: %f, backward time: %f, update time: %f'%((t1-t0),(t2-t1),(t3-t2),(t4-t3)))
        if i % 500 == 0 and i > 0:
            weightname = backupdir + '/' + netname + '.backup'
            model.save_weights(weightname)
            adjust_learning_rate(optimizer, i, model, num_gpu)
        if args.vis:
            if args.cuda:
                loss = criterion.loss.cpu().data.view(1)
                loss_coords = criterion.loss_coords.cpu().data.view(1)
                loss_obj = criterion.loss_obj.cpu().data.view(1)
                loss_noobj = criterion.loss_noobj.cpu().data.view(1)
                loss_classes = criterion.loss_classes.cpu().data.view(1)
            else:
                loss = criterion.loss.data.view(1)
                loss_coords = criterion.loss_coords.data.view(1)
                loss_obj = criterion.loss_obj.data.view(1)
                loss_noobj = criterion.loss_noobj.data.view(1)
                loss_classes = criterion.loss_classes.data.view(1)
            if i > 0:
                vis.line(loss,X=np.array([i]),win='loss',update='append')
                vis.line(loss_obj,X=np.array([i]),win='loss_obj',update='append')
                vis.line(loss_noobj,X=np.array([i]),win='loss_noobj',update='append')
                vis.line(loss_coords,X=np.array([i]),win='loss_coords',update='append')
                vis.line(loss_classes,X=np.array([i]),win='loss_classes',update='append')
            else:
                vis.line(loss,X=np.array([0]),win='loss',opts=dict(title='loss'))
                vis.line(loss_obj,X=np.array([0]),win='loss_obj',opts=dict(title='obj_loss'))
                vis.line(loss_noobj,X=np.array([0]),win='loss_noobj',opts=dict(title='noobj_loss'))
                vis.line(loss_coords,X=np.array([0]),win='loss_coords',opts=dict(title='coords_loss'))
                vis.line(loss_classes,X=np.array([0]),win='loss_classes',opts=dict(title='classes_loss'))
