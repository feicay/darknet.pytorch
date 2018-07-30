import torch
import torch.cuda
import argparse
import os
import sys
import cv2
import re
import math
import model.network as net
import model.eval as eva
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import time
import visdom
import numpy as np
from PIL import Image
from torchvision import transforms as T 

def parse_args():
    parser = argparse.ArgumentParser(description='train a network')
    parser.add_argument('--dataset',help='training set config file',default='dataset/coco.data',type=str)
    parser.add_argument('--netcfg',help='the network config file',default='cfg/yolov2.cfg',type=str)
    parser.add_argument('--weight',help='the network weight file',default='backup/yolov2.backup',type=str)
    parser.add_argument('--vis',help='visdom the training process',default=1,type=int)
    parser.add_argument('--img',help='the input file for detection',default='dog.jpg',type=str)
    parser.add_argument('--thresh',help='the input file for detection',default=0.5,type=float)
    parser.add_argument('--cuda',help='use the GPU',default=1,type=int)
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

def get_names(nameFile):
    with open(nameFile,'r') as fp:
        names = []
        for line in fp.readlines():
            line = line.replace(' ','').replace('\n','')
            if line != '':
                names.append(line)
    return names

def plot_boxes_cv2(image, boxes, class_names=None, color=None):
    img = cv2.imread(image)
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]])
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)
    width = img.shape[1]
    height = img.shape[0]
    num, _ = boxes.size()
    for i in range(num):
        box = boxes[i, :]
        x1 = int((box[1] - box[3]/2.0) * width)
        y1 = int((box[2] - box[4]/2.0) * height)
        x2 = int((box[1] + box[3]/2.0) * width)
        y2 = int((box[2] + box[4]/2.0) * height)
        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[6]
            cls_id = int(box[5])
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue  = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, class_names[cls_id], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
        img = cv2.rectangle(img, (x1,y1), (x2,y2), rgb, 1)
    savename = 'prediction.png'
    print("save plot results to %s" %savename)
    cv2.imwrite(savename, img)
    return img

def detect_image(image, network, thresh, names):
    pil_img = Image.open(image)
    transform = T.Compose([T.ToTensor(),T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
    img = pil_img.resize( (network.width, network.height) )
    img = transform(img).cuda()
    img = img.view(1,3,network.height,network.width)
    pred = network(img)
    evaluator = eva.evalYolov2(network.layers[-1].flow[0], obj_thresh=0.5, nms_thresh=0.45)
    result = evaluator.forward(pred)
    print(result)
    im = plot_boxes_cv2(image, result, names)
    #cv2.imshow('prediction',im)
    #cv2.waitKey(0)
    return 

if __name__ == '__main__':
    args = parse_args()
    print(args)
    classes, trainlist, namesdir, backupdir = parse_dataset_cfg(args.dataset)
    print('%d classes in dataset'%classes)
    print('trainlist directory is ' + trainlist)
    names = get_names(namesdir)
    #step 1: parse the network
    layerList = parse_network_cfg(args.netcfg)
    netname = args.netcfg.split('.')[0].split('/')[-1]
    layer = []
    print('the depth of the network is %d'%(layerList.__len__()-1))
    network = net.network(layerList)
    #step 2: load network parameters
    network.load_weights(args.weight)
    seen = network.seen
    #network.init_weights()
    layerNum = network.layerNum
    if args.cuda:
        network = network.cuda()
    #step 3: load data 
    image = args.img
    img_tail =  image.split('.')[-1] 
    if img_tail == 'jpg' or img_tail =='jpeg' or img_tail == 'png':
        detect_image(image, network, args.thresh, names)
        '''
    elif img_tail == 'mp4' or img_tail =='mkv' or img_tail == 'avi':
        detect_vedio(image, network)
        '''

