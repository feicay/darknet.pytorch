import torch as t
from torch import nn
from torch.nn import functional as F
from . import layer as l
import re

class network(nn.Module):
    def __init__(self, layerlist):
        super(network, self).__init__()
        ChannelIn = []
        ChannelOut = []
        width, height, self.channels, self.lr, self.momentum, self.decay, self.max_batches, self.burn_in, self.policy, self.steps, self.scales = make_input(layerlist[0], ChannelIn, ChannelOut)
        i = 1
        widthList = []
        heightList = []
        widthList.append(width)
        heightList.append(height)
        self.layers = []
        for i in range(layerlist.__len__()):
            layer = l.make_layer(layerlist[i], widthList, heightList, ChannelIn, ChannelOut, i-1)
            self.layers.append(layer)


def make_input(layercfg, ChannelIn, ChannelOut):
    line = layercfg.split('\n')
    p1 = re.compile(r'width=')
    p2 = re.compile(r'height=')
    p3 = re.compile(r'channels=')
    p4 = re.compile(r'learning_rate=')
    p5 = re.compile(r'momentum=')
    p6 = re.compile(r'decay=')
    p7 = re.compile(r'max_batches=')
    p8 = re.compile(r'burn_in=')
    p9 = re.compile(r'policy=')
    p10 = re.compile(r'steps=')
    p11 = re.compile(r'scales=')
    width = height = channels = max_batches = burn_in = 0
    lr = momentum = decay = 0.0
    policy = ''
    steps = []
    scales = []
    for info in line:
        if p1.findall(info):
            width = int( re.sub('width=','',info) )
        if p2.findall(info):
            height = int( re.sub('height=','',info) )
        if p3.findall(info):
            channels = int( re.sub('channels=','',info) )
        if p4.findall(info):
            lr = float( re.sub('learning_rate=','',info) )
        if p5.findall(info):
            momentum = float( re.sub('momentum=','',info) )
        if p6.findall(info):
            decay = float( re.sub('decay=','',info) )
        if p7.findall(info):
            max_batches = int( re.sub('max_batches=','',info) )
        if p8.findall(info):
            burn_in = int( re.sub('burn_in=','',info) )
        if p9.findall(info):
            policy = re.sub('policy=','',info)
        if p10.findall(info):
            steps_str = re.sub('steps=','',info).split(',')
            for s in steps_str:
                steps.append( int(s) )
        if p11.findall(info):
            scales_str = re.sub('scales=','',info).split(',')
            for s in scales_str:
                scales.append( float(s) )
    ChannelIn.append(0)
    ChannelOut.append(channels)
    return width, height, channels, lr, momentum, decay, max_batches, burn_in, policy, steps, scales