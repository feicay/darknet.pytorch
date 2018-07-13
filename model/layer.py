import torch
from torch import nn
from torch.nn import functional
from torch.autograd import Function
import re
import sys
from . import function as F

class Layer(nn.Module):
    def __init__(self, name, order, in_channel, out_channel, layers=None, l_in=-1, l_route=0, l_shortcut=0):
        super(Layer, self).__init__()
        self.name = name
        self.order = order
        self.in_channel = in_channel
        self.out_channel = out_channel
        if layers:
            self.flow = nn.Sequential(*layers)
        self.l_in = l_in
        self.l_route = l_route
        self.l_shortcut = l_shortcut
        self.input = None
        self.output = None
    def forward(self, x, Layers):       
        if self.l_route != 0:
            self.input = torch.cat( (Layers[self.order + self.l_in].output, Layers[self.order + self.l_route].output), 1)
        else:
            self.input = x
        if self.l_shortcut != 0:
            self.output = self.input + Layers[self.order + self.l_shortcut].output
        else:
            self.output = self.flow(self.input)
        return self.output

        
def make_conv_layers(cfglist, widthlist, heightlist, ChannelIn, ChannelOut, order):
    p1 = re.compile(r'batch_normalize=\d')
    p2 = re.compile(r'filters=')
    p3 = re.compile(r'size=')
    p4 = re.compile(r'stride=')
    p5 = re.compile(r'pad=')
    p6 = re.compile(r'activation=')
    layers = []
    pad = bn = size = padding_size = 0
    activation = ''
    for info in cfglist:
        if p1.findall(info):
            bn = int( re.sub('batch_normalize=','',info) )
        if p2.findall(info):
            out_channel = int( re.sub('filters=','',info) )
        if p3.findall(info):
            size = int( re.sub('size=','',info))
        if p4.findall(info):
            stride = int( re.sub('stride=','',info) )
        if p5.findall(info):
            pad = int( re.sub('pad=','',info))
        if p6.findall(info):
            activation = re.sub('activation=','',info)
    if pad == 1:
        padding_size = int(size/2)
    in_channel = ChannelOut[order]
    ChannelIn.append(in_channel)
    ChannelOut.append(out_channel)
    if bn:
        layers.append( nn.Conv2d(in_channel, out_channel, size, stride=stride, padding=padding_size, bias=False) )
        layers.append( nn.BatchNorm2d(out_channel) )
    else:
        layers.append( nn.Conv2d(in_channel, out_channel, size, stride=stride, padding=padding_size, bias=True) )
    if activation == 'leaky':
        layers.append( nn.LeakyReLU(negative_slope=0.1, inplace=True) )
    elif activation == 'relu':
        layers.append( nn.ReLU(inplace=True) )
    else:
        pass
    l_conv = Layer('conv', order, in_channel, out_channel, layers=layers)
    w_in = widthlist[order]
    w_out = (w_in + 2*padding_size - size)/stride + 1
    h_in = heightlist[order]
    h_out = (h_in + 2*padding_size - size)/stride + 1
    widthlist.append(w_out)
    heightlist.append(h_out)
    name = 'conv2d'
    print('%3d  %8s  %4d  %d x %d / %d  %4d x %4d x %4d  ->  %4d x %4d x %4d'%(order,name,out_channel,size,size,stride,w_in,h_in,in_channel, w_out,h_out,out_channel))
    return l_conv

def make_maxpool_layers(cfglist, widthlist, heightlist, ChannelIn, ChannelOut, order):
    p1 = re.compile(r'size=')
    p2 = re.compile(r'stride=')
    p3 = re.compile(r'pad=')
    size = stride = pad = 0
    for info in cfglist:
        if p1.findall(info):
            size = int( re.sub('size=','',info) )
        if p2.findall(info):
            stride = int( re.sub('stride=','',info) )
        if p3.findall(info):
            pad = int( re.sub('pad=','',info) )
    layers = []
    in_channel = ChannelOut[order]
    out_channel = in_channel
    ChannelIn.append(in_channel)
    ChannelOut.append(out_channel)
    w_in = widthlist[order]
    w_out = (w_in + 2*pad - size)/stride + 1
    h_in = heightlist[order]
    h_out = (h_in + 2*pad - size)/stride + 1
    widthlist.append(w_out)
    heightlist.append(h_out)
    name = 'maxpool'
    layers.append( nn.MaxPool2d(size, stride=stride, padding=pad) )
    l_pool = Layer('maxpool', order, in_channel, out_channel, layers=layers)
    print('%3d  %8s  %4d  %d / %d      %4d x %4d x %4d  ->  %4d x %4d x %4d'%(order,name,out_channel,size,stride,w_in,h_in,in_channel, w_out,h_out,out_channel))
    return l_pool

def make_reorg_layers(cfglist, widthlist, heightlist, ChannelIn, ChannelOut, order):
    p1 = re.compile(r'stride=')
    for info in cfglist:
        if p1.findall(info):
            stride = int( re.sub('stride=','',info) )
    layers = []
    in_channel = ChannelOut[order]
    ChannelIn.append(in_channel)
    out_channel = in_channel*stride*stride
    ChannelOut.append(out_channel)
    w_in = widthlist[order]
    w_out = w_in /stride 
    h_in = heightlist[order]
    h_out = h_in /stride
    widthlist.append(w_out)
    heightlist.append(h_out)
    name = 'reorg'
    layers.append( F.reorg(stride) )
    l_reorg = Layer('reorg', order, in_channel, out_channel, layers=layers)
    print('%3d  %8s  %4d  %d          %4d x %4d x %4d  ->  %4d x %4d x %4d'%(order,name,out_channel,stride,w_in,h_in,in_channel, w_out,h_out,out_channel))
    return l_reorg

def make_route_layers(cfglist, widthlist, heightlist, ChannelIn, ChannelOut, order):
    p1 = re.compile(r'layers=')
    for info in cfglist:
        if p1.findall(info):
            layers_str = re.sub('layers=','',info).split(',')
    if layers_str.__len__() == 1:
        l0 = int(layers_str[0])
        print_str = layers_str[0]
        if l0 > 0:
            l0 = l0 - order
        in_channel = ChannelOut[order + l0 + 1] 
        out_channel = in_channel
        ChannelIn.append(in_channel)
        ChannelOut.append(out_channel)
        l_route = Layer('route', order, in_channel, out_channel, l_in=l0 )
    elif layers_str.__len__() == 2:
        l0 = int(layers_str[0])
        if l0 > 0:
            l0 = l0 - order
        l1 = int(layers_str[1])
        if l1 > 0:
            l1 = l1 - order
        print_str = layers_str[0] + ',' + layers_str[1]
        in_channel = ChannelOut[order + l0 + 1] + ChannelOut[order + l1 + 1]
        out_channel = in_channel
        ChannelIn.append(in_channel)
        ChannelOut.append(out_channel)
        l_route = Layer('route',order, in_channel, out_channel, l_in=l0, l_route=l1 )
    else:
        print('error route layer parameter!')
        sys.exit(1)
    w_in = widthlist[order + l0 + 1]
    w_out = w_in
    h_in = heightlist[order + l0 + 1]
    h_out = h_in
    widthlist.append(w_out)
    heightlist.append(h_out)
    name = 'route'
    print('%3d  %8s  %4d  %6s     %4d x %4d x %4d  ->  %4d x %4d x %4d'%(order,name,out_channel,print_str,w_in,h_in,in_channel, w_out,h_out,out_channel))
    return l_route


def make_layer(layercfg, widthlist, heightlist, ChannelIn, ChannelOut, order):
    line = layercfg.split('\n')
    layer = None
    if line[0] == '[convolutional]':
        layer = make_conv_layers(line ,widthlist, heightlist, ChannelIn, ChannelOut, order)
    if line[0] == '[maxpool]':
        layer = make_maxpool_layers(line, widthlist, heightlist, ChannelIn, ChannelOut, order)
    if line[0] == '[reorg]':
        layer = make_reorg_layers(line, widthlist, heightlist, ChannelIn, ChannelOut, order)
    if line[0] == '[route]':
        layer = make_route_layers(line, widthlist, heightlist, ChannelIn, ChannelOut, order)
    return layer
