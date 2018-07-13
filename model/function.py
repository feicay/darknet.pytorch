import torch
from torch import nn
from torch.nn import functional
from torch.autograd import Function

class reorg(nn.Module):
    def __init__(self, stride):
        self.stride = stride
    def forward(self, x):
        batch_size, channels, in_height, in_width = x.size()
        ChannelOut = channels*(self.stride)*(self.stride)
        out_height = in_height/self.stride
        out_width = in_width/self.stride
        input_view = x.contiguous().view(batch_size, channels, out_height, self.stride, out_width, self.stride)
        shuffle_out = input_view.permute(0,1,3,5,2,4).contiguous()
        return shuffle_out.view(batch_size, ChannelOut, out_height, out_width)
    def backward(self, x):
        batch_size, channels, in_height, in_width = x.size()
        ChannelOut = channels/(self.stride)/(self.stride)
        out_height = in_height*self.stride
        out_width = in_width*self.stride
        input_view = x.contiguous().view(batch_size, ChannelOut, self.stride, self.stride, in_height, in_width)
        shuffle_out = input_view.permute(0,1,4,2,5,3).contiguous()
        return shuffle_out.view(batch_size, ChannelOut, out_height, out_width)