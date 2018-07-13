import torch as t
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

class reorg(nn.Module):
    def __init__(self, stride):
        super(reorg, self).__init__()
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

#the output on the channel dim is [x,y,w,h,C,c0,c1,...,cn], the 1th dim of the Tensor(batch x channel x height x width) 
#(x,y,w,h) is the coords, C is the confidence of is_object, c0,c1 ... cn is the classes confidence
class region(nn.Module):
    def __init__(self, classes, coords, num, bias_match):
        super(region, self).__init__()
        self.classes = classes
        self.coords = coords
        self.num = num
        self.out_len = num*(coords + 1 + classes)
        self.bias_match = bias_match
    def forward(self, x):
        batch_size, channels, in_height, in_width = x.size()
        i = 0
        len_per_anchor = self.coords + 1 + self.classes
        for i in range(self.num):
            index = t.arange(i*len_per_anchor, (i+1)*len_per_anchor, 1)
            index = index.type(t.LongTensor)
            v = t.index_select(x, 1, index)
            #sigmod the object location(x,y)
            index1 = t.LongTensor([0,1])
            v1 = t.index_select(v, 1, index1)
            out1 = F.sigmoid(v1)
            #sigmod the object confidence
            index2 = t.LongTensor([4])
            v2 = t.index_select(v, 1, index2)
            out2 = F.sigmoid(v2)
            #softmax the classes confidence
            index3 = t.arange(5, len_per_anchor, 1)
            index3 = index3.type(t.LongTensor)
            v3 = t.index_select(v, 1, index3)
            out3 = F.softmax(v3, 1)
            out_buf = t.cat( (out1, out2, out3), 1)
            if i == 0:
                output = out_buf
            else:
                output = t.cat((output, out_buf), 1)
        return output
        

