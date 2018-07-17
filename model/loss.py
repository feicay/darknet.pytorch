import torch
from torch import nn
from torch.nn import functional
from torch.autograd import Function
from torch.autograd import Variable
from . import function as F

def box_iou(box1, box2):
#box is a [x,y,w,h] tensor
    w = Variable(torch.Tensor([[1,0,-0.5,0],[0,1,0,-0.5],[1,0,0.5,0],[0,1,0,0.5]]))
    box_a = w.mm(box1.view(4,1))
    box_b = w.mm(box2.view(4,1))
    left = torch.max(box_a[0], box_b[0])
    right = torch.min(box_a[2], box_b[2])
    up = torch.max(box_a[1], box_b[1])
    down = torch.min(box_a[3], box_b[3])
    intersection_w = right - left
    intersection_h = down - up
    if intersection_w < 0 or intersection_h < 0:
        return 0
    else:
        intersection = intersection_w * intersection_h
        union = box1[2] * box1[3] + box2[2] * box2[3] - intersection
        return intersection/union

class lossYoloV2(nn.Module):
    def __init__(self, RegionLayer):
        super(lossYoloV2, self).__init__()
        self.count = 0
        self.iou = 0.0
        self.recall = 0.0
        self.precision = 0.0
        self.obj = 0.0
        self.loss_obj = 0.0
        self.loss_noobj = 0.0
        self.loss_coords = 0.0
        self.loss_classes = 0.0
        self.loss = 0.0
        self.anchors = RegionLayer.anchors
        self.object_scale = RegionLayer.object_scale
        self.noobject_scale = RegionLayer.noobject_scale
        self.class_scale = RegionLayer.class_scale
        self.coord_scale = RegionLayer.coord_scale
        self.classes = RegionLayer.classes
        self.coords = RegionLayer.coords
        self.num = RegionLayer.num
        self.anchor_len = self.classes + self.coords + 1
        self.thresh = 0.5
        self.seen = 0
        print(self.num)
    def forward(self, x, truth):
        batch_size, channels, in_height, in_width = x.size()
        self.seen += 1
        self.loss_obj = 0.0
        self.loss_noobj = 0.0
        self.loss_coords = 0.0
        self.loss_classes = 0.0
        for b in range(batch_size):
            x_b = x[b][:][:][:]
            truth_b = truth[b][:][:]
            #get the no object loss
            for j in range(in_height):
                for i in range(in_width):
                    for n in range(self.num):
                        box_pred = x_b[0][(n*self.anchor_len):(n*self.anchor_len+4)][j][i]
                        obj_pred = x_b[0][4][j][i]
                        best_iou = 0.0
                        for t in range(30):
                            box_truth = truth_b[0][0:4][t]
                            if box_truth[2] < 0.00001:
                                break
                            iou = box_iou(box_pred.view(4,1), box_truth.view(4,1))
                            if iou > best_iou:
                                best_iou = iou
                        if best_iou < self.thresh:
                            self.loss_noobj += (self.noobject_scale * (0 - obj_pred) ) ** 2
                        if self.seen < 500:
                            box_truth = Variable(torch.Tensor(4))
                            box_truth[0] = (i + 0.5)/in_width
                            box_truth[1] = (j + 0.5)/in_height
                            box_truth[2] = self.anchors[2*n]/in_width
                            box_truth[3] = self.anchors[2*n + 1]/in_height

        self.loss = self.loss_obj + self.loss_noobj + self.loss_coords + self.loss_classes
        return self.loss
