import torch
from torch import nn
from torch.nn import functional
from torch.autograd import Function
from torch.autograd import Variable
from . import function as F

def box_iou(box1, box2):
#box is a [x,y,w,h] tensor
    w = Variable(torch.Tensor([[1,0,-0.5,0],[0,1,0,-0.5],[1,0,0.5,0],[0,1,0,0.5]]))
    if box1.is_cuda:
        w = w.cuda()
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
        self.noobj = 0.0
        self.obj_pred_count = 0
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
    def forward(self, x, truth):
        batch_size, channels, in_height, in_width = x.size()
        self.seen += 1
        self.count = 0
        self.obj_pred_count = 0
        self.iou = 0.0
        self.recall = 0.0
        self.precision = 0.0
        self.class_precision = 0.0
        self.obj = 0.0
        self.noobj = 0.0
        self.loss_obj = 0.0
        self.loss_noobj = 0.0
        self.loss_coords = 0.0
        self.loss_classes = 0.0
        self.Aiou = 0.0
        self.AclassP = 0.0
        self.Aobj = 0.0
        self.Anoobj = 0.0
        self.AP = 0.0
        self.Arecall = 0.0
        for b in range(batch_size):
            x_b = x[b, :, :, :]
            truth_b = truth[b, :, :]
            #get the no object loss
            for j in range(in_height):
                for i in range(in_width):
                    for n in range(self.num):
                        idx = n * self.anchor_len
                        idx_end = idx + 4
                        box_pred = x_b[idx:idx_end, j, i]
                        obj_pred = x_b[4, j, i]
                        best_iou = 0.0
                        for t in range(50):
                            box_truth = truth_b[t, 0:4]
                            if box_truth[2] < 0.00001:
                                break
                            iou = box_iou(box_pred, box_truth)
                            if iou > best_iou:
                                best_iou = iou
                        if best_iou < self.thresh:
                            self.loss_noobj += (self.noobject_scale * (0 - obj_pred) ) ** 2
                            self.noobj += obj_pred
                        else:
                            self.obj_pred_count += 1
                        if self.seen < 500:
                            box_truth = Variable(torch.Tensor(4))
                            box_truth[0] = (i + 0.5)/in_width
                            box_truth[1] = (j + 0.5)/in_height
                            box_truth[2] = self.anchors[2*n]/in_width
                            box_truth[3] = self.anchors[2*n + 1]/in_height
                            tx = 0.5
                            ty = 0.5
                            tw = 0
                            th = 0
                            box_t = Variable(torch.Tensor([tx,ty,tw,th]))
                            if x.is_cuda:
                                box_t = box_t.cuda()
                            box_delta = box_t.sub(box_pred.view(4)) * 0.01
                            self.loss_coords += (box_delta**2).sum()
            for t in range(50):
                box_truth = truth_b[t, 0:4].view(4,1)
                class_truth = int( truth_b[t, 4].view(1) )
                if box_truth[2] < 0.00001:
                    break
                best_iou = 0.0
                best_n = 0
                i = int(box_truth[0] * in_width)
                j = int(box_truth[1] * in_height)
                #find the best iou for the current label
                box_truth_shift = box_truth
                box_truth_shift[0] = 0.0
                box_truth_shift[1] = 0.0
                for n in range(self.num):
                    box_pred = x_b[(n*self.anchor_len):(n*self.anchor_len+4), j, i].view(4,1)
                    box_pred_shift = box_pred
                    box_pred_shift[0] = 0.0
                    box_pred_shift[1] = 0.0
                    iou = box_iou(box_pred_shift, box_truth_shift)
                    if iou > best_iou:
                        best_iou = iou
                        best_n = n
                #calculate the coords loss
                loss_box_scale = self.loss_coords * (2 - box_truth[2]*box_truth[3])
                tx = box_truth[0] * in_width - i
                ty = box_truth[1] * in_height - j
                tw = torch.log( box_truth[2] * in_width / self.anchors[2*n] )
                th = torch.log( box_truth[3] * in_height / self.anchors[2*n + 1] )
                box_t = Variable(torch.Tensor([tx,ty,tw,th]))
                if x.is_cuda:
                    box_t = box_t.cuda()
                box_best = x_b[(best_n*self.anchor_len):(best_n*self.anchor_len+4), j, i].view(4,1)
                box_delta = box_t.sub(box_best.view(4)) * loss_box_scale
                self.loss_coords += (box_delta**2).sum()
                if iou > 0.5:
                    self.recall += 1
                self.iou += best_iou
                #calculate the object loss
                obj_pred = x_b[4][j][i].view(1)
                self.obj += obj_pred
                obj_delta = (1 - obj_pred) * self.loss_obj
                self.loss_obj += obj_delta**2
                #calculate the class loss
                classes_pred = x_b[(best_n*self.anchor_len + 5):((best_n+1)*self.anchor_len), j, i].view(self.classes)
                classes_truth = Variable(torch.zeros(self.classes))
                classes_truth[class_truth] = 1
                if x.is_cuda:
                    classes_truth = classes_truth.cuda()
                classes_delta = classes_truth.sub(classes_pred) * self.class_scale
                self.class_precision += classes_pred[class_truth]
                if classes_pred[class_truth] > 0.5:
                    self.precision += 1
                self.loss_classes += (classes_delta ** 2).sum()
                #use for statistic 
                self.count += 1
        if self.count != 0:
            self.loss = self.loss_obj + self.loss_noobj + self.loss_coords + self.loss_classes
            if self.obj_pred_count != 0:
                self.AP = self.precision/self.obj_pred_count
            self.Arecall = self.recall/self.count
            self.Aiou = self.iou/self.count
            self.AclassP = self.class_precision/self.count
            self.Aobj = self.obj/self.count
            self.Anoobj = self.noobj/ (in_height * in_width * self.num * batch_size)
        print('Average IoU: %5f, class: %5f, Obj: %5f, No obj: %5f, AP: %5f, Recall: %5f, count: %3d'%(self.Aiou \
                ,self.AclassP,self.Aobj,self.Anoobj,self.AP,self.Arecall,self.count) )
        return self.loss
